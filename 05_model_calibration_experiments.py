"""Compare OPS features, estimators, and calibration using development data only."""

import importlib.util
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from common import (
    BASE_FEATS,
    OPS_FEATS,
    OUT_DIR,
    SPLIT_FIT_END,
    pinball,
)

QUANTILES = (0.5, 0.9, 0.95)
MIN_PEAK_CALIBRATION_ROWS = 100
BASELINE_OPS_FEATS = BASE_FEATS + ["lag_24", "lag_168", "roll24"]
FEATURE_SETS = {
    "baseline_ops": BASELINE_OPS_FEATS,
    "enhanced_ops": OPS_FEATS,
}
FOLDS = (
    {
        "name": "dev_1",
        "train_end": "2011-05-01",
        "calibration_start": "2011-05-01",
        "calibration_end": "2011-07-01",
        "validation_start": "2011-07-01",
        "validation_end": "2011-09-01",
    },
    {
        "name": "dev_2",
        "train_end": "2011-09-01",
        "calibration_start": "2011-09-01",
        "calibration_end": "2011-11-01",
        "validation_start": "2011-11-01",
        "validation_end": "2012-01-01",
    },
    {
        "name": "dev_3",
        "train_end": "2012-01-01",
        "calibration_start": "2012-01-01",
        "calibration_end": "2012-03-01",
        "validation_start": "2012-03-01",
        "validation_end": "2012-05-01",
    },
    {
        "name": "dev_4",
        "train_end": "2012-03-01",
        "calibration_start": "2012-03-01",
        "calibration_end": "2012-05-01",
        "validation_start": "2012-05-01",
        "validation_end": "2012-06-01",
    },
)
PEAK_WINDOWS = {
    "morning_peak": (7, 9),
    "evening_peak": (16, 19),
}


def load_development_rows():
    """Read only rows strictly before June 2012; never load the final test period."""
    path = os.path.join(OUT_DIR, "features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Run 01_data_quality_and_features.py before this experiment.")

    start = pd.read_csv(path, usecols=["ts"], nrows=1, parse_dates=["ts"]).iloc[0, 0]
    row_limit = int((SPLIT_FIT_END - start) / pd.Timedelta(hours=1))
    data = pd.read_csv(path, nrows=row_limit, parse_dates=["ts", "dteday"])
    if data.empty or data["ts"].max() >= SPLIT_FIT_END:
        raise ValueError("The development-only reader crossed the June 2012 cutoff.")
    if not data["ts"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("features.csv must be a complete, hourly chronological grid.")
    required = set(BASELINE_OPS_FEATS + OPS_FEATS)
    if not required.issubset(data.columns):
        raise ValueError(
            "features.csv is missing experiment features; rerun 01_data_quality_and_features.py."
        )
    return data


def point_model_candidates():
    """Reuse the point-regressor configurations already used by the project."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "03_fleet_model.py")
    spec = importlib.util.spec_from_file_location("bike_fleet_model", path)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load model candidates from 03_fleet_model.py.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.model_candidates


def finite_sample_offset(residuals, quantile):
    """Use the finite-sample corrected one-based conformal order statistic."""
    values = np.asarray(residuals, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot calibrate a quantile with no finite residuals.")
    rank = min(int(np.ceil((values.size + 1) * quantile)), values.size)
    return float(np.sort(values)[rank - 1])


def peak_masks(frame):
    workday = frame["workingday"].eq(1)
    return {
        name: workday & frame["hr"].between(start, end)
        for name, (start, end) in PEAK_WINDOWS.items()
    }


def fit_calibration_offsets(raw_cal, calibration):
    y = calibration["bikes_in_use"].to_numpy(dtype=float)
    masks = peak_masks(calibration)
    offsets = {}
    rows = []
    for quantile in QUANTILES:
        residuals = y - raw_cal[quantile]
        global_offset = finite_sample_offset(residuals, quantile)
        offsets[(quantile, "global")] = global_offset
        rows.append({
            "calibration_strategy": "global",
            "calibration_scope": "global",
            "quantile": quantile,
            "sample_count": int(len(residuals)),
            "offset": global_offset,
            "local_offset_applied": True,
            "fallback": False,
        })
        rows.append({
            "calibration_strategy": "peak_aware",
            "calibration_scope": "off_peak_global_fallback",
            "quantile": quantile,
            "sample_count": int(len(residuals)),
            "offset": global_offset,
            "local_offset_applied": False,
            "fallback": False,
        })
        for name, mask in masks.items():
            count = int(mask.sum())
            fallback = count < MIN_PEAK_CALIBRATION_ROWS
            offset = (
                global_offset
                if fallback
                else finite_sample_offset(residuals[mask.to_numpy()], quantile)
            )
            offsets[(quantile, name)] = offset
            rows.append({
                "calibration_strategy": "peak_aware",
                "calibration_scope": name,
                "quantile": quantile,
                "sample_count": count,
                "offset": offset,
                "local_offset_applied": not fallback,
                "fallback": fallback,
            })
    return offsets, rows


def apply_calibration(raw, frame, offsets, strategy):
    adjusted = []
    for quantile in QUANTILES:
        values = raw[quantile].copy()
        values += offsets[(quantile, "global")]
        if strategy == "peak_aware":
            for name, mask in peak_masks(frame).items():
                values[mask.to_numpy()] = (
                    raw[quantile][mask.to_numpy()] + offsets[(quantile, name)]
                )
        adjusted.append(np.clip(values, 0, None))
    matrix = np.vstack(adjusted)
    crossed = (matrix[0] > matrix[1]) | (matrix[1] > matrix[2])
    monotone = np.maximum.accumulate(matrix, axis=0)
    predictions = {
        quantile: monotone[index]
        for index, quantile in enumerate(QUANTILES)
    }
    return predictions, crossed


def make_candidate_specs():
    specs = []
    for estimator in ("RandomForestRegressor", "HistGradientBoostingRegressor", "ExtraTreesRegressor"):
        for feature_set, features in FEATURE_SETS.items():
            specs.append({
                "estimator": estimator,
                "feature_set": feature_set,
                "kind": "point",
                "features": features,
            })
    for feature_set, features in FEATURE_SETS.items():
        specs.append({
            "estimator": "HistGradientBoostingRegressor (native quantile)",
            "feature_set": feature_set,
            "kind": "quantile",
            "features": features,
        })
    specs.extend([
        {
            "estimator": "Seasonal naive: same hour one week earlier",
            "feature_set": "seasonal_naive",
            "kind": "lag_168",
            "features": [],
        },
        {
            "estimator": "Seasonal naive: training mean by hour and working-day flag",
            "feature_set": "seasonal_naive",
            "kind": "profile",
            "features": [],
        },
    ])
    return specs


def raw_predictions(spec, train, calibration, validation, model_candidates):
    kind = spec["kind"]
    if kind == "point":
        model = model_candidates()[spec["estimator"]]
        features = spec["features"]
        model.fit(train[features], train["bikes_in_use"])
        cal_point = model.predict(calibration[features])
        val_point = model.predict(validation[features])
        return (
            {q: cal_point.copy() for q in QUANTILES},
            {q: val_point.copy() for q in QUANTILES},
        )
    if kind == "quantile":
        features = spec["features"]
        cal_predictions = {}
        val_predictions = {}
        for quantile in QUANTILES:
            model = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=quantile,
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=25,
                l2_regularization=1.0,
                random_state=0,
            )
            model.fit(train[features], train["bikes_in_use"])
            cal_predictions[quantile] = model.predict(calibration[features])
            val_predictions[quantile] = model.predict(validation[features])
        return cal_predictions, val_predictions

    if kind == "lag_168":
        cal_point = calibration["lag_168"].to_numpy(dtype=float)
        val_point = validation["lag_168"].to_numpy(dtype=float)
    elif kind == "profile":
        profile = train.groupby(["hr", "workingday"])["bikes_in_use"].mean()
        fallback = float(train["bikes_in_use"].mean())

        def predict_profile(frame):
            keys = pd.MultiIndex.from_frame(frame[["hr", "workingday"]])
            result = keys.map(profile).to_numpy(dtype=float)
            return np.where(np.isfinite(result), result, fallback)

        cal_point = predict_profile(calibration)
        val_point = predict_profile(validation)
    else:
        raise ValueError(f"Unknown experiment candidate type: {kind}")
    return (
        {q: cal_point.copy() for q in QUANTILES},
        {q: val_point.copy() for q in QUANTILES},
    )


def add_validation_segments(frame):
    result = frame.copy()
    result["peak_period"] = "off_peak"
    workday = result["workingday"].eq(1)
    for name, (start, end) in PEAK_WINDOWS.items():
        result.loc[workday & result["hr"].between(start, end), "peak_period"] = name
    result["day_type"] = np.where(
        result["workingday"].eq(1), "working_day", "non_working_day"
    )
    result["source_season"] = result["season"].astype(int).astype(str)
    return result


def metric_row(frame, subset_type, subset):
    y = frame["observed"].to_numpy(dtype=float)
    p50 = frame["p50"].to_numpy(dtype=float)
    denominator = np.abs(y).sum()
    row = {
        "slice_type": subset_type,
        "slice": subset,
        "n": int(len(frame)),
        "MAE": float(mean_absolute_error(y, p50)),
        "RMSE": float(mean_squared_error(y, p50) ** 0.5),
        "WAPE": float(np.abs(y - p50).sum() / denominator) if denominator else np.nan,
        "R2": float(r2_score(y, p50)) if len(y) > 1 and np.var(y) > 0 else np.nan,
        "mean_error_y_minus_P50": float(np.mean(y - p50)),
        "underprediction_rate_P50": float(np.mean(y > p50)),
        "underprediction_rate_P90": float(np.mean(y > frame["p90"].to_numpy(dtype=float))),
        "underprediction_rate_P95": float(np.mean(y > frame["p95"].to_numpy(dtype=float))),
        "crossed_rows_before_projection": int(frame["crossed_before_projection"].sum()),
        "crossing_rate_before_projection": float(frame["crossed_before_projection"].mean()),
    }
    for quantile in QUANTILES:
        prediction = frame[f"p{int(quantile * 100)}"].to_numpy(dtype=float)
        row[f"coverage_P{int(quantile * 100)}"] = float(np.mean(y <= prediction))
        row[f"pinball_P{int(quantile * 100)}"] = pinball(y, prediction, quantile)
    return row


def build_metrics(predictions):
    dimensions = {
        "peak_period": "peak_period",
        "day_type": "day_type",
        "season": "source_season",
    }
    rows = []

    for keys, candidate in predictions.groupby(
        ["estimator", "feature_set", "calibration"], sort=False
    ):
        estimator, feature_set, calibration = keys
        windows = [("pooled_validation", candidate)]
        windows.extend(
            (str(fold), group)
            for fold, group in candidate.groupby("fold", sort=True)
        )
        for fold_name, window in windows:
            groups = [("overall", "all", window)]
            for dimension, column in dimensions.items():
                groups.extend(
                    (dimension, str(name), part)
                    for name, part in window.groupby(column, sort=True)
                )
            for slice_type, slice_name, group in groups:
                row = metric_row(group, slice_type, slice_name)
                row.update({
                    "fold": fold_name,
                    "estimator": estimator,
                    "feature_set": feature_set,
                    "calibration": calibration,
                })
                rows.append(row)
    return pd.DataFrame(rows)


def feature_set_comparison(metrics):
    overall = metrics[
        metrics["fold"].eq("pooled_validation")
        & metrics["slice_type"].eq("overall")
        & metrics["calibration"].eq("global")
        & metrics["feature_set"].isin(FEATURE_SETS)
    ]
    metric_names = [
        "n", "MAE", "RMSE", "WAPE", "R2", "coverage_P50", "pinball_P50",
        "coverage_P90", "pinball_P90", "coverage_P95", "pinball_P95",
    ]
    wide = overall.pivot(index="estimator", columns="feature_set", values=metric_names)
    result = pd.DataFrame(index=wide.index)
    for metric in metric_names:
        baseline = wide[(metric, "baseline_ops")]
        enhanced = wide[(metric, "enhanced_ops")]
        result[f"baseline_{metric}"] = baseline
        result[f"enhanced_{metric}"] = enhanced
        result[f"enhanced_minus_baseline_{metric}"] = enhanced - baseline
    return result.reset_index()


def calibration_comparison(metrics):
    pooled = metrics[
        metrics["fold"].eq("pooled_validation")
        & metrics["slice_type"].isin(["overall", "peak_period", "day_type", "season"])
    ]
    metric_names = [
        "n", "MAE", "RMSE", "coverage_P90", "pinball_P90",
        "coverage_P95", "pinball_P95", "mean_error_y_minus_P50",
        "underprediction_rate_P50", "crossing_rate_before_projection",
    ]
    wide = pooled.pivot(
        index=["estimator", "feature_set", "slice_type", "slice"],
        columns="calibration",
        values=metric_names,
    )
    result = pd.DataFrame(index=wide.index)
    for metric in metric_names:
        global_values = wide[(metric, "global")]
        peak_values = wide[(metric, "peak_aware")]
        result[f"global_{metric}"] = global_values
        result[f"peak_aware_{metric}"] = peak_values
        result[f"peak_aware_minus_global_{metric}"] = peak_values - global_values
    return result.reset_index()


def make_plots(metrics, predictions):
    global_metrics = metrics[
        metrics["fold"].eq("pooled_validation")
        & metrics["slice_type"].eq("overall")
        & metrics["calibration"].eq("global")
    ].copy()
    global_metrics["label"] = global_metrics.apply(
        lambda row: f"{row.estimator}\n{row.feature_set}", axis=1
    )
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, metric in zip(axes, ("MAE", "RMSE")):
        ordered = global_metrics.sort_values(metric)
        colors = [
            "#4472C4" if feature_set == "baseline_ops"
            else "#70AD47" if feature_set == "enhanced_ops"
            else "#A5A5A5"
            for feature_set in ordered["feature_set"]
        ]
        ax.barh(ordered["label"], ordered[metric], color=colors)
        ax.invert_yaxis()
        ax.set(title=f"Validation {metric} (global calibration)", xlabel=metric)
    fig.suptitle("Development-only estimator and feature comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_04_validation_model_comparison.png"), dpi=160)
    plt.close(fig)

    anchor_name = "HistGradientBoostingRegressor (native quantile)"
    anchor = metrics[
        metrics["fold"].eq("pooled_validation")
        & metrics["slice_type"].eq("peak_period")
        & metrics["estimator"].eq(anchor_name)
        & metrics["feature_set"].eq("enhanced_ops")
        & metrics["slice"].isin(["morning_peak", "evening_peak", "off_peak"])
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for ax, coverage, target in zip(axes, ("coverage_P90", "coverage_P95"), (0.9, 0.95)):
        pivot = anchor.pivot(index="slice", columns="calibration", values=coverage)
        pivot = pivot.reindex(["morning_peak", "evening_peak", "off_peak"])
        pivot.plot(kind="bar", ax=ax, color=["#4472C4", "#ED7D31"])
        ax.axhline(target, color="black", linestyle="--", linewidth=1)
        ax.set(title=f"{coverage.replace('_', ' ')} (target {target:.0%})", ylabel="Observed coverage")
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Global vs peak-aware calibration: enhanced OPS quantile model")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_05_validation_calibration.png"), dpi=160)
    plt.close(fig)

    anchor_predictions = predictions[
        predictions["estimator"].eq(anchor_name)
        & predictions["feature_set"].eq("enhanced_ops")
        & predictions["calibration"].isin(["global", "peak_aware"])
    ]
    anchor_predictions["error_y_minus_P50"] = (
        anchor_predictions["observed"] - anchor_predictions["p50"]
    )
    anchor_predictions["underprediction_P50"] = (
        anchor_predictions["observed"] > anchor_predictions["p50"]
    )
    hourly = anchor_predictions.groupby(
        ["calibration", "hr"], as_index=False
    ).agg(
        mean_error_y_minus_P50=("error_y_minus_P50", "mean"),
        underprediction_rate_P50=("underprediction_P50", "mean"),
    )
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for calibration in ("global", "peak_aware"):
        subset = hourly[hourly["calibration"].eq(calibration)]
        ax.plot(subset["hr"], subset["mean_error_y_minus_P50"], marker="o", label=calibration)
    ax.axhline(0, color="black", linewidth=1)
    ax.set(
        title="Validation P50 bias by hour (positive means underprediction)",
        xlabel="Hour",
        ylabel="Mean observed minus predicted",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_06_validation_hourly_bias.png"), dpi=160)
    plt.close(fig)


def main():
    model_candidates = point_model_candidates()
    data = load_development_rows()
    eligible = data[
        data["is_observed"]
        & data["outage_day"].eq(0)
        & data["bikes_in_use"].notna()
    ].dropna(subset=BASELINE_OPS_FEATS + OPS_FEATS).copy()
    if eligible.empty:
        raise ValueError("No development rows have complete baseline and enhanced OPS features.")

    candidate_specs = make_candidate_specs()
    prediction_frames = []
    calibration_rows = []
    fold_counts = {}
    for fold in FOLDS:
        train = eligible[eligible["ts"] < pd.Timestamp(fold["train_end"])]
        calibration = eligible[
            (eligible["ts"] >= pd.Timestamp(fold["calibration_start"]))
            & (eligible["ts"] < pd.Timestamp(fold["calibration_end"]))
        ]
        validation = eligible[
            (eligible["ts"] >= pd.Timestamp(fold["validation_start"]))
            & (eligible["ts"] < pd.Timestamp(fold["validation_end"]))
        ]
        if min(len(train), len(calibration), len(validation)) == 0:
            raise ValueError(f"Fold {fold['name']} has an empty train/calibration/validation split.")
        if train["ts"].max() >= calibration["ts"].min():
            raise ValueError(f"Fold {fold['name']} training and calibration overlap.")
        if calibration["ts"].max() >= validation["ts"].min():
            raise ValueError(f"Fold {fold['name']} calibration and validation overlap.")
        if validation["ts"].max() >= SPLIT_FIT_END:
            raise ValueError(f"Fold {fold['name']} crossed the development cutoff.")
        fold_counts[fold["name"]] = {
            "train_rows": int(len(train)),
            "calibration_rows": int(len(calibration)),
            "validation_rows": int(len(validation)),
            "train_start": train["ts"].min().isoformat(),
            "train_end": train["ts"].max().isoformat(),
            "calibration_start": calibration["ts"].min().isoformat(),
            "calibration_end": calibration["ts"].max().isoformat(),
            "validation_start": validation["ts"].min().isoformat(),
            "validation_end": validation["ts"].max().isoformat(),
            "calibration_peak_rows": {
                name: int(mask.sum()) for name, mask in peak_masks(calibration).items()
            },
            "validation_peak_rows": {
                name: int(mask.sum()) for name, mask in peak_masks(validation).items()
            },
        }

        for spec in candidate_specs:
            raw_cal, raw_val = raw_predictions(spec, train, calibration, validation, model_candidates)
            offsets, offset_rows = fit_calibration_offsets(raw_cal, calibration)
            for row in offset_rows:
                row.update({
                    "fold": fold["name"],
                    "estimator": spec["estimator"],
                    "feature_set": spec["feature_set"],
                    "train_end_exclusive": fold["train_end"],
                    "calibration_start": fold["calibration_start"],
                    "calibration_end_exclusive": fold["calibration_end"],
                })
                calibration_rows.append(row)
            for strategy in ("global", "peak_aware"):
                calibrated, crossed = apply_calibration(raw_val, validation, offsets, strategy)
                result = validation[
                    ["ts", "dteday", "hr", "workingday", "season", "bikes_in_use"]
                ].copy()
                result = add_validation_segments(result)
                result = result.rename(columns={"bikes_in_use": "observed"})
                result["fold"] = fold["name"]
                result["estimator"] = spec["estimator"]
                result["feature_set"] = spec["feature_set"]
                result["calibration"] = strategy
                result["crossed_before_projection"] = crossed
                for quantile, values in calibrated.items():
                    result[f"p{int(quantile * 100)}"] = values
                prediction_frames.append(result)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = build_metrics(predictions)
    comparison = feature_set_comparison(metrics)
    calibration_comparison_table = calibration_comparison(metrics)
    estimator_comparison = metrics[
        metrics["fold"].eq("pooled_validation")
        & metrics["slice_type"].eq("overall")
        & metrics["calibration"].eq("global")
    ].sort_values(["MAE", "RMSE"])
    offsets = pd.DataFrame(calibration_rows)

    metrics.to_csv(os.path.join(OUT_DIR, "development_validation_metrics.csv"), index=False)
    comparison.to_csv(os.path.join(OUT_DIR, "development_feature_set_comparison.csv"), index=False)
    calibration_comparison_table.to_csv(
        os.path.join(OUT_DIR, "development_calibration_comparison.csv"), index=False
    )
    estimator_comparison.to_csv(
        os.path.join(OUT_DIR, "development_estimator_comparison.csv"), index=False
    )
    offsets.to_csv(os.path.join(OUT_DIR, "development_calibration_offsets.csv"), index=False)
    predictions.to_csv(
        os.path.join(OUT_DIR, "development_validation_predictions.csv"), index=False
    )

    point_models = model_candidates()
    config = {
        "evaluation_scope": "Development and chronological validation only; no final test rows loaded.",
        "data_cutoff_exclusive": SPLIT_FIT_END.isoformat(),
        "last_loaded_timestamp": data["ts"].max().isoformat(),
        "development_rows_loaded": int(len(data)),
        "candidate_training_rows": int(len(eligible)),
        "folds": fold_counts,
        "estimators": {
            name: {key: value for key, value in model.get_params().items()}
            for name, model in point_models.items()
        },
        "quantile_estimator": {
            "name": "HistGradientBoostingRegressor (native quantile)",
            "parameters": {
                "loss": "quantile",
                "quantiles": list(QUANTILES),
                "max_iter": 300,
                "learning_rate": 0.05,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 25,
                "l2_regularization": 1.0,
                "random_state": 0,
            },
        },
        "feature_sets": {
            "baseline_ops": BASELINE_OPS_FEATS,
            "enhanced_ops": OPS_FEATS,
            "comparison_uses_common_complete_rows": True,
        },
        "seasonal_naive_baselines": [
            "same hour one week earlier (lag_168)",
            "training-period mean by hour and working-day flag",
        ],
        "xgboost_available": importlib.util.find_spec("xgboost") is not None,
        "lightgbm_available": importlib.util.find_spec("lightgbm") is not None,
        "peak_definitions": {
            "morning_peak": "workingday == 1 and hour in 07-09",
            "evening_peak": "workingday == 1 and hour in 16-19",
            "off_peak": "all rows not in morning_peak or evening_peak",
        },
        "peak_calibration": {
            "minimum_rows_per_calibration_peak": MIN_PEAK_CALIBRATION_ROWS,
            "fallback": "Use the global offset for a peak when its calibration count is below the minimum.",
        },
        "finite_sample_offset": (
            "For q, sort calibration residuals and use the one-based rank "
            "min(ceil((n+1)*q), n)."
        ),
        "crossing_handling": (
            "Measure crossings after calibration and before monotone projection; "
            "apply cumulative maximum before scoring."
        ),
        "limitations": [
            "Validation estimates are for modelled average concurrent use, not observed station inventory.",
            "Finite-sample ranks do not guarantee conditional coverage under serial dependence or distribution shift.",
            "Hyperparameters are held at existing project values; no test-set tuning is performed.",
        ],
    }
    with open(os.path.join(OUT_DIR, "development_experiment_config.json"), "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    make_plots(metrics, predictions)
    print("Development-only fold setup:\n", json.dumps(fold_counts, indent=2))
    print("\nOverall pooled validation comparison (global calibration):\n")
    print(estimator_comparison.round(3).to_string(index=False))
    print("\nFeature-set comparison (enhanced minus baseline):\n")
    print(comparison.round(3).to_string(index=False))
    print("\nValidation metric slices:\n")
    print(
        metrics[
            metrics["fold"].eq("pooled_validation")
            & metrics["slice_type"].isin(["peak_period", "day_type", "season"])
        ].round(3).to_string(index=False)
    )
    print("\nCalibration offsets saved to outputs/development_calibration_offsets.csv")
    print("No data at or after 2012-06-01 was loaded or evaluated.")


if __name__ == "__main__":
    main()
