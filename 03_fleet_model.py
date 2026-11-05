"""Chronological bike-use forecasting, model comparison, and fleet scenarios."""

import json
import hashlib
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    BUFFER, CAS_TRIP_MIN, OPS_FEATS, OPS_HISTORY_CUTOFF_HOURS,
    OPS_ROLLING_MIN_PERIODS, OPS_ROLLING_WINDOW_HOURS, OUT_DIR, PLAN_FEATS, REG_TRIP_MIN,
    SPLIT_CAL_END, SPLIT_FIT_END, build_features, complete_grid,
    fit_quantile_bundle, fleet_from_bikes_in_use, load_raw, pinball,
    predict_quantiles,
)

import joblib

from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SELECTION_END = pd.Timestamp("2012-04-01")
QUANTILES = (0.5, 0.9, 0.95)
FROZEN_QUANTILE_ESTIMATOR = "HistGradientBoostingRegressor (native quantile)"
FROZEN_CALIBRATION_METHOD = "peak_aware"
MIN_PEAK_CALIBRATION_ROWS = 100


def regression_metrics(y, prediction):
    """Return point-forecast metrics, guarding zero-volume WAPE."""
    y = np.asarray(y)
    prediction = np.asarray(prediction)
    denominator = np.abs(y).sum()
    return {
        "MAE": float(mean_absolute_error(y, prediction)),
        "RMSE": float(mean_squared_error(y, prediction) ** 0.5),
        "WAPE": float(np.abs(y - prediction).sum() / denominator) if denominator else None,
        "R2": float(r2_score(y, prediction)) if len(y) > 1 else None,
    }


def quantile_metrics(y, predictions):
    metrics = regression_metrics(y, predictions[0.5])
    for q, prediction in predictions.items():
        metrics[f"coverage_P{int(q * 100)}"] = float(np.mean(np.asarray(y) <= prediction))
        metrics[f"pinball_P{int(q * 100)}"] = pinball(y, prediction, q)
    return metrics


def ops_segment_metrics(frame, predictions):
    """Evaluate OPS forecasts overall and by the frozen rush-hour definitions."""
    working_day = frame["workingday"].eq(1)
    segments = {
        "overall": pd.Series(True, index=frame.index),
        "morning_peak": working_day & frame["hr"].between(7, 9),
        "evening_peak": working_day & frame["hr"].between(16, 19),
    }
    peak_rows = segments["morning_peak"] | segments["evening_peak"]
    segments["off_peak"] = ~peak_rows
    rows = []
    for name, mask in segments.items():
        positions = np.flatnonzero(mask.to_numpy())
        if not len(positions):
            continue
        y = frame.iloc[positions]["bikes_in_use"].to_numpy()
        subset_predictions = {
            q: values[positions] for q, values in predictions.items()
        }
        rows.append({
            "segment": name,
            "n": int(len(positions)),
            **quantile_metrics(y, subset_predictions),
        })
    return pd.DataFrame(rows)


def markdown_metrics_table(frame, columns):
    """Format evaluation tables without requiring optional Markdown dependencies."""
    headers = {
        "segment": "Segment",
        "model": "Model",
        "n": "N",
        "MAE": "MAE",
        "RMSE": "RMSE",
        "WAPE": "WAPE",
        "R2": "R²",
        "coverage_P50": "P50 coverage",
        "pinball_P50": "P50 pinball",
        "coverage_P90": "P90 coverage",
        "pinball_P90": "P90 pinball",
        "coverage_P95": "P95 coverage",
        "pinball_P95": "P95 pinball",
    }
    lines = [
        "| " + " | ".join(headers[column] for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if column == "n":
                cells.append(str(int(value)))
            elif column in {"segment", "model"}:
                cells.append(str(value))
            elif column.startswith("coverage_"):
                cells.append(f"{float(value):.1%}")
            elif column == "WAPE":
                cells.append(f"{float(value):.4f}")
            elif column == "R2":
                cells.append(f"{float(value):.3f}")
            else:
                cells.append(f"{float(value):.2f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def verify_frozen_development_config():
    """Confirm the frozen OPS choice matches pre-test experiment evidence."""
    config_path = os.path.join(OUT_DIR, "development_experiment_config.json")
    predictions_path = os.path.join(OUT_DIR, "development_validation_predictions.csv")
    if not os.path.exists(config_path) or not os.path.exists(predictions_path):
        raise FileNotFoundError(
            "Pre-test development configuration and validation predictions are required."
        )
    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)
    if config.get("data_cutoff_exclusive") != "2012-06-01T00:00:00":
        raise ValueError("Development experiment cutoff is not the frozen pre-test cutoff.")
    if pd.Timestamp(config["last_loaded_timestamp"]) >= SPLIT_FIT_END:
        raise ValueError("Development experiment data include rows at or after the fit cutoff.")
    if config.get("evaluation_scope") != "Development and chronological validation only; no final test rows loaded.":
        raise ValueError("Development experiment scope does not verify a test-free model decision.")
    if config.get("quantile_estimator", {}).get("name") != FROZEN_QUANTILE_ESTIMATOR:
        raise ValueError("Frozen quantile estimator differs from the development experiment.")
    if config.get("feature_sets", {}).get("enhanced_ops") != OPS_FEATS:
        raise ValueError("Frozen enhanced OPS feature list differs from validation.")
    expected_peaks = {
        "morning_peak": "workingday == 1 and hour in 07-09",
        "evening_peak": "workingday == 1 and hour in 16-19",
        "off_peak": "all rows not in morning_peak or evening_peak",
    }
    if config.get("peak_definitions") != expected_peaks:
        raise ValueError("Frozen peak definitions differ from development validation.")
    if config.get("peak_calibration", {}).get("minimum_rows_per_calibration_peak") != MIN_PEAK_CALIBRATION_ROWS:
        raise ValueError("Frozen local calibration sample threshold differs from validation.")
    validation_timestamps = pd.read_csv(
        predictions_path, usecols=["ts"], parse_dates=["ts"]
    )["ts"]
    if validation_timestamps.empty or validation_timestamps.max() >= SPLIT_FIT_END:
        raise ValueError("Development validation predictions contain data at or after June 2012.")
    return {
        "development_cutoff_exclusive": config["data_cutoff_exclusive"],
        "last_development_timestamp": config["last_loaded_timestamp"],
        "validation_prediction_max_timestamp": validation_timestamps.max().isoformat(),
        "development_rows_loaded": int(config["development_rows_loaded"]),
    }


def verify_preserved_baseline():
    """Check baseline evaluation files still match their preserved SHA-256 manifest."""
    snapshot_dir = os.path.join(OUT_DIR, "baseline_snapshot")
    manifest_path = os.path.join(snapshot_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError("The preserved baseline_snapshot/manifest.json is missing.")
    with open(manifest_path, encoding="utf-8-sig") as file:
        manifest = json.load(file)
    expected = {item["file"]: item for item in manifest}
    for name in ("model_metrics.csv", "forecast_test_predictions.csv"):
        item = expected.get(name)
        path = os.path.join(snapshot_dir, name)
        if item is None or not os.path.exists(path):
            raise FileNotFoundError(f"Preserved baseline artifact is missing: {name}")
        digest = hashlib.sha256()
        with open(path, "rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest().upper() != item["sha256"].upper():
            raise ValueError(f"Preserved baseline artifact failed SHA-256 verification: {name}")
    return snapshot_dir


def write_final_report(
    overall_metrics,
    segment_metrics,
    baseline_comparison,
    evaluation_setup,
    frozen_config,
):
    """Write the concise, data-derived final OPS evaluation report."""
    metric_columns = [
        "n", "MAE", "RMSE", "WAPE", "R2",
        "coverage_P50", "pinball_P50", "coverage_P90", "pinball_P90",
        "coverage_P95", "pinball_P95",
    ]
    report_lines = [
        "# Frozen OPS model: final evaluation",
        "",
        f"**Final test window:** {evaluation_setup['test_start'][:10]} through "
        f"{evaluation_setup['test_end'][:10]} (chronological held-out rows).",
        "",
        "## Frozen model and calibration",
        "",
        f"- **Estimator:** {frozen_config['ops_model']}; quantiles P50, P90, and P95.",
        f"- **Features:** enhanced OPS set ({len(frozen_config['features'])} leakage-safe features); "
        "feature names are recorded in `frozen_model_config.json`.",
        f"- **Calibration:** {FROZEN_CALIBRATION_METHOD}; finite-sample residual offsets "
        "from June–August 2012. Morning (working-day 07:00–09:00) and evening "
        "(working-day 16:00–19:00) offsets are local when each calibration subset has "
        f"at least {MIN_PEAK_CALIBRATION_ROWS} observations; otherwise the global offset "
        "is used. Off-peak rows use the global offset.",
        "- **Leakage guard:** the model fit ends before June 1, 2012; calibration is "
        "June–August; the September–December test was evaluated once and was not used "
        "for feature, estimator, or calibration selection. In the rolling OPS test, "
        "each target hour may use observed history through t−24h, including earlier "
        "test-period hours, as available at forecast issue time.",
        "",
        "## Overall OPS test performance",
        "",
        markdown_metrics_table(
            pd.DataFrame([overall_metrics])[metric_columns], metric_columns
        ),
        "",
        "## Performance by operating period",
        "",
        markdown_metrics_table(segment_metrics, ["segment"] + metric_columns),
        "",
        "## Comparison with preserved baseline",
        "",
        "Both models are scored on the same timestamps available in the preserved "
        "baseline predictions. The baseline uses its preserved P50/P90/P95 outputs; "
        "the frozen model uses the corresponding final predictions.",
        "",
        markdown_metrics_table(
            baseline_comparison,
            ["segment", "model"] + metric_columns,
        ),
        "",
        "## Limitations and business recommendations",
        "",
        "- The target is estimated average concurrent use from Little’s Law and assumed "
        "registered/casual trip durations (12/30 minutes), not observed station inventory.",
        "- Peak-aware calibration is marginal, based on a finite calibration window, and "
        "does not guarantee conditional or future-period coverage. Review the evening "
        "P90/P95 results before using these intervals for service commitments.",
        "- The UCI data lack station-level stock, failed rental attempts, actual fleet "
        "size, and trip durations. These results cannot establish whether complaints "
        "require purchasing bikes or redistribution.",
        "- Pair the forecasts with station inventory and failed-rental telemetry; pilot "
        "targeted redistribution before making capital decisions.",
        "- No model or calibration changes were made in response to final-test results.",
        "",
    ]
    report_path = os.path.join(OUT_DIR, "final_evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(report_lines))
    return report_path


def model_candidates():
    return {
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=25, l2_regularization=1.0, random_state=0,
        ),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=4, max_features=0.8,
            n_jobs=1, random_state=0,
        ),
        "ExtraTreesRegressor": ExtraTreesRegressor(
            n_estimators=200, min_samples_leaf=4, max_features=0.8,
            n_jobs=1, random_state=0,
        ),
    }


def select_point_model(data, features):
    """Choose among regressors on a pre-calibration chronological validation window."""
    training = data[data["ts"] < SELECTION_END]
    validation = data[
        (data["ts"] >= SELECTION_END) & (data["ts"] < SPLIT_FIT_END)
    ]
    if training.empty or validation.empty:
        raise ValueError("Insufficient data for the chronological model-selection window.")
    rows = []
    for name, model in model_candidates().items():
        model.fit(training[features], training["bikes_in_use"])
        prediction = model.predict(validation[features])
        rows.append({"model": name, **regression_metrics(validation["bikes_in_use"], prediction)})
    result = pd.DataFrame(rows).sort_values(["MAE", "RMSE"]).reset_index(drop=True)
    return str(result.iloc[0]["model"]), result


def chronological_windows(data):
    """Return development, calibration, and final-test rows without overlapping time."""
    eligible = data[
        data["is_observed"]
        & data["outage_day"].eq(0)
        & data["bikes_in_use"].notna()
    ].copy()
    selection_train = eligible[eligible["ts"] < SELECTION_END].dropna(subset=OPS_FEATS)
    validation = eligible[
        (eligible["ts"] >= SELECTION_END) & (eligible["ts"] < SPLIT_FIT_END)
    ].dropna(subset=OPS_FEATS)
    fit = eligible[eligible["ts"] < SPLIT_FIT_END]
    calibration = eligible[
        (eligible["ts"] >= SPLIT_FIT_END) & (eligible["ts"] < SPLIT_CAL_END)
    ]
    test = eligible[eligible["ts"] >= SPLIT_CAL_END]
    fit_ops = fit.dropna(subset=OPS_FEATS)
    calibration_ops = calibration.dropna(subset=OPS_FEATS)
    test_ops = test.dropna(subset=OPS_FEATS)

    windows = {
        "selection_train": selection_train,
        "validation": validation,
        "fit": fit,
        "calibration": calibration,
        "test": test,
        "fit_ops": fit_ops,
        "calibration_ops": calibration_ops,
        "test_ops": test_ops,
    }
    if any(frame.empty for frame in windows.values()):
        raise ValueError("A chronological development, calibration, or test window is empty.")
    if selection_train["ts"].max() >= validation["ts"].min():
        raise ValueError("Model-selection training and validation timestamps overlap.")
    if fit["ts"].max() >= calibration["ts"].min():
        raise ValueError("Fit and calibration timestamps overlap.")
    if calibration["ts"].max() >= test["ts"].min():
        raise ValueError("Calibration and final-test timestamps overlap.")
    if validation["ts"].max() >= SPLIT_FIT_END or test["ts"].min() < SPLIT_CAL_END:
        raise ValueError("Chronological evaluation boundaries are inconsistent.")
    return windows


def validation_feature_importance(data, features):
    """Estimate median-quantile importance using only pre-test development data."""
    data = data[
        data["is_observed"]
        & data["outage_day"].eq(0)
        & data["bikes_in_use"].notna()
    ]
    training = data[data["ts"] < SELECTION_END].dropna(subset=features)
    validation = data[
        (data["ts"] >= SELECTION_END) & (data["ts"] < SPLIT_FIT_END)
    ].dropna(subset=features)
    if training.empty or validation.empty:
        raise ValueError("Insufficient pre-test data for validation feature importance.")
    model = HistGradientBoostingRegressor(
        loss="quantile", quantile=0.5, max_iter=300, learning_rate=0.05,
        max_leaf_nodes=31, min_samples_leaf=25, l2_regularization=1.0,
        random_state=0,
    )
    model.fit(training[features], training["bikes_in_use"])
    sample = validation.sample(min(1500, len(validation)), random_state=0)
    result = permutation_importance(
        model,
        sample[features],
        sample["bikes_in_use"],
        n_repeats=5,
        n_jobs=1,
        random_state=0,
        scoring="neg_mean_absolute_error",
    )
    return pd.Series(result.importances_mean, index=features).sort_values(ascending=False)


def main():
    run_marker_path = os.path.join(OUT_DIR, "final_evaluation_run.json")
    if os.path.exists(run_marker_path):
        raise RuntimeError(
            "The frozen September–December final evaluation has already been attempted. "
            "Do not rerun it; inspect outputs/final_evaluation_run.json."
        )
    development_evidence = verify_frozen_development_config()
    baseline_snapshot = verify_preserved_baseline()

    features_path = os.path.join(OUT_DIR, "features.csv")
    if os.path.exists(features_path):
        data = pd.read_csv(features_path, parse_dates=["ts", "dteday"])
    else:
        data = pd.DataFrame()
    if not set(PLAN_FEATS + OPS_FEATS).issubset(data.columns):
        raw_hour, _ = load_raw()
        data = build_features(complete_grid(raw_hour))
        data.to_csv(features_path, index=False)

    windows = chronological_windows(data)
    eligible = data[
        data["is_observed"]
        & data["outage_day"].eq(0)
        & data["bikes_in_use"].notna()
    ].copy()
    fit = windows["fit"]
    calibration = windows["calibration"]
    test = windows["test"]
    fit_ops = windows["fit_ops"]
    calibration_ops = windows["calibration_ops"]
    test_ops = windows["test_ops"]
    if min(len(fit_ops), len(calibration_ops), len(test_ops)) == 0:
        raise ValueError("A chronological train/calibration/test split is empty.")
    print(
        f"fit={len(fit_ops)}, calibration={len(calibration_ops)}, "
        f"test={len(test_ops)}; test range "
        f"{test_ops.ts.min().date()} to {test_ops.ts.max().date()}"
    )
    evaluation_setup = {
        "selection_train_start": windows["selection_train"]["ts"].min().isoformat(),
        "selection_train_end": windows["selection_train"]["ts"].max().isoformat(),
        "validation_start": windows["validation"]["ts"].min().isoformat(),
        "validation_end": windows["validation"]["ts"].max().isoformat(),
        "fit_end_exclusive": SPLIT_FIT_END.isoformat(),
        "calibration_start": calibration["ts"].min().isoformat(),
        "calibration_end_exclusive": SPLIT_CAL_END.isoformat(),
        "test_start": test["ts"].min().isoformat(),
        "test_end": test["ts"].max().isoformat(),
        "ops_history_contract": {
            "latest_target_lag_hours": OPS_HISTORY_CUTOFF_HOURS,
            "recent_and_prior_window_hours": OPS_ROLLING_WINDOW_HOURS,
            "rolling_minimum_observations": OPS_ROLLING_MIN_PERIODS,
            "forecast_horizon_note": (
                "OPS predicts one row at a time using observed demand through t-24h; "
                "for start-of-hour timestamps, the latest input hour must have completed "
                "before issue, about 23 hours before the target hour starts."
            ),
            "target_leakage_guardrail": (
                "No target at or after t is used; missing observations and configured "
                "event-date targets remain unavailable to historical lag and rolling features."
            ),
            "test_guardrail": (
                "September-December 2012 is reserved for final evaluation only; it is not "
                "used for selection, feature importance, calibration, or tuning."
            ),
        },
        "rows": {name: int(len(frame)) for name, frame in windows.items()},
        "test_used_for_model_or_calibration_selection": False,
        "final_test_evaluation_run_count": 1,
    }
    with open(os.path.join(OUT_DIR, "evaluation_setup.json"), "w", encoding="utf-8") as file:
        json.dump(evaluation_setup, file, indent=2)

    frozen_config = {
        "status": "frozen_before_final_test_evaluation",
        "ops_model": FROZEN_QUANTILE_ESTIMATOR,
        "quantiles": list(QUANTILES),
        "feature_set": "enhanced_ops",
        "features": OPS_FEATS,
        "estimator_parameters": {
            "loss": "quantile",
            "max_iter": 300,
            "learning_rate": 0.05,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 25,
            "l2_regularization": 1.0,
            "random_state": 0,
        },
        "calibration": {
            "method": FROZEN_CALIBRATION_METHOD,
            "fit_period_end_exclusive": SPLIT_FIT_END.isoformat(),
            "calibration_start": SPLIT_FIT_END.isoformat(),
            "calibration_end_exclusive": SPLIT_CAL_END.isoformat(),
            "minimum_peak_rows": MIN_PEAK_CALIBRATION_ROWS,
            "morning_peak": "workingday == 1 and hr in 7..9",
            "evening_peak": "workingday == 1 and hr in 16..19",
            "off_peak": "global offset",
            "under_supported_peak_fallback": "global offset",
            "finite_sample_rank": "min(ceil((n + 1) * q), n), one-based",
            "monotone_projection": "Cumulative maximum across P50, P90, and P95 after calibration.",
        },
        "decision_evidence": {
            "experiment_config": "development_experiment_config.json",
            "validation_metrics": "development_validation_metrics.csv",
            "validation_predictions_max_timestamp": development_evidence[
                "validation_prediction_max_timestamp"
            ],
            "development_data_cutoff_exclusive": development_evidence[
                "development_cutoff_exclusive"
            ],
            "test_used_for_selection_or_calibration": False,
        },
        "final_test_period": {
            "start": evaluation_setup["test_start"],
            "end": evaluation_setup["test_end"],
            "run_count": 1,
        },
    }
    config_path = os.path.join(OUT_DIR, "frozen_model_config.json")
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(frozen_config, file, indent=2)

    # Select the point regressor before calibration/test using an earlier validation window.
    selected_name, selection_scores = select_point_model(fit_ops, OPS_FEATS)
    selection_scores.to_csv(os.path.join(OUT_DIR, "model_selection_metrics.csv"), index=False)
    importance = validation_feature_importance(fit_ops, OPS_FEATS)
    importance.rename_axis("feature").rename("permutation_importance").to_csv(
        os.path.join(OUT_DIR, "validation_feature_importance.csv")
    )
    selected_model = model_candidates()[selected_name]
    selected_model.fit(fit_ops[OPS_FEATS], fit_ops["bikes_in_use"])

    # Fit independent calendar-only and day-ahead quantile bundles; calibration is separate.
    plan_fit = fit.dropna(subset=PLAN_FEATS)
    plan_cal = calibration.dropna(subset=PLAN_FEATS)
    plan_test = test.dropna(subset=PLAN_FEATS)
    plan = fit_quantile_bundle(plan_fit, plan_cal, PLAN_FEATS, QUANTILES)
    ops = fit_quantile_bundle(
        fit_ops,
        calibration_ops,
        OPS_FEATS,
        QUANTILES,
        calibration_method=FROZEN_CALIBRATION_METHOD,
        min_peak_calibration_rows=MIN_PEAK_CALIBRATION_ROWS,
    )
    plan_cal_predictions = predict_quantiles(plan, plan_cal)
    ops_cal_predictions = predict_quantiles(ops, calibration_ops)

    calibration_offset_rows = []
    for q, item in ops["quantiles"].items():
        calibration_offset_rows.append({
            "quantile": q,
            "scope": "global_and_off_peak",
            "sample_count": item["calibration_sample_count"],
            "offset": item["offset"],
            "fallback": False,
        })
        for peak_name, offset in item["peak_offsets"].items():
            calibration_offset_rows.append({
                "quantile": q,
                "scope": peak_name,
                "sample_count": item["peak_calibration_counts"][peak_name],
                "offset": offset,
                "fallback": item["peak_fallbacks"][peak_name],
            })
    pd.DataFrame(calibration_offset_rows).to_csv(
        os.path.join(OUT_DIR, "calibration_offsets.csv"), index=False
    )

    with open(run_marker_path, "w", encoding="utf-8") as file:
        json.dump({
            "status": "started",
            "final_test_evaluation_run_count": 1,
            "test_start": evaluation_setup["test_start"],
            "test_end": evaluation_setup["test_end"],
            "frozen_config": "frozen_model_config.json",
        }, file, indent=2)

    point_prediction = selected_model.predict(test_ops[OPS_FEATS])
    plan_predictions = predict_quantiles(plan, plan_test)
    ops_predictions = predict_quantiles(ops, test_ops)

    calibration_rows = []
    for name, y_values, predictions in (
        ("PLAN calendar/weather quantile model", plan_cal["bikes_in_use"], plan_cal_predictions),
        ("OPS day-ahead quantile model", calibration_ops["bikes_in_use"], ops_cal_predictions),
    ):
        calibration_rows.append({
            "model": name,
            "evaluation_period": "calibration (used to estimate residual offsets)",
            **quantile_metrics(y_values, predictions),
        })
    pd.DataFrame(calibration_rows).to_csv(
        os.path.join(OUT_DIR, "calibration_metrics.csv"), index=False
    )

    metrics_rows = []
    for name, y_values, predictions in (
        ("PLAN calendar/weather quantile model", plan_test["bikes_in_use"], plan_predictions),
        (
            "OPS day-ahead quantile model (enhanced features, peak-aware)",
            test_ops["bikes_in_use"],
            ops_predictions,
        ),
    ):
        metrics_rows.append({"model": name, **quantile_metrics(y_values, predictions)})
    point_test_predictions = {selected_name: point_prediction}
    for name, model in model_candidates().items():
        if name != selected_name:
            model.fit(fit_ops[OPS_FEATS], fit_ops["bikes_in_use"])
            point_test_predictions[name] = model.predict(test_ops[OPS_FEATS])
    for name, prediction in point_test_predictions.items():
        metrics_rows.append({
            "model": f"{'Selected' if name == selected_name else 'Candidate'} point regressor: {name}",
            **regression_metrics(test_ops["bikes_in_use"], prediction),
        })
    profile = fit_ops.groupby(["hr", "workingday"])["bikes_in_use"].mean()
    naive = test_ops.set_index(["hr", "workingday"]).index.map(profile).to_numpy()
    metrics_rows.append({
        "model": "Baseline: fit-period mean by hour and working-day flag",
        **regression_metrics(test_ops["bikes_in_use"], naive),
    })
    last_week = test_ops["lag_168"].to_numpy()
    metrics_rows.append({
        "model": "Baseline: same hour one week earlier",
        **regression_metrics(test_ops["bikes_in_use"], last_week),
    })
    metric_table = pd.DataFrame(metrics_rows)
    metric_table.to_csv(os.path.join(OUT_DIR, "model_metrics.csv"), index=False)
    print("\nHeld-out test metrics:\n", metric_table.round(3).to_string(index=False))
    print("\nPoint-regressor selection on the earlier validation window:\n", selection_scores.round(3).to_string(index=False))

    # Test-period predictions and conditional fleet scenarios.
    predictions_frame = test_ops[
        ["ts", "dteday", "hr", "weekday", "workingday", "weathersit", "bikes_in_use"]
    ].copy()
    predictions_frame["ops_point_prediction"] = selected_model.predict(test_ops[OPS_FEATS])
    for q, values in ops_predictions.items():
        predictions_frame[f"ops_p{int(q * 100)}"] = values
    predictions_frame["p95_fleet_scenario_with_buffer"] = fleet_from_bikes_in_use(
        predictions_frame["ops_p95"].to_numpy()
    )
    predictions_frame.to_csv(os.path.join(OUT_DIR, "forecast_test_predictions.csv"), index=False)

    segment_metrics = ops_segment_metrics(test_ops, ops_predictions)
    segment_metrics.to_csv(
        os.path.join(OUT_DIR, "ops_test_metrics_by_period.csv"), index=False
    )

    baseline_predictions_path = os.path.join(
        baseline_snapshot, "forecast_test_predictions.csv"
    )
    baseline_predictions = pd.read_csv(baseline_predictions_path, parse_dates=["ts"])
    baseline_columns = {"ts", "bikes_in_use", "ops_p50", "ops_p90", "ops_p95"}
    if not baseline_columns.issubset(baseline_predictions.columns):
        raise ValueError("Preserved baseline predictions lack the expected OPS quantiles.")
    baseline_predictions = baseline_predictions[
        ["ts", "bikes_in_use", "ops_p50", "ops_p90", "ops_p95"]
    ].rename(columns={
        "bikes_in_use": "baseline_bikes_in_use",
        "ops_p50": "baseline_p50",
        "ops_p90": "baseline_p90",
        "ops_p95": "baseline_p95",
    })
    comparison = predictions_frame.merge(
        baseline_predictions,
        on="ts",
        how="inner",
        validate="one_to_one",
    )
    if comparison.empty:
        raise ValueError("No matching timestamps exist for the preserved baseline comparison.")
    if not np.allclose(
        comparison["bikes_in_use"],
        comparison["baseline_bikes_in_use"],
        rtol=0,
        atol=1e-8,
    ):
        raise ValueError("Preserved baseline target values differ from the final test rows.")
    comparison_frame = comparison[["hr", "workingday", "bikes_in_use"]].copy()
    comparison_frame = comparison_frame.reset_index(drop=True)
    comparison_specs = (
        (
            "Frozen OPS: enhanced features + peak-aware calibration",
            {
                q: comparison[f"ops_p{int(q * 100)}"].to_numpy()
                for q in QUANTILES
            },
        ),
        (
            "Preserved baseline OPS",
            {
                q: comparison[f"baseline_p{int(q * 100)}"].to_numpy()
                for q in QUANTILES
            },
        ),
    )
    comparison_rows = []
    for model_name, comparison_predictions in comparison_specs:
        rows = ops_segment_metrics(comparison_frame, comparison_predictions)
        for row in rows.to_dict("records"):
            comparison_rows.append({"model": model_name, **row})
    baseline_comparison = pd.DataFrame(comparison_rows)
    baseline_comparison.to_csv(
        os.path.join(OUT_DIR, "baseline_comparison.csv"), index=False
    )

    test_peak = test_ops["commute_peak"].eq(1).to_numpy()
    if test_peak.any():
        peak_metrics = {
            "peak_hour_rows": int(test_peak.sum()),
            "ops_p50_mae_peak_hours": float(mean_absolute_error(
                test_ops.loc[test_peak, "bikes_in_use"], ops_predictions[0.5][test_peak]
            )),
            "ops_p95_coverage_peak_hours": float(np.mean(
                test_ops.loc[test_peak, "bikes_in_use"].to_numpy()
                <= ops_predictions[0.95][test_peak]
            )),
        }
    else:
        peak_metrics = {"peak_hour_rows": 0}

    sensitivity = {}
    observed_2012 = fit[fit["yr"].eq(1)]
    for registered_minutes, casual_minutes in ((8, 20), (12, 30), (15, 40), (20, 50)):
        scenario_use = (
            observed_2012["registered"] * registered_minutes
            + observed_2012["casual"] * casual_minutes
        ) / 60
        sensitivity[f"registered_{registered_minutes}_casual_{casual_minutes}"] = {
            "trip_duration_minutes": {
                "registered": registered_minutes,
                "casual": casual_minutes,
            },
            "observed_2012_median_hourly_use_estimate": float(scenario_use.median()),
            "observed_2012_p95_hourly_use_estimate": float(scenario_use.quantile(0.95)),
            "observed_2012_p99_hourly_use_estimate": float(scenario_use.quantile(0.99)),
            "p99_scenario_fleet_with_buffer": float(fleet_from_bikes_in_use(
                scenario_use.quantile(0.99)
            )),
        }
    fleet_results = {
        "interpretation": (
            "Estimated average concurrent use derived from rentals using Little's Law. "
            "This is not observed station inventory, fleet size, or failed demand."
        ),
        "assumptions": {
            "registered_trip_min": REG_TRIP_MIN,
            "casual_trip_min": CAS_TRIP_MIN,
            "fleet_buffer": BUFFER,
            "service_quantile": 0.95,
        },
        "selected_point_model": selected_name,
        "selection_rule": "Lowest MAE on chronological validation before calibration and test.",
        "ops_quantile_model": FROZEN_QUANTILE_ESTIMATOR,
        "ops_feature_set": "enhanced_ops",
        "ops_calibration_method": FROZEN_CALIBRATION_METHOD,
        "ops_minimum_peak_calibration_rows": MIN_PEAK_CALIBRATION_ROWS,
        "final_test_evaluation_run_count": 1,
        "preserved_baseline_comparison_rows": int(len(comparison)),
        "test_selection_guardrail": (
            "The selected model is not changed based on held-out test rankings. The test "
            "metrics report every candidate for transparent comparison without using test "
            "performance to select or tune the final model."
        ),
        "periods": {
            "fit_end_exclusive": SPLIT_FIT_END.date().isoformat(),
            "calibration_start": SPLIT_FIT_END.date().isoformat(),
            "calibration_end_exclusive": SPLIT_CAL_END.date().isoformat(),
            "test_start": SPLIT_CAL_END.date().isoformat(),
            "test_end": test_ops["ts"].max().isoformat(),
            "fit_rows": int(len(fit_ops)),
            "calibration_rows": int(len(calibration_ops)),
            "test_rows": int(len(test_ops)),
        },
        "fit_period_2012_estimated_use": {
            "median": float(observed_2012["bikes_in_use"].median()),
            "p95": float(observed_2012["bikes_in_use"].quantile(0.95)),
            "p99": float(observed_2012["bikes_in_use"].quantile(0.99)),
            "peak": float(observed_2012["bikes_in_use"].max()),
        },
        "heldout_test_ops_p95_fleet_scenario": {
            "max_hourly_requirement_with_buffer": float(
                predictions_frame["p95_fleet_scenario_with_buffer"].max()
            ),
            "p99_of_hourly_requirements_with_buffer": float(
                predictions_frame["p95_fleet_scenario_with_buffer"].quantile(0.99)
            ),
        },
        "peak_hour_test_metrics": peak_metrics,
        "trip_duration_sensitivity": sensitivity,
        "calibration_note": (
            "The frozen OPS model uses finite-sample residual quantiles from the separate "
            "calibration period, with local offsets for supported working-day peaks and global "
            "offsets elsewhere or when under-supported. Coverage is evaluated on the held-out "
            "test period; marginal coverage is not a station-level guarantee."
        ),
        "limitations": [
            "Trip durations and the fleet buffer are assumed, not measured in these CSVs.",
            "The hourly estimate represents average concurrency and cannot reveal station stock-outs.",
            "No purchase or redistribution verdict is supported without actual fleet and station data.",
        ],
    }
    overall_ops_metrics = segment_metrics.loc[
        segment_metrics["segment"].eq("overall")
    ].iloc[0]
    final_report_path = write_final_report(
        overall_ops_metrics,
        segment_metrics,
        baseline_comparison,
        evaluation_setup,
        frozen_config,
    )
    fleet_results["final_evaluation_report"] = os.path.basename(final_report_path)
    fleet_results["frozen_model_config"] = os.path.basename(config_path)
    fleet_results["test_metrics"] = {
        key: float(value) if isinstance(value, (np.floating, float)) else int(value)
        for key, value in overall_ops_metrics.items()
        if key != "segment"
    }

    # Fit-period profiles and data-derived lookups support transparent future scenarios.
    profile_2012 = fit[fit["yr"].eq(1)].copy()
    profile_2012["daytype"] = np.where(profile_2012["workingday"].eq(1), "workday", "non_working_day")
    fleet_profile = profile_2012.groupby(["daytype", "hr"])["bikes_in_use"].agg(
        P50="median", P95=lambda values: values.quantile(0.95)
    )
    fleet_profile["fleet_P95_with_buffer"] = np.ceil(fleet_profile["P95"] * (1 + BUFFER))
    fleet_profile.to_csv(os.path.join(OUT_DIR, "fleet_profile_by_hour.csv"))

    weather_data = fit.groupby(["mnth", "weathersit"])[
        ["temp_c", "hum_pct", "wind_kmh"]
    ].median()
    weather_all = fit.groupby("weathersit")[["temp_c", "hum_pct", "wind_kmh"]].median()
    season_by_month = (
        fit.groupby("mnth")["season"].agg(lambda values: int(values.mode().iloc[0])).to_dict()
    )
    fleet_results["validation_ops_permutation_importance"] = {
        key: float(value) for key, value in importance.head(10).items()
    }

    joblib.dump({
        "plan": plan,
        "ops": ops,
        "point_model": selected_model,
        "point_model_name": selected_name,
        "weather_by_month": weather_data,
        "weather_all": weather_all,
        "season_by_month": season_by_month,
        "profile": fleet_profile,
        "assumptions": fleet_results["assumptions"],
        "features": {"plan": PLAN_FEATS, "ops": OPS_FEATS},
        "ops_history_contract": evaluation_setup["ops_history_contract"],
        "frozen_model_config": frozen_config,
    }, os.path.join(OUT_DIR, "plan_model.joblib"))
    with open(os.path.join(OUT_DIR, "fleet_results.json"), "w", encoding="utf-8") as file:
        json.dump(fleet_results, file, indent=2)

    make_figures(test_ops, ops_predictions, importance, fleet_profile)
    with open(run_marker_path, "w", encoding="utf-8") as file:
        json.dump({
            "status": "completed",
            "final_test_evaluation_run_count": 1,
            "test_start": evaluation_setup["test_start"],
            "test_end": evaluation_setup["test_end"],
            "frozen_config": os.path.basename(config_path),
            "report": os.path.basename(final_report_path),
        }, file, indent=2)
    print("\nFleet scenario results:\n", json.dumps(fleet_results, indent=2))


def make_figures(test, predictions, importance, profile):
    example = test[
        (test["ts"] >= "2012-09-17") & (test["ts"] < "2012-09-24")
    ].copy()
    positions = test.index.get_indexer(example.index)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(example["ts"], example["bikes_in_use"], color="black", label="Observed estimate")
    axes[0].plot(example["ts"], predictions[0.5][positions], label="P50 forecast")
    axes[0].fill_between(
        example["ts"], predictions[0.5][positions], predictions[0.95][positions],
        alpha=0.25, label="P50–P95",
    )
    axes[0].set(title="Held-out week: estimated bikes in use", xlabel="Date", ylabel="Bikes (estimate)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend(fontsize=8)
    for daytype in ("workday", "non_working_day"):
        subset = profile.loc[daytype]
        axes[1].plot(subset.index, subset["fleet_P95_with_buffer"], label=f"{daytype}: P95 + buffer")
        axes[1].plot(subset.index, subset["P50"], linestyle="--", label=f"{daytype}: median use")
    axes[1].set(title="Historical hourly scenario profile", xlabel="Hour", ylabel="Estimated bikes")
    axes[1].legend(fontsize=7)
    importance.head(10).sort_values().plot.barh(ax=axes[2], color="#4472C4")
    axes[2].set(title="Development validation permutation importance", xlabel="Decrease in model score")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_03_model.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
