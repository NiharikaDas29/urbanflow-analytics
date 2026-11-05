"""Shared paths, data loading, feature engineering, and model helpers."""

from __future__ import annotations

import glob
import os
import zipfile

# Avoid a costly Windows physical-core probe while retaining parallel capacity.
_logical_cpus = os.cpu_count() or 1
if _logical_cpus > 1:
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(min(4, _logical_cpus - 1)))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# These are scenario assumptions, not measurements present in the UCI dataset.
REG_TRIP_MIN = 12
CAS_TRIP_MIN = 30
BUFFER = 0.15
SERVICE_Q = 0.95
CURRENT_FLEET = None
OUTAGE_DAYS = ("2012-10-29", "2012-10-30")
POTENTIAL_OUTAGE_MIN_HOURS = 6
SPLIT_FIT_END = pd.Timestamp("2012-06-01")
SPLIT_CAL_END = pd.Timestamp("2012-09-01")
PEAK_HOURS = [7, 8, 9, 16, 17, 18, 19]
OPS_HISTORY_CUTOFF_HOURS = 24
OPS_ROLLING_WINDOW_HOURS = 168
OPS_ROLLING_MIN_PERIODS = 120

HOUR_REQUIRED = {
    "dteday", "season", "yr", "mnth", "hr", "holiday", "weekday",
    "workingday", "weathersit", "temp", "atemp", "hum", "windspeed",
    "casual", "registered", "cnt",
}
DAY_REQUIRED = {"dteday", "cnt", "casual", "registered"}

BASE_FEATS = [
    "hr_sin", "hr_cos", "weekday", "workingday", "holiday", "mnth",
    "season", "temp_c", "hum_pct", "wind_kmh", "weathersit",
    "rain_or_snow", "commute_peak", "weekend_daytime",
]
PLAN_FEATS = BASE_FEATS
OPS_FEATS = BASE_FEATS + [
    "lag_24", "lag_168", "lag_336", "roll24", "roll168", "demand_trend_7d",
    "mnth_sin", "mnth_cos", "hr_month_sin_sin", "hr_month_sin_cos",
    "hr_month_cos_sin", "hr_month_cos_cos", "temp_weather_2",
    "temp_weather_3", "temp_weather_4",
]


def ensure_data() -> str:
    """Return the data directory, extracting a local UCI archive if needed."""
    hour_path = os.path.join(DATA_DIR, "hour.csv")
    if not os.path.exists(hour_path):
        os.makedirs(DATA_DIR, exist_ok=True)
        archives = glob.glob(os.path.join(ROOT, "*.zip")) + glob.glob(os.path.join(DATA_DIR, "*.zip"))
        if not archives:
            raise FileNotFoundError(
                "Could not find data/hour.csv. Put hour.csv and day.csv in ./data."
            )
        with zipfile.ZipFile(archives[0]) as archive:
            archive.extractall(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, "day.csv")):
        raise FileNotFoundError("Could not find data/day.csv.")
    return DATA_DIR


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both UCI tables and validate the columns required downstream."""
    data_dir = ensure_data()
    hour = pd.read_csv(os.path.join(data_dir, "hour.csv"))
    day = pd.read_csv(os.path.join(data_dir, "day.csv"))
    for frame, required, name in (
        (hour, HOUR_REQUIRED, "hour.csv"),
        (day, DAY_REQUIRED, "day.csv"),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")
        frame["dteday"] = pd.to_datetime(frame["dteday"], errors="coerce")
    if hour["dteday"].isna().any() or day["dteday"].isna().any():
        raise ValueError("Invalid dates found in dteday; correct the source CSV before continuing.")
    hour["ts"] = hour["dteday"] + pd.to_timedelta(hour["hr"], unit="h")
    return hour, day


def complete_grid(hour: pd.DataFrame) -> pd.DataFrame:
    """Add absent hourly timestamps without converting unknown rentals to zero.

    An observed row with ``cnt == 0`` is a genuine recorded zero. A missing row
    has unknown demand and stays null; its context is filled only so the time
    series can be plotted, and ``context_imputed`` records that fact.
    """
    source = hour.copy()
    if "ts" not in source:
        source["ts"] = pd.to_datetime(source["dteday"]) + pd.to_timedelta(source["hr"], unit="h")
    source = source.sort_values("ts")
    if source["ts"].duplicated().any():
        raise ValueError("Duplicate hourly timestamps prevent building a unique hourly grid.")
    grid = pd.DataFrame({"ts": pd.date_range(source.ts.min(), source.ts.max(), freq="h")})
    grid = grid.merge(source, on="ts", how="left", validate="one_to_one", suffixes=("", "_source"))
    grid["was_missing"] = grid["cnt"].isna().astype("int8")
    grid["is_observed"] = grid["was_missing"].eq(0)
    grid["dteday"] = grid["ts"].dt.normalize()
    grid["hr"] = grid["ts"].dt.hour
    outage_dates = pd.to_datetime(list(OUTAGE_DAYS))
    grid["outage_day"] = grid["dteday"].isin(outage_dates).astype("int8")
    missing_groups = (~grid["was_missing"].eq(1)).cumsum()
    missing_run_length = grid["was_missing"].groupby(missing_groups).transform("sum")
    grid["potential_outage"] = (
        grid["was_missing"].eq(1)
        & (
            grid["outage_day"].eq(1)
            | missing_run_length.ge(POTENTIAL_OUTAGE_MIN_HOURS)
        )
    )

    context = [
        "season", "yr", "mnth", "holiday", "weekday", "workingday",
        "weathersit", "temp", "atemp", "hum", "windspeed",
    ]
    available = [column for column in context if column in grid]
    grid["context_imputed"] = grid[available].isna().any(axis=1).astype("int8")
    grid[available] = grid[available].ffill().bfill()
    return grid


def build_features(
    g: pd.DataFrame,
    reg_min: float = REG_TRIP_MIN,
    cas_min: float = CAS_TRIP_MIN,
) -> pd.DataFrame:
    """Add PLAN features and strictly historical OPS features.

    Little's Law is applied as arrivals per hour × assumed average trip duration
    in hours. It estimates average concurrent use; it is not station inventory.

    OPS predicts the row at timestamp ``t`` using demand history through ``t-24h``.
    UCI timestamps label the start of each hour, so the latest included hourly
    target is available after that prior-day hour has completed (about 23 hours
    before the target hour starts). OPS is a rolling day-ahead forecast, not a
    multi-month forecast; PLAN remains independent of historical demand lags.
    """
    if reg_min <= 0 or cas_min <= 0:
        raise ValueError("Trip-duration assumptions must be positive minutes.")
    g = g.sort_values("ts").reset_index(drop=True).copy()
    g["temp_c"] = g["temp"] * 41
    g["atemp_c"] = g["atemp"] * 50
    g["hum_pct"] = g["hum"] * 100
    g["wind_kmh"] = g["windspeed"] * 67
    g["wind_reported_zero"] = g["windspeed"].eq(0).astype("int8")
    g["hr_sin"] = np.sin(2 * np.pi * g["hr"] / 24)
    g["hr_cos"] = np.cos(2 * np.pi * g["hr"] / 24)
    g["rain_or_snow"] = g["weathersit"].ge(3).astype("int8")
    g["commute_peak"] = (g["workingday"].eq(1) & g["hr"].isin(PEAK_HOURS)).astype("int8")
    g["weekend_daytime"] = (
        g["workingday"].eq(0) & g["hr"].between(10, 17)
    ).astype("int8")
    month_angle = 2 * np.pi * (g["mnth"] - 1) / 12
    g["mnth_sin"] = np.sin(month_angle)
    g["mnth_cos"] = np.cos(month_angle)
    g["hr_month_sin_sin"] = g["hr_sin"] * g["mnth_sin"]
    g["hr_month_sin_cos"] = g["hr_sin"] * g["mnth_cos"]
    g["hr_month_cos_sin"] = g["hr_cos"] * g["mnth_sin"]
    g["hr_month_cos_cos"] = g["hr_cos"] * g["mnth_cos"]
    for weather_code in (2, 3, 4):
        g[f"temp_weather_{weather_code}"] = (
            g["temp_c"] * g["weathersit"].eq(weather_code).astype("int8")
        )
    g["casual_share"] = g["casual"] / g["cnt"].replace(0, np.nan)
    g["bikes_in_use"] = (g["registered"] * reg_min + g["casual"] * cas_min) / 60.0
    g.loc[~g["is_observed"], "bikes_in_use"] = np.nan

    # Exclude missing rows and configured event dates from historical demand inputs.
    history = g["bikes_in_use"].where(
        g["is_observed"] & g["outage_day"].eq(0)
    )
    g["lag_24"] = history.shift(OPS_HISTORY_CUTOFF_HOURS)
    g["lag_168"] = history.shift(168)
    g["lag_336"] = history.shift(336)
    g["roll24"] = history.shift(OPS_HISTORY_CUTOFF_HOURS).rolling(
        24, min_periods=18
    ).mean()

    # Each window ends at t-24, the latest completed hourly target available to OPS.
    recent_7d = history.shift(OPS_HISTORY_CUTOFF_HOURS).rolling(
        OPS_ROLLING_WINDOW_HOURS, min_periods=OPS_ROLLING_MIN_PERIODS
    ).mean()
    prior_7d = history.shift(OPS_HISTORY_CUTOFF_HOURS + OPS_ROLLING_WINDOW_HOURS).rolling(
        OPS_ROLLING_WINDOW_HOURS, min_periods=OPS_ROLLING_MIN_PERIODS
    ).mean()
    g["roll168"] = recent_7d
    g["demand_trend_7d"] = recent_7d - prior_7d
    return g


def fit_quantile_bundle(
    fit_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    feats: list[str],
    qs: tuple[float, ...] = (0.5, 0.9, 0.95),
    target: str = "bikes_in_use",
    calibration_method: str = "global",
    min_peak_calibration_rows: int = 100,
) -> dict:
    """Fit quantile regressors and estimate finite-sample calibration offsets."""
    if fit_df.empty or cal_df.empty:
        raise ValueError("Training and calibration periods must both contain observations.")
    if calibration_method not in {"global", "peak_aware"}:
        raise ValueError("calibration_method must be 'global' or 'peak_aware'.")
    if min_peak_calibration_rows < 1:
        raise ValueError("min_peak_calibration_rows must be positive.")
    if calibration_method == "peak_aware" and not {"workingday", "hr"}.issubset(cal_df.columns):
        raise ValueError("Peak-aware calibration requires workingday and hr columns.")

    peak_masks = {}
    if calibration_method == "peak_aware":
        working_day = cal_df["workingday"].eq(1)
        peak_masks = {
            "morning_peak": working_day & cal_df["hr"].between(7, 9),
            "evening_peak": working_day & cal_df["hr"].between(16, 19),
        }

    bundle = {}
    for q in qs:
        if not 0 < q < 1:
            raise ValueError("Quantiles must be strictly between zero and one.")
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=q,
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=25,
            l2_regularization=1.0,
            random_state=0,
        )
        model.fit(fit_df[feats], fit_df[target])
        raw = model.predict(cal_df[feats])
        residuals = cal_df[target].to_numpy(dtype=float) - raw
        sorted_residuals = np.sort(residuals)
        rank = min(int(np.ceil((len(sorted_residuals) + 1) * q)), len(sorted_residuals)) - 1
        offset = float(sorted_residuals[rank])
        peak_offsets = {}
        peak_calibration_counts = {}
        peak_fallbacks = {}
        for name, mask in peak_masks.items():
            count = int(mask.sum())
            peak_calibration_counts[name] = count
            fallback = count < min_peak_calibration_rows
            peak_fallbacks[name] = fallback
            if fallback:
                peak_offsets[name] = offset
            else:
                local_residuals = np.sort(residuals[mask.to_numpy()])
                local_rank = min(
                    int(np.ceil((len(local_residuals) + 1) * q)),
                    len(local_residuals),
                ) - 1
                peak_offsets[name] = float(local_residuals[local_rank])
        bundle[q] = {
            "model": model,
            "offset": offset,
            "peak_offsets": peak_offsets,
            "peak_calibration_counts": peak_calibration_counts,
            "peak_fallbacks": peak_fallbacks,
            "calibration_sample_count": int(len(residuals)),
        }
    return {
        "quantiles": bundle,
        "feats": feats,
        "calibration_method": calibration_method,
        "min_peak_calibration_rows": min_peak_calibration_rows,
    }


def predict_bundle(bundle: dict, X: pd.DataFrame, q: float) -> np.ndarray:
    """Predict nonnegative bikes-in-use estimates for one calibrated quantile."""
    if q not in bundle["quantiles"]:
        raise ValueError(f"Quantile {q} is not available in this model.")
    item = bundle["quantiles"][q]
    offsets = np.full(len(X), item["offset"], dtype=float)
    if bundle.get("calibration_method", "global") == "peak_aware":
        if not {"workingday", "hr"}.issubset(X.columns):
            raise ValueError("Peak-aware predictions require workingday and hr columns.")
        working_day = X["workingday"].eq(1)
        peak_masks = {
            "morning_peak": working_day & X["hr"].between(7, 9),
            "evening_peak": working_day & X["hr"].between(16, 19),
        }
        for name, mask in peak_masks.items():
            offsets[mask.to_numpy()] = item["peak_offsets"].get(name, item["offset"])
    return np.clip(item["model"].predict(X[bundle["feats"]]) + offsets, 0, None)


def predict_quantiles(bundle: dict, X: pd.DataFrame) -> dict[float, np.ndarray]:
    """Predict quantiles and prevent crossing after separate calibration."""
    quantiles = sorted(bundle["quantiles"])
    predictions = [predict_bundle(bundle, X, q) for q in quantiles]
    monotone = np.maximum.accumulate(np.vstack(predictions), axis=0)
    return {q: monotone[index] for index, q in enumerate(quantiles)}


def pinball(y, prediction, q: float) -> float:
    residual = np.asarray(y) - np.asarray(prediction)
    return float(np.mean(np.maximum(q * residual, (q - 1) * residual)))


def fleet_from_bikes_in_use(bikes, buffer: float = BUFFER) -> np.ndarray:
    """Convert estimated concurrent use to a scenario fleet requirement."""
    if buffer < 0:
        raise ValueError("Fleet buffer must be nonnegative.")
    return np.asarray(bikes) * (1 + buffer)
