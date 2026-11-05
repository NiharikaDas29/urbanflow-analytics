"""Data validation, missing-hour diagnostics, feature engineering, and EDA."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import (
    BUFFER, CAS_TRIP_MIN, OUT_DIR, PLAN_FEATS, OPS_FEATS, POTENTIAL_OUTAGE_MIN_HOURS,
    REG_TRIP_MIN, OUTAGE_DAYS,
    build_features, complete_grid, load_raw,
)

SEASON_CODES = {
    1: "Spring (UCI source category)",
    2: "Summer (UCI source category)",
    3: "Fall (UCI source category)",
    4: "Winter (UCI source category)",
}
WEATHER_CODES = {
    1: "Clear, few clouds, partly cloudy",
    2: "Mist and/or cloudy conditions",
    3: "Light snow or light rain with scattered clouds",
    4: "Heavy rain/ice pellets/thunderstorm/mist or snow/fog",
}
SOURCE_DESCRIPTIONS = {
    "instant": "Source row identifier; not a demand predictor.",
    "dteday": "Calendar date associated with the hourly or daily record.",
    "season": "UCI season category code: 1 spring, 2 summer, 3 fall, 4 winter.",
    "yr": "Dataset year indicator: 0=2011, 1=2012.",
    "mnth": "Calendar month, 1 through 12.",
    "hr": "Hour of day, 0 through 23; hourly table only.",
    "holiday": "Whether the date is a source-calendar holiday.",
    "weekday": "Day of week, 0=Sunday through 6=Saturday.",
    "workingday": "Whether the source marks the date as a working day.",
    "weathersit": "UCI weather category code; see weather_code_definitions.",
    "temp": "Normalized air temperature; approximately Celsius = temp × 41.",
    "atemp": "Normalized feels-like temperature; approximately Celsius = atemp × 50.",
    "hum": "Normalized relative humidity; approximately percent = hum × 100.",
    "windspeed": "Normalized wind speed; approximately km/h = windspeed × 67.",
    "casual": "Observed completed rentals by casual users.",
    "registered": "Observed completed rentals by registered users.",
    "cnt": "Observed total completed rentals; equals casual + registered in source data.",
}
ENGINEERED_DESCRIPTIONS = {
    "was_missing": "1 when the hourly timestamp is absent from the source; demand remains unknown.",
    "is_observed": "1 only for source-observed hourly rows; used to exclude unknown targets.",
    "outage_day": "Configured October 29–30, 2012 event-date flag; not a measured outage label.",
    "potential_outage": "Missing-hour candidate flag from event dates or a >=6-hour gap; unconfirmed.",
    "context_imputed": "Marks calendar/weather context filled only to render the complete grid.",
    "temp_c": "Approximate Celsius temperature for interpretable relationships and OPS modeling.",
    "atemp_c": "Approximate Celsius feels-like temperature.",
    "hum_pct": "Approximate relative humidity percentage.",
    "wind_kmh": "Approximate wind speed in km/h.",
    "wind_reported_zero": "Source windspeed equals zero; quality flag retained for review.",
    "hr_sin": "Sine cyclic encoding of hour so adjacent clock hours remain close in feature space.",
    "hr_cos": "Cosine cyclic encoding of hour so adjacent clock hours remain close in feature space.",
    "rain_or_snow": "Indicator for source weather codes 3 or 4.",
    "commute_peak": "Working-day indicator for the configured 07–09 and 16–19 hours.",
    "weekend_daytime": "Non-working-day indicator for daytime hours 10–17.",
    "mnth_sin": "Sine cyclic encoding of month to represent year-boundary continuity.",
    "mnth_cos": "Cosine cyclic encoding of month to represent year-boundary continuity.",
    "hr_month_sin_sin": "Interaction between hourly sine and monthly sine.",
    "hr_month_sin_cos": "Interaction between hourly sine and monthly cosine.",
    "hr_month_cos_sin": "Interaction between hourly cosine and monthly sine.",
    "hr_month_cos_cos": "Interaction between hourly cosine and monthly cosine.",
    "temp_weather_2": "Temperature interaction for weather category 2.",
    "temp_weather_3": "Temperature interaction for weather category 3.",
    "temp_weather_4": "Temperature interaction for weather category 4.",
    "casual_share": "Descriptive target-derived rider mix; not included in PLAN or OPS predictor lists.",
    "bikes_in_use": "Little's Law estimate from completed rentals and assumed 12/30-minute trip durations; not observed inventory.",
    "lag_24": "Estimated use at t−24h; latest historical target available to day-ahead OPS.",
    "lag_168": "Estimated use at the same hour one week earlier (t−168h).",
    "lag_336": "Estimated use at the same hour two weeks earlier (t−336h).",
    "roll24": "Mean target estimate over t−47h through t−24h (at least 18 observed values).",
    "roll168": "Mean target estimate over t−191h through t−24h (at least 120 observed values).",
    "demand_trend_7d": "Difference between recent and preceding seven-day means, both ending before target time.",
}


def _gap_intervals(hour):
    observed = pd.DatetimeIndex(hour["ts"].dropna().sort_values().unique())
    full = pd.date_range(observed.min(), observed.max(), freq="h")
    missing = full.difference(observed)
    if len(missing) == 0:
        return []
    starts = np.r_[True, np.diff(missing.asi8) != pd.Timedelta(hours=1).value]
    groups = np.cumsum(starts) - 1
    intervals = []
    for group in np.unique(groups):
        gap = missing[groups == group]
        intervals.append({
            "start": gap[0].isoformat(),
            "end": gap[-1].isoformat(),
            "hours": int(len(gap)),
            "classification": (
                "known_event_date_candidate"
                if gap[0].strftime("%Y-%m-%d") in {"2012-10-29", "2012-10-30"}
                else "long_gap_potential_outage_unconfirmed"
                if len(gap) >= POTENTIAL_OUTAGE_MIN_HOURS
                else "unknown_missing_observation"
            ),
        })
    return intervals


def quality_checks(hour, day):
    """Return data validation results without imputing missing rental outcomes."""
    issues = []

    def frame_profile(frame):
        numeric = frame.select_dtypes(include=[np.number])
        return {
            "dtypes": {name: str(dtype) for name, dtype in frame.dtypes.items()},
            "missing_by_column": {
                name: {
                    "count": int(frame[name].isna().sum()),
                    "percent": float(frame[name].isna().mean() * 100),
                }
                for name in frame.columns
            },
            "numeric_distributions": {
                name: {
                    key: (None if pd.isna(value) else float(value))
                    for key, value in numeric[name].describe(
                        percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]
                    ).items()
                }
                for name in numeric.columns
            },
        }

    report = {
        "shape_hour": list(hour.shape),
        "shape_day": list(day.shape),
        "date_range_hour": {
            "start": hour["dteday"].min().isoformat(),
            "end": hour["dteday"].max().isoformat(),
        },
        "date_range_day": {
            "start": day["dteday"].min().isoformat(),
            "end": day["dteday"].max().isoformat(),
        },
        "null_cells_hour": int(hour.isna().sum().sum()),
        "null_cells_day": int(day.isna().sum().sum()),
        "duplicate_rows_hour": int(hour.duplicated().sum()),
        "duplicate_rows_day": int(day.duplicated().sum()),
        "duplicate_timestamps": int(hour["ts"].duplicated().sum()),
        "hour_table_profile": frame_profile(hour),
        "day_table_profile": frame_profile(day),
        "season_code_definitions": {str(code): value for code, value in SEASON_CODES.items()},
        "weather_code_definitions": {str(code): value for code, value in WEATHER_CODES.items()},
    }

    range_checks = {
        "hr": (0, 23), "weekday": (0, 6), "season": (1, 4), "mnth": (1, 12),
        "weathersit": (1, 4), "holiday": (0, 1), "workingday": (0, 1),
        "yr": (0, 1), "temp": (0, 1), "atemp": (0, 1), "hum": (0, 1),
        "windspeed": (0, 1), "casual": (0, np.inf), "registered": (0, np.inf),
        "cnt": (0, np.inf),
    }
    report["range_violations"] = {}
    for column, (minimum, maximum) in range_checks.items():
        invalid = ~hour[column].between(minimum, maximum)
        report["range_violations"][column] = int(invalid.sum())
    report["normalized_weather_columns_within_0_1"] = all(
        report["range_violations"][column] == 0
        for column in ("temp", "atemp", "hum", "windspeed")
    )
    report["count_sum_mismatches"] = int(
        (hour["casual"] + hour["registered"] != hour["cnt"]).sum()
    )
    report["non_integer_or_negative_count_violations"] = {
        column: int((hour[column].lt(0) | hour[column].mod(1).ne(0)).sum())
        for column in ("casual", "registered", "cnt")
    }
    expected_weekday = (hour["dteday"].dt.day_of_week + 1) % 7
    report["weekday_calendar_mismatches"] = int((expected_weekday != hour["weekday"]).sum())
    report["month_calendar_mismatches"] = int((hour["dteday"].dt.month != hour["mnth"]).sum())
    report["year_calendar_mismatches"] = int(
        ((hour["dteday"].dt.year - 2011) != hour["yr"]).sum()
    )
    report["workingday_weekend_or_holiday"] = int(
        (
            hour["workingday"].eq(1)
            & (hour["weekday"].isin([0, 6]) | hour["holiday"].eq(1))
        ).sum()
    )
    season_counts = hour.groupby(["mnth", "season"]).size().unstack(fill_value=0)
    report["season_code_counts_by_month"] = {
        str(int(month)): {str(int(season)): int(count) for season, count in row.items()}
        for month, row in season_counts.iterrows()
    }
    report["weather_code_counts"] = {
        str(int(k)): int(v) for k, v in hour["weathersit"].value_counts().sort_index().items()
    }
    report["observed_zero_rental_rows"] = int(hour["cnt"].eq(0).sum())
    report["windspeed_reported_zero_rows"] = int(hour["windspeed"].eq(0).sum())
    report["humidity_zero_rows"] = int(hour["hum"].eq(0).sum())

    intervals = _gap_intervals(hour)
    report["expected_hourly_rows_between_observed_endpoints"] = int(
        (hour["ts"].max() - hour["ts"].min()) / pd.Timedelta(hours=1) + 1
    )
    report["missing_timestamp_count"] = int(sum(item["hours"] for item in intervals))
    report["missing_timestamp_intervals"] = intervals
    report["potential_outage_gap_threshold_hours"] = POTENTIAL_OUTAGE_MIN_HOURS
    report["potential_outage_candidate_intervals"] = int(sum(
        item["classification"] != "unknown_missing_observation" for item in intervals
    ))
    observed_timestamps = pd.DatetimeIndex(hour["ts"].dropna().sort_values().unique())
    missing_timestamps = pd.date_range(
        observed_timestamps.min(), observed_timestamps.max(), freq="h"
    ).difference(observed_timestamps)
    report["missing_hours_by_hour_of_day"] = {
        str(int(k)): int(v)
        for k, v in pd.Series(missing_timestamps.hour).value_counts().sort_index().items()
    }
    report["daily_total_mismatches_hour_vs_day"] = int(
        hour.groupby("dteday")["cnt"].sum().reindex(day["dteday"]).to_numpy().__ne__(
            day["cnt"].to_numpy()
        ).sum()
    )
    report["timestamp_hour_mismatches"] = int(
        (hour["ts"] != hour["dteday"] + pd.to_timedelta(hour["hr"], unit="h")).sum()
    )

    cell = hour.groupby(["hr", "workingday"])["cnt"]
    q1 = cell.transform(lambda values: values.quantile(0.25))
    q3 = cell.transform(lambda values: values.quantile(0.75))
    outliers = hour["cnt"] > q3 + 3 * (q3 - q1)
    report["high_outlier_rows_3_iqr_by_hour_and_day_type"] = int(outliers.sum())
    report["outliers_removed"] = 0
    report["outlier_policy"] = (
        "Flagged within hour/day-type cells for review; genuine demand peaks are retained."
    )
    report["missing_data_policy"] = (
        "Observed cnt=0 is retained as zero. Missing timestamps remain unknown and are not "
        f"imputed as zero. Gaps of {POTENTIAL_OUTAGE_MIN_HOURS} or more consecutive hours "
        "are marked as unconfirmed outage candidates. October 29-30, 2012 are marked as "
        "known-event-date candidates."
    )
    if report["missing_timestamp_count"]:
        issues.append(
            f"{report['missing_timestamp_count']} timestamps are absent; all remain unknown "
            "rather than being treated as zero rentals."
        )
    if report["potential_outage_candidate_intervals"]:
        issues.append(
            f"{report['potential_outage_candidate_intervals']} gaps are flagged as candidate "
            "outages based on the configured event dates or a run of at least "
            f"{POTENTIAL_OUTAGE_MIN_HOURS} missing hours; none is declared a confirmed outage "
            "from hourly counts alone."
        )
    if report["count_sum_mismatches"]:
        issues.append(f"{report['count_sum_mismatches']} rows have cnt != casual + registered.")
    if report["weekday_calendar_mismatches"]:
        issues.append("Dataset weekday values do not match the calendar for some rows.")
    if report["humidity_zero_rows"]:
        issues.append(
            f"{report['humidity_zero_rows']} rows report zero humidity; retained and flagged "
            "for source validation."
        )
    issues.append(
        "Season values are reported as source codes by month; no relabeling is inferred."
    )
    report["issues_and_decisions"] = issues
    return report


def feature_dictionary():
    """Build a reproducible data dictionary and document predictor leakage boundaries."""
    descriptions = dict(SOURCE_DESCRIPTIONS)
    descriptions.update(ENGINEERED_DESCRIPTIONS)
    return {
        "source": {
            "name": "UCI Bike Sharing Dataset",
            "url": "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset",
            "tables": ["hour.csv", "day.csv"],
        },
        "season_code_definitions": {str(code): value for code, value in SEASON_CODES.items()},
        "weather_code_definitions": {str(code): value for code, value in WEATHER_CODES.items()},
        "source_variables": SOURCE_DESCRIPTIONS,
        "engineered_variables": {
            column: descriptions[column]
            for column in descriptions
            if column not in SOURCE_DESCRIPTIONS
        },
        "feature_sets": {
            "PLAN": {
                "columns": PLAN_FEATS,
                "purpose": "Calendar/weather-only scenarios; no demand lags.",
            },
            "OPS": {
                "columns": OPS_FEATS,
                "purpose": (
                    "Rolling day-ahead use estimates with history cut off at t−24h; "
                    "all rolling windows use earlier observed targets only."
                ),
            },
        },
        "target_and_assumptions": {
            "observed_demand": "cnt, casual, and registered are completed rentals observed by the source.",
            "estimated_use": "bikes_in_use is derived with Little's Law and assumed rider-specific trip durations.",
            "trip_duration_minutes": {"registered": REG_TRIP_MIN, "casual": CAS_TRIP_MIN},
            "fleet_buffer": BUFFER,
            "not_observed": [
                "station inventory", "failed rental attempts", "actual fleet size",
                "trip durations",
            ],
        },
        "leakage_controls": [
            "casual_share and bikes_in_use are target-derived and are not predictor features.",
            "PLAN_FEATS contains no demand lags.",
            "OPS historical targets are shifted so inputs end at t−24h.",
            "Missing target observations and configured event-date targets are excluded from OPS history.",
        ],
    }


def eda_figures(g):
    """Save demand, calendar/weather, and timestamp-gap plots."""
    sns.set_theme(style="whitegrid", context="notebook")
    observed = g[g["is_observed"] & g["outage_day"].eq(0)].copy()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for workingday, label in ((1, "Working day"), (0, "Non-working day")):
        means = observed[observed.workingday.eq(workingday)].groupby("hr")["cnt"].mean()
        axes[0, 0].plot(means.index, means.values, marker="o", label=label)
    axes[0, 0].set(title="Average hourly rentals by day type", xlabel="Hour", ylabel="Rentals/hour")
    axes[0, 0].legend()

    sns.lineplot(data=observed, x="mnth", y="cnt", hue="yr", estimator="mean", errorbar=None, ax=axes[0, 1])
    axes[0, 1].set(title="Mean hourly rentals by month", xlabel="Month", ylabel="Rentals/hour")
    axes[0, 1].legend(title="Dataset year", labels=["2011", "2012"])

    weather = observed.groupby("weathersit", as_index=False)["cnt"].mean()
    sns.barplot(data=weather, x="weathersit", y="cnt", color="#4472C4", ax=axes[0, 2])
    axes[0, 2].set(title="Mean rentals by source weather code", xlabel="Weather code", ylabel="Rentals/hour")

    heatmap = observed.pivot_table(index="weekday", columns="hr", values="cnt", aggfunc="mean")
    sns.heatmap(heatmap, cmap="YlGnBu", ax=axes[1, 0], cbar_kws={"label": "Rentals/hour"})
    axes[1, 0].set(title="Mean rentals by weekday and hour", xlabel="Hour", ylabel="Weekday (0=Sun)")

    rider = observed.groupby("hr")[["registered", "casual"]].mean().reset_index().melt(
        id_vars="hr", var_name="rider_type", value_name="rentals"
    )
    sns.lineplot(data=rider, x="hr", y="rentals", hue="rider_type", marker="o", ax=axes[1, 1])
    axes[1, 1].set(title="Registered and casual demand profiles", xlabel="Hour", ylabel="Rentals/hour")

    sns.scatterplot(data=observed, x="temp_c", y="cnt", hue="weathersit", alpha=0.35, s=18, ax=axes[1, 2])
    axes[1, 2].set(title="Demand and temperature (descriptive)", xlabel="Temperature (°C)", ylabel="Rentals/hour")
    fig.suptitle("Bike-sharing demand: exploratory analysis", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_01_eda.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 5))
    outlier_plot = observed.copy()
    outlier_plot["Day type"] = outlier_plot["workingday"].map(
        {1: "Working day", 0: "Non-working day"}
    )
    sns.boxplot(
        data=outlier_plot,
        x="hr",
        y="cnt",
        hue="Day type",
        showfliers=True,
        ax=ax,
    )
    ax.set(
        title="Hourly rental distributions by day type (flagged values retained)",
        xlabel="Hour of day",
        ylabel="Completed rentals/hour",
    )
    ax.legend(title="Day type", loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_01_outliers.png"), dpi=160)
    plt.close(fig)

    missing = g[g["was_missing"].eq(1)]
    fig, ax = plt.subplots(figsize=(13, 3.5))
    if missing.empty:
        ax.text(0.5, 0.5, "No missing timestamps in this dataset", ha="center", va="center")
    else:
        ax.scatter(missing["ts"], np.ones(len(missing)), s=9, color="#C44E52")
        ax.set_yticks([])
    ax.set(title="Absent hourly timestamps (unknown demand, not imputed zeros)", xlabel="Timestamp")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_01_missing_hours.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    hour, day = load_raw()
    report = quality_checks(hour, day)
    grid = complete_grid(hour)
    features = build_features(grid)
    eda_figures(features)

    valid = features[features["is_observed"] & features["outage_day"].eq(0)]
    rental_total = float(valid["cnt"].sum())
    clear_mean = valid.loc[valid["weathersit"].eq(1), "cnt"].mean()
    rain_mean = valid.loc[valid["weathersit"].ge(3), "cnt"].mean()
    report["eda"] = {
        "observed_rentals_excluding_known_event_dates": int(rental_total),
        "rentals_by_dataset_year": {
            str(int(year) + 2011): int(rows["cnt"].sum())
            for year, rows in valid.groupby("yr")
        },
        "registered_share": float(valid["registered"].sum() / rental_total) if rental_total else None,
        "rain_or_snow_vs_clear_mean_ratio": (
            float(rain_mean / clear_mean) if pd.notna(clear_mean) and clear_mean else None
        ),
        "temperature_correlation_with_rentals": float(valid["cnt"].corr(valid["temp_c"])),
        "humidity_correlation_with_rentals": float(valid["cnt"].corr(valid["hum_pct"])),
        "outage_dates_excluded": list(OUTAGE_DAYS),
        "bike_use_estimate_assumptions": {
            "registered_trip_minutes": REG_TRIP_MIN,
            "casual_trip_minutes": CAS_TRIP_MIN,
            "fleet_buffer": BUFFER,
        },
    }
    report["dataset_identity"] = {
        "name": "UCI Bike Sharing Dataset",
        "source_url": "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset",
        "hour_table_rows_columns": list(hour.shape),
        "day_table_rows_columns": list(day.shape),
        "source_unit": (
            "Hourly and daily completed rental counts; not station-level inventory."
        ),
    }
    with open(os.path.join(OUT_DIR, "quality_report.json"), "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, default=str)
    features.to_csv(os.path.join(OUT_DIR, "features.csv"), index=False)
    with open(os.path.join(OUT_DIR, "data_dictionary.json"), "w", encoding="utf-8") as file:
        json.dump(feature_dictionary(), file, indent=2)
    print(json.dumps(report, indent=2, default=str))
    print(
        f"\nSaved features.csv ({features.shape}), quality_report.json, "
        f"data_dictionary.json and figures in {OUT_DIR}"
    )
