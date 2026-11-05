"""Streamlit interface for bike-sharing demand and fleet scenarios."""

import json
import importlib.util
import os
from datetime import date

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from common import BUFFER, CAS_TRIP_MIN, OUT_DIR, REG_TRIP_MIN, build_features, complete_grid, load_raw

_assistant_spec = importlib.util.spec_from_file_location(
    "bike_ops_assistant", os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_ops_assistant.py")
)
if _assistant_spec is None or _assistant_spec.loader is None:
    raise ImportError("Could not load 04_ops_assistant.py.")
_assistant_module = importlib.util.module_from_spec(_assistant_spec)
_assistant_spec.loader.exec_module(_assistant_module)
Forecaster = _assistant_module.Forecaster
answer = _assistant_module.answer

ROOT = os.path.dirname(os.path.abspath(__file__))
WEATHER_OPTIONS = {
    "Clear / partly cloudy (1)": 1,
    "Mist / cloudy (2)": 2,
    "Light rain or snow (3)": 3,
    "Severe weather (4)": 4,
}

st.set_page_config(
    page_title="Bike Demand & Fleet Planning",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {background: #f4f7fb; padding: 1rem; border-radius: .6rem;}
    .limit-note {background: #fff7e6; border-left: 4px solid #e0a12f; padding: .8rem 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_features():
    path = os.path.join(OUT_DIR, "features.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["ts", "dteday"])
    hour, _ = load_raw()
    return build_features(complete_grid(hour))


@st.cache_resource(show_spinner=False)
def load_forecaster():
    return Forecaster()


def load_json(name):
    path = os.path.join(OUT_DIR, name)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def make_forecast(forecaster, selected_date, daytype_choice, start_hour, end_hour, weather, growth):
    kwargs = {
        "hours": list(range(start_hour, end_hour + 1)),
        "weather": weather,
        "month": selected_date.month,
        "growth": growth,
    }
    if daytype_choice == "Use selected date":
        kwargs["forecast_date"] = pd.Timestamp(selected_date)
    else:
        kwargs["daytype"] = {
            "Average workday": "workday",
            "Average weekend": "weekend",
            "Sunday": "sunday",
            "Holiday scenario": "holiday",
        }[daytype_choice]
    return forecaster.run(**kwargs)


def display_forecast(forecast):
    peak_row = forecast.loc[forecast["fleet_needed_p95"].idxmax()]
    first, second, third, fourth = st.columns(4)
    first.metric("P50 estimated bikes in use", f"{forecast['median_in_use'].max():.0f}")
    second.metric("P90 estimated bikes in use", f"{forecast['p90_in_use'].max():.0f}")
    third.metric("P95 estimated bikes in use", f"{forecast['p95_in_use'].max():.0f}")
    fourth.metric("P95 fleet scenario + buffer", f"{forecast['fleet_needed_p95'].max():.0f}")
    st.caption(
        f"Highest scenario hour: {int(peak_row['hour']):02d}:00. "
        "These are model estimates of average concurrent use; they are not station stock counts."
    )
    st.subheader("Estimated concurrent use by hour")
    chart = forecast.set_index("hour")[
        ["median_in_use", "p90_in_use", "p95_in_use"]
    ].rename(columns={
        "median_in_use": "P50 estimated bikes in use",
        "p90_in_use": "P90 estimated bikes in use",
        "p95_in_use": "P95 estimated bikes in use",
    })
    st.line_chart(chart, height=300)
    st.caption("X-axis: hour of day. Y-axis: estimated concurrently used bikes.")
    st.dataframe(
        forecast.rename(columns={
            "hour": "Hour", "median_in_use": "P50 in use",
            "p90_in_use": "P90 in use", "p95_in_use": "P95 in use",
            "fleet_needed_p95": "P95 fleet scenario + buffer",
        }).set_index("Hour"),
        width="stretch",
    )
    st.caption("Assumptions: " + json.dumps(forecast.attrs["assumed"], ensure_ascii=False))


def executive_dashboard(features, metrics, fleet_results):
    observed = features[features["is_observed"] & features["outage_day"].eq(0)]
    total = int(observed["cnt"].sum())
    mean_use = float(observed["bikes_in_use"].mean())
    peak_row = observed.loc[observed["cnt"].idxmax()]
    cols = st.columns(4)
    cols[0].metric("Recorded rentals", f"{total:,}")
    cols[1].metric("Mean estimated bikes in use", f"{mean_use:.1f}")
    cols[2].metric("Peak observed hourly rentals", f"{int(peak_row['cnt']):,}")
    selected = fleet_results.get("selected_point_model", "Not trained")
    cols[3].metric("Selected point model", selected)
    st.caption(
        "Rental totals exclude the two configured event dates; the bike-use estimate relies on "
        f"{REG_TRIP_MIN}/{CAS_TRIP_MIN}-minute registered/casual trip assumptions."
    )
    st.subheader("Demand trends")
    left, right = st.columns(2)
    with left:
        month_data = observed.groupby(["yr", "mnth"], as_index=False)["cnt"].mean()
        month_data["year"] = month_data["yr"].map({0: "2011", 1: "2012"})
        fig, ax = plt.subplots(figsize=(8, 3.5))
        sns.lineplot(data=month_data, x="mnth", y="cnt", hue="year", marker="o", ax=ax)
        ax.set(title="Average hourly rentals by month", xlabel="Month", ylabel="Rentals/hour")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        hourly = observed.groupby(["hr", "workingday"], as_index=False)["cnt"].mean()
        hourly["Day type"] = hourly["workingday"].map({1: "Working day", 0: "Non-working day"})
        fig, ax = plt.subplots(figsize=(8, 3.5))
        sns.lineplot(data=hourly, x="hr", y="cnt", hue="Day type", marker="o", ax=ax)
        ax.set(title="Hourly rentals by day type", xlabel="Hour", ylabel="Rentals/hour")
        st.pyplot(fig)
        plt.close(fig)
    st.subheader("Held-out model performance")
    if metrics.empty:
        st.info("Model metrics are not available. Run 03_fleet_model.py to train and evaluate.")
    else:
        st.dataframe(metrics.round(3), width="stretch")
        st.caption(
            "Frozen OPS day-ahead model: native-quantile HistGradientBoosting with the "
            "enhanced OPS feature set and peak-aware finite-sample calibration. The "
            "interactive scenario controls use the separate PLAN calendar/weather model; "
            "they do not supply OPS historical-lag inputs."
        )
        st.warning(
            "OPS test metrics use realized target-hour weather from the historical data. "
            "They are conditional on realized weather, not an evaluation using weather "
            "forecasts available at operational issue time."
        )
        segment_path = os.path.join(OUT_DIR, "ops_test_metrics_by_period.csv")
        if os.path.exists(segment_path):
            st.markdown("**Frozen OPS test metrics by operating period**")
            segment_metrics = pd.read_csv(segment_path).set_index("segment")
            st.dataframe(segment_metrics.round(3), width="stretch")
            st.caption(
                "The final OPS evaluation report is saved at "
                "outputs/final_evaluation_report.md."
            )
    with st.expander("Interpretation limits"):
        st.markdown(
            '<div class="limit-note">The dataset has no station inventories, trip durations, failed rental attempts, '
            'or actual fleet size. The estimated bikes-in-use and fleet requirement are conditional scenarios, '
            'not evidence of actual availability or a purchase recommendation.</div>',
            unsafe_allow_html=True,
        )


def demand_analysis(features):
    observed = features[features["is_observed"] & features["outage_day"].eq(0)]
    st.subheader("Historical demand patterns")
    tab_hour, tab_weather, tab_peak = st.tabs(["Hourly profiles", "Weather association", "Peak vs off-peak"])
    with tab_hour:
        by_day = observed.groupby(["hr", "workingday"], as_index=False)["cnt"].mean()
        by_day["Day type"] = by_day["workingday"].map({1: "Working day", 0: "Non-working day"})
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.lineplot(data=by_day, x="hr", y="cnt", hue="Day type", marker="o", ax=ax)
        ax.set(title="Average rentals by hour and day type", xlabel="Hour", ylabel="Rentals/hour")
        st.pyplot(fig)
        plt.close(fig)
        daily = observed.groupby("workingday")["cnt"].agg(["mean", "median", "count"])
        daily.index = daily.index.map({1: "Working day", 0: "Non-working day"})
        st.dataframe(daily.round(2), width="stretch")
    with tab_weather:
        weather = observed.groupby("weathersit")["cnt"].agg(
            mean="mean", median="median", count="count"
        ).reset_index().rename(columns={"weathersit": "Source weather code"})
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(
            data=weather,
            x="Source weather code",
            y="mean",
            color="#4472C4",
            ax=ax,
        )
        ax.set(
            title="Observed hourly rentals by UCI weather category",
            xlabel="UCI source weather code",
            ylabel="Mean completed rentals/hour",
        )
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(weather.round(2), width="stretch")
        st.caption(
            "Codes: 1=clear/partly cloudy, 2=mist/cloudy, 3=light rain or snow, "
            "4=severe weather. Descriptive association only; categories may differ "
            "in other conditions and seasons."
        )
    with tab_peak:
        result = load_json("hypothesis_results.json")
        if result:
            st.write(f"**H0:** {result.get('H0')}")
            st.write(f"**H1:** {result.get('H1')}")
            a, b, c = st.columns(3)
            a.metric("Median peak / off-peak ratio", f"{result['median_peak_to_offpeak_ratio']:.2f}×")
            b.metric("Paired Wilcoxon p-value", f"{result['p_value']:.3g}")
            c.metric("Complete paired days", result["assumption_checks"]["paired_observations"])
            st.write(
                "Bootstrap 95% CI for median ratio: "
                f"{result['median_ratio_block_bootstrap_ci95'][0]:.2f}–"
                f"{result['median_ratio_block_bootstrap_ci95'][1]:.2f}. "
                + result["historical_evidence_limit"]
            )
        else:
            st.info("Run 02_hypothesis_test.py to save the paired historical analysis.")


def forecast_tab(forecaster, key_prefix):
    st.subheader("Demand scenario")
    first, second, third = st.columns(3)
    with first:
        selected_date = st.date_input(
            "Scenario date / month", value=date(2012, 7, 1), key=f"{key_prefix}_date"
        )
        daytype = st.selectbox(
            "Day type",
            ["Use selected date", "Average workday", "Average weekend", "Sunday", "Holiday scenario"],
            key=f"{key_prefix}_daytype",
        )
    with second:
        start_hour, end_hour = st.slider(
            "Hour range", 0, 23, (7, 19), key=f"{key_prefix}_hours"
        )
        weather_label = st.selectbox(
            "Weather scenario", list(WEATHER_OPTIONS), key=f"{key_prefix}_weather"
        )
        weather = WEATHER_OPTIONS[weather_label]
    with third:
        growth_percent = st.number_input(
            "Demand growth versus historical pattern (%)",
            min_value=-90.0, max_value=500.0, value=0.0, step=5.0,
            key=f"{key_prefix}_growth",
        )
        growth = 1 + growth_percent / 100
        st.caption("Weather covariates use month/weather medians from the pre-test training data.")
    forecast = make_forecast(
        forecaster, selected_date, daytype, start_hour, end_hour, weather, growth
    )
    display_forecast(forecast)
    st.caption(
        "PLAN scenario output: calendar/weather-only model with no demand lags. "
        "Weather conditions are represented by pre-test historical month/weather medians; "
        "this is not an OPS day-ahead forecast."
    )
    return forecast, selected_date, daytype, start_hour, end_hour, weather, growth


def fleet_planning(forecast, selected_date, daytype, start_hour, end_hour, weather, growth):
    st.subheader("System-level capacity scenario")
    st.caption(
        "Enter the company's actual active fleet count. Because the CSVs have no station-level data, "
        "this comparison cannot determine whether bikes are misplaced."
    )
    current_fleet = st.number_input("Current active fleet size (0 = not provided)", min_value=0, value=0, step=10)
    if current_fleet == 0:
        st.info("Enter the actual fleet size to compare it with the modelled scenario.")
        return
    requirement = int(forecast["fleet_needed_p95"].max())
    cols = st.columns(3)
    cols[0].metric("Current fleet (user input)", f"{current_fleet:,}")
    cols[1].metric("Estimated P95 + buffer", f"{requirement:,}")
    cols[2].metric("Difference (fleet - scenario)", f"{current_fleet - requirement:+,}")
    if current_fleet < requirement:
        st.warning(
            "The entered fleet is below this modeled scenario. This alone does not establish a shortage. "
            "First validate trip-duration and maintenance assumptions and instrument failed attempts and "
            "station inventory. If unmet demand is localized while usable bikes are idle elsewhere, pilot "
            "redistribution; consider expansion only if measured unmet demand persists and the economics support it."
        )
    else:
        st.info(
            "The entered fleet is at or above this system-level scenario. This does not rule out "
            "station-specific stock-outs. Monitor station inventory and failed attempts; pilot redistribution "
            "only if measured imbalances support it. This aggregate estimate cannot decide a purchase."
        )


def assistant_tab(forecaster):
    st.subheader("Ask a demand or fleet scenario")
    st.caption("Every numeric forecast is produced by the saved model and includes its assumptions.")
    question = st.text_input(
        "Question",
        placeholder="How many bikes on a rainy Tuesday at 8am in October with 20% growth?",
    )
    if st.button("Get model-backed answer", type="primary") and question.strip():
        st.write(answer(question, forecaster))
    with st.expander("Example questions"):
        st.markdown(
            "- How many bikes do we need on a weekday morning rush?\n"
            "- Forecast for 2012-10-15 at 17:00, clear weather.\n"
            "- What if it rains on a weekday at 8am?\n"
            "- When is demand highest on weekends in July?\n"
            "- Compare a fleet of 300 with P95 need on a rainy Sunday.\n"
            "- What happens with 20% growth next summer?"
        )


def main():
    st.title("Bike Sharing Demand & Fleet Planning")
    st.caption("UCI Bike Sharing Dataset | Historical evidence + modelled scenarios")
    try:
        features = load_features()
    except (FileNotFoundError, ValueError, OSError) as error:
        st.error(f"Could not load the project data: {error}")
        st.info("Place hour.csv and day.csv in the data/ directory.")
        return

    metrics_path = os.path.join(OUT_DIR, "model_metrics.csv")
    metrics = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else pd.DataFrame()
    fleet_results = load_json("fleet_results.json")
    try:
        forecaster = load_forecaster()
    except (FileNotFoundError, KeyError, OSError) as error:
        forecaster = None
        st.warning(f"Forecast model not ready: {error} Run `python 03_fleet_model.py` first.")

    names = [
        "Executive Dashboard", "Demand Forecasting", "Demand Analysis",
        "Fleet Planning", "AI Operations Assistant",
    ]
    tabs = st.tabs(names)
    with tabs[0]:
        executive_dashboard(features, metrics, fleet_results)
    with tabs[1]:
        if forecaster is not None:
            forecast_tab(forecaster, "forecast")
    with tabs[2]:
        demand_analysis(features)
    with tabs[3]:
        if forecaster is not None:
            result = forecast_tab(forecaster, "fleet")
            fleet_planning(*result)
    with tabs[4]:
        if forecaster is not None:
            assistant_tab(forecaster)
    st.divider()
    st.caption(
        f"Little's Law assumptions: registered trip {REG_TRIP_MIN} min; casual trip "
        f"{CAS_TRIP_MIN} min; fleet buffer {BUFFER:.0%}. No station-level availability claims."
    )


if __name__ == "__main__":
    main()
