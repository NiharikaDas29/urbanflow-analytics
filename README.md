# UrbanFlow Analytics
### Bike Sharing Demand Forecasting, Statistical Analysis & Fleet Optimization

An end-to-end business analytics and machine learning project that transforms historical bike-sharing data into demand forecasts, fleet utilization estimates, and actionable operational insights.

Built using the UCI Bike Sharing Dataset (2011–2012), this project explores demand patterns, tests peak-hour demand differences, develops quantile-based forecasting models, and provides an interactive Streamlit application for operational decision-making.

---

## Project Overview

UrbanFlow Analytics addresses a practical business problem:

**How can a bike-sharing operator anticipate rental demand, estimate fleet utilization, and make better operational planning decisions?**

The solution combines statistical analysis, predictive modeling, fleet estimation, and an interactive operations interface.

### Key Results

| Metric | Result |
|---|---:|
| Hourly observations | 17,379 |
| Daily observations | 731 |
| Complete working days in hypothesis test | 495 |
| Median peak/off-peak rental ratio | 3.15× |
| Wilcoxon test p-value | 4.736 × 10⁻⁸³ |
| OPS model MAE | 12.59 |
| OPS model RMSE | 19.60 |
| OPS model WAPE | 20.78% |
| OPS model R² | 0.880 |
| Held-out test observations | 2,730 |

*These are historical dataset results, not guarantees of future operational performance.*

---

## Business Objectives

The project addresses four core objectives:

1. **Data Quality & Feature Engineering:** Identify data inconsistencies, missing timestamps, and useful demand-related features.
2. **Statistical Hypothesis Testing:** Determine whether working-day peak-hour demand is systematically higher than selected off-peak demand.
3. **Demand Forecasting & Fleet Planning:** Estimate concurrent bike utilization and develop demand prediction models with uncertainty estimates.
4. **Operational Decision Support:** Build an interactive application through which an operations manager can explore demand scenarios and ask natural-language questions.

---

## Solution Architecture

The project follows a modular analytics pipeline:

1. Raw bike-sharing data
2. Data validation and exploratory analysis
3. Feature engineering
4. Statistical hypothesis testing
5. Chronological model development and calibration
6. Fleet utilization estimation
7. Operations assistant and Streamlit dashboard

---

## 1. Data Quality & Exploratory Analysis

The project uses the UCI Bike Sharing Dataset containing hourly and daily rental records for Washington, D.C., from 2011 to 2012.

**Data quality checks include:**
- Missing values and duplicate records.
- Missing timestamps and chronological consistency.
- Data types, valid ranges, and calendar agreement.
- Consistency between casual, registered, and total rentals.
- Outlier detection across hour and day-type combinations.

**Feature engineering includes:**
- Conversion of normalized weather variables into interpretable units.
- Cyclic encoding of hour-related patterns.
- Historical demand lags at 24, 168, and 336 hours.
- Rolling demand averages and weekly trend features.
- Hour, month, temperature, and weather interactions.

Missing timestamps are treated as unknown demand rather than zero rentals. Observed zero-rental records are retained, and outliers are flagged rather than automatically removed.

---

## 2. Statistical Hypothesis Testing

**Research question:** Is typical working-day peak-hour rental demand higher than selected off-peak demand?

Peak hours:
- 08:00
- 17:00
- 18:00

Off-peak hours:
- 10:00–15:00
- 21:00–23:00

A one-sided paired Wilcoxon signed-rank test was performed using 495 complete working days.

**Hypotheses:**

- H₀: The paired peak-minus-off-peak differences have no positive location shift.
- H₁: The paired differences have a positive location shift.

**Results:**

| Statistical measure | Result |
|---|---:|
| Complete working days | 495 |
| Wilcoxon statistic | 122,738 |
| p-value | 4.736 × 10⁻⁸³ |
| Median paired difference | 338.67 rentals/hour |
| Median peak/off-peak ratio | 3.15× |
| 95% bootstrap CI for ratio | 3.12–3.18 |

The findings provide strong historical evidence of concentrated peak-hour rental demand.

The result describes completed rentals; it does not establish that customers experienced stockouts or that fleet redistribution would causally improve service.

---

## 3. Demand Forecasting & Fleet Optimization

### Fleet Utilization Estimation

The project applies Little's Law:

**L = λ × W**

<p align="center">
  <strong>Estimated Concurrent Bikes in Use</strong>
</p>

<p align="center">
  <i>L</i> =
  (<i>R</i> × <i>T</i><sub>r</sub> +
  <i>C</i> × <i>T</i><sub>c</sub>) / 60
</p>

Where:
- R = Registered rentals per hour.
- C = Casual rentals per hour.
- Tᵣ = Assumed registered trip duration.
- T꜀ = Assumed casual trip duration.

Default assumptions:
- Registered trip duration: 12 minutes.
- Casual trip duration: 30 minutes.
- Additional fleet buffer: 15%.

These are planning assumptions, not measured trip durations or actual fleet inventory.

### Two Forecasting Approaches

**PLAN Model**
- Calendar and weather-based forecasting.
- Designed for arbitrary future scenarios.
- Does not depend on historical demand lags.
- Supports interactive scenario analysis.

**OPS Model**
- Rolling day-ahead demand forecasting.
- Uses historical demand lags and engineered features.
- Employs HistGradientBoostingRegressor with native quantile loss.
- Produces P50, P90, and P95 demand estimates.
- Uses peak-aware residual calibration.

### Chronological Evaluation

The project avoids random train-test splitting to preserve time order.

| Stage | Period |
|---|---|
| Initial model comparison | Before April 2022? |

**Chronological split used in the actual experiment:**

| Stage | Period |
|---|---|
| Initial point-model comparison fit | Before April 1, 2012 |
| Initial validation | April–May 2012 |
| Final model fit | Before June 1, 2012 |
| Calibration | June–August 2012 |
| Final held-out test | September–December 2012 |

The final test period was evaluated once after model and calibration decisions were frozen.

### Final OPS Model Performance

| Metric | Test Result |
|---|---:|
| MAE | 12.59 |
| RMSE | 19.60 |
| WAPE | 20.78% |
| R² | 0.880 |
| P50 coverage | 56.45% |
| P90 coverage | 89.27% |
| P95 coverage | 93.52% |

The P90 and P95 coverage values are below their nominal levels. Evening peak performance also shows weaker P95 coverage, which is important when interpreting the model for operational planning.

**Evaluation limitation:** OPS test metrics use realized target-hour weather covariates from the historical dataset. They do not represent a fully operational forecast using weather forecasts available at forecast issue time.

---

## 4. Operations Assistant

The project includes a rule-based operations assistant that translates natural-language questions into model-backed demand insights.

It supports questions involving:
- Dates and weekdays.
- Weekends and working days.
- Specific hours and time ranges.
- Months and weather categories.
- Demand growth and reduction scenarios.
- Fleet-size comparisons.

All numerical answers are generated using the saved PLAN model artifact. The assistant does not require an LLM, external API, or paid service.

It explicitly communicates assumptions and provides explanations for unsupported queries rather than inventing results.

---

## 5. Interactive Streamlit Dashboard

The application provides five main sections:

| Module | Functionality |
|---|---|
| Executive Dashboard | Overview of demand patterns and model performance |
| Demand Forecasting | Explore demand under different scenarios |
| Demand Analysis | Understand historical rental patterns |
| Fleet Planning | Estimate concurrent utilization and fleet scenarios |
| AI Operations Assistant | Ask natural-language operational questions |

The application separates interactive future scenarios using PLAN from historical rolling forecasts using OPS.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn |
| Forecasting | HistGradientBoostingRegressor |
| Statistical Analysis | SciPy, Wilcoxon Signed-Rank Test |
| Application | Streamlit |
| Model Persistence | Joblib |
| Version Control | Git, GitHub |

---

## Project Structure

```text
urbanflow-analytics/
│
├── 01_data_quality_and_features.py
├── 02_hypothesis_test.py
├── 03_fleet_model.py
├── 04_ops_assistant.py
├── 05_model_calibration_experiments.py
│
├── app.py
├── common.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── hour.csv
│   └── day.csv
│
└── outputs/
    ├── reports and metrics
    ├── forecast predictions
    ├── model artifacts
    ├── validation and calibration results
    ├── visualizations
    └── baseline_snapshot/
```

---

## Installation & Execution

### Prerequisites

- Python 3.10 or newer.
- Git.
- VS Code or another Python IDE.

### 1. Clone the repository

```bash
git clone https://github.com/NiharikaDas29/urbanflow-analytics.git
cd urbanflow-analytics
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The repository includes completed model artifacts and evaluation outputs.

**Important:** The supplied completed project has already performed its final held-out test evaluation once. Do not rerun `03_fleet_model.py`, remove its evaluation guard, or tune the model against the held-out test period.

For details about safe reproduction and development-only experiments, see the relevant instructions in the project scripts and audit reports.

---

## Limitations & Assumptions

This project is based on completed rental transactions and does not contain:
- Station-level bike inventory.
- Failed rental attempts.
- Actual fleet size.
- Measured trip durations.
- Real-time bike availability.

Consequently:
- Fleet utilization is estimated using assumed trip durations.
- Fleet buffers are planning assumptions.
- Forecast intervals are not guarantees of station-level service coverage.
- Historical peak demand does not prove that shortages occurred.
- Aggregate forecasts cannot identify where bikes should be redistributed.

A randomized zone-level field experiment using failed-rental and station-stock telemetry is proposed in `outputs/experiment_design.md` to investigate these questions.

Additional technical audit and evaluation details are available in:
- `outputs/audit_report.md`
- `outputs/final_submission_audit.md`
- `outputs/final_evaluation_report.md`
- `outputs/experiment_design.md`

---

## Future Improvements

Potential extensions include:
- Incorporating real-time station inventory and availability.
- Integrating weather forecasts instead of realized weather observations.
- Modelling station-level demand and spatial imbalance.
- Evaluating demand-responsive redistribution strategies.
- Integrating failed-rental and stockout telemetry.
- Conducting controlled field experiments to validate operational improvements.

---

## Dataset & References

**Dataset:** [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

The dataset contains hourly and daily rental counts along with calendar and weather attributes.

**Core methodology:** Statistical hypothesis testing, chronological validation, quantile forecasting, residual calibration, and Little's Law-based utilization estimation.

---

## Author

**Niharika Das**  
Integrated M.Sc. Mathematics | NIT Rourkela

[GitHub Profile](https://github.com/NiharikaDas29)

---

*UrbanFlow Analytics — Turning mobility data into actionable operational insights.*
