# Bike Sharing Demand Prediction & Fleet Planning

An interview-ready, reproducible case study using the UCI Bike Sharing hourly
and daily CSVs ([UCI dataset 275](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)).
The project examines demand patterns, estimates average
concurrent bike use, evaluates demand forecasts, and presents fleet scenarios
without claiming to observe station availability.

## What this project can and cannot answer

The source data contains completed rental counts and calendar/weather
attributes: hourly `hour.csv` and daily `day.csv` records for 2011–2012, with
calendar, weather, casual/registered, and total rental variables. It does **not** contain station-level inventory, failed rental
attempts, actual fleet size, or trip durations. Therefore:

- Peak/off-peak and weather results are descriptive historical evidence.
- “Bikes in use” is an estimate based on Little’s Law and assumed trip lengths.
- A P95 fleet figure is a conditional scenario, not an observed bike requirement.
- The project cannot prove whether complaints were caused by insufficient
  capacity or poor distribution. A randomized zone-level field experiment with
  failed-rental and station-stock telemetry is proposed in
  `outputs/experiment_design.md`.

The project audit is documented in `outputs/audit_report.md`; the final
submission status and verification evidence are in
`outputs/final_submission_audit.md`.

## Project layout

```text
.
├── common.py
├── 01_data_quality_and_features.py
├── 02_hypothesis_test.py
├── 03_fleet_model.py
├── 04_ops_assistant.py
├── 05_model_calibration_experiments.py
├── app.py
├── requirements.txt
├── data/
│   ├── hour.csv
│   └── day.csv
└── outputs/                   # reports, figures, metrics, and fitted model
```

## Setup (Windows PowerShell)

Use Python 3.10 or newer. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents virtual-environment activation, run the interpreter
directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Place the supplied `hour.csv` and `day.csv` in the `data/` directory. The code
uses paths relative to the project files and does not require an API key or
paid service.

## Reproduce the pipeline

The commands below describe a **fresh, un-evaluated working copy**. In the
supplied completed project, the final test evaluation has already run once and
the guard is set to count 1. Do not rerun `03_fleet_model.py` in this folder.
Reproduce data quality and hypothesis artifacts safely with the first two
commands; run final model evaluation only in a separate clean copy after
freezing all decisions.

Run commands in order from the project root:

```powershell
python .\01_data_quality_and_features.py
python .\02_hypothesis_test.py
# Development-only comparisons; review and freeze choices before the final test run.
python .\05_model_calibration_experiments.py
# Run only after model and calibration choices are approved.
python .\03_fleet_model.py
python .\04_ops_assistant.py --demo
streamlit run .\app.py
```

The Streamlit app is also usable after running the data and model stages; the
dashboard and analysis pages show an explanatory message if model artifacts
have not yet been created.

## Pipeline details

### 1. Data quality and exploratory analysis

`01_data_quality_and_features.py` checks nulls, duplicate rows and timestamps,
types/ranges, date/calendar agreement, rental-count consistency, day/hour
totals, source weather/season codes, and extreme values within hour/day-type
cells. Outliers are flagged for review, never removed automatically.
`outputs/fig_01_outliers.png` visualizes the hourly rental distributions and
retains their high-demand values.

An observed row with `cnt == 0` remains a genuine recorded zero. A missing
timestamp is unknown demand and is **not** filled with zero. Missing context
used to render a complete hourly timeline is marked as imputed; such rows are
not used for training or evaluation. October 29–30, 2012 are excluded as
configured event dates, not treated as zero-rental dates. Consecutive gaps of
six or more hours are flagged as unconfirmed outage candidates; that heuristic
does not establish an outage. Season values are retained as source codes; no
new season labels are inferred.

The UCI weather variables are transformed to approximate physical units for
interpretability (`temp × 41`, `atemp × 50`, `hum × 100`, and `windspeed × 67`).
Season codes follow the source definitions (1 spring, 2 summer, 3 fall,
4 winter); source weather codes 1–4 are retained with their UCI category
definitions. Engineered feature descriptions and predictor leakage boundaries
are in `outputs/data_dictionary.json`; row-level values are in
`outputs/features.csv`.

### 2. Historical hypothesis test

The paired historical comparison asks whether the typical daily working-day
peak-minus-off-peak rental difference has a positive location shift. Peak
hours are 08:00, 17:00, and 18:00; off-peak hours are 10:00–15:00 and
21:00–23:00. Only days with every specified hour observed are included.
The one-sided Wilcoxon signed-rank test assumes approximately symmetric paired
differences and tests a location shift; it is **not** a direct test of
arithmetic means. The report includes the paired sample size, statistic,
p-value, median difference and ratio, weekly block-bootstrap 95% intervals,
and descriptive symmetry diagnostics. The bootstrap effect intervals do not
replace the test's independence/symmetry assumptions.

In the current run, 495 complete working days yielded a Wilcoxon statistic of
122,738 and one-sided p-value 4.736e-83. The median paired difference was
338.67 rentals/hour (95% weekly block-bootstrap CI 315.11–360.67); the median
peak/off-peak ratio was 3.15 (95% CI 3.12–3.18). This is historical evidence
about completed rentals, not stockouts, shortages, or a causal rebalancing
effect. The exact results and limitations are in
`outputs/hypothesis_results.json` and `outputs/experiment_design.md`.

Statistical significance is not treated as operational or causal significance:
rentals do not show whether a customer failed to find a bike. The field
experiment design and planning sample-size scenarios are saved separately.
Scenario baseline failure rates and clustering inputs are explicitly marked
as assumptions.

### 3. Forecasting and fleet scenarios

The target is estimated average concurrent bikes in use:

```text
(registered rentals/hour × registered trip minutes
 + casual rentals/hour × casual trip minutes) / 60
```

Default trip durations are **12 minutes registered** and **30 minutes
casual**. The optional **15% fleet buffer** is a planning assumption, not a
measured maintenance or redistribution rate. The model script also reports
sensitivity to 8/20, 12/30, 15/40, and 20/50-minute trip-duration pairs.

Data are not randomly shuffled:

- Point-model comparison fit: before April 1, 2012; validation: April–May 2012.
- Fit: before June 1, 2012.
- Calibration: June–August 2012.
- Held-out test: September–December 2012.

Only observed, non-excluded target rows are used. PLAN remains a
calendar/weather-only model for arbitrary future scenarios; it does not require
demand lags. OPS is a separate rolling day-ahead use case. For a target hour
starting at timestamp `t`, its latest historical target is at `t-24h`; because
the dataset timestamps label the start of each hour, that input hour must have
completed before the forecast is issued (about 23 hours before `t`). OPS is not
intended for a multi-month forecast horizon.

OPS features include the same-hour one-day, one-week, and two-week lags; the
existing `roll24` is the mean of target hours `t-47h` through `t-24h`. The
seven-day mean uses `t-191h` through `t-24h`, requiring at least 120 observed
hours; the seven-day trend compares that window with the preceding seven-day
window (`t-359h` through `t-192h`). Missing timestamps and configured event
dates do not contribute target values to these histories. Hour/month cyclic
interactions and temperature/weather-category interactions are OPS-only, so
PLAN and its future-scenario interface remain unchanged.

The final model script's point-model comparison fits before April 1, 2012 and
selects using April–May validation. The frozen OPS quantile model is native
quantile `HistGradientBoostingRegressor` with the enhanced 29-feature OPS set
and P50/P90/P95 outputs. Its peak-aware calibration estimates finite-sample
residual offsets on June–August 2012: working-day 07:00–09:00 and 16:00–19:00
use local offsets when the calibration subset has at least 100 rows; otherwise
they fall back to the global offset. All other hours use the global offset.
Offsets use the one-based rank `min(ceil((n + 1) × q), n)`, followed by a
monotone projection to prevent quantile crossing.

September–December 2012 is reserved for one final evaluation after the
estimator, features, and calibration are frozen. The final run writes
`outputs/frozen_model_config.json`, a single-run guard
`outputs/final_evaluation_run.json`, subgroup metrics, a comparison against the
preserved baseline, and `outputs/final_evaluation_report.md`. The guard prevents
accidentally rerunning the final test through the same pipeline; do not remove
it to tune the model against the test period.

The reported OPS test metrics use realized target-hour weather covariates
available in the historical dataset. They are conditional on that realized
weather and do **not** measure a fully operational forecast using weather
forecasts available at issue time. Weather forecast error and availability
would need a separate evaluation.

#### Frozen OPS held-out results

The one recorded test run covers September–December 2012 (2,730 observations).
Overall metrics are MAE 12.59, RMSE 19.60, WAPE 0.2078, and R² 0.880.
P50/P90/P95 coverage is 56.4% / 89.3% / 93.5%, respectively. Coverage is
below nominal for P90 and P95; in the evening peak, P95 coverage is 89.3%.
These finite-sample results are not a guarantee of future or station-level
coverage. Segment-level metrics, pinball loss, and preserved-baseline
comparison are in `outputs/final_evaluation_report.md`.

#### Key business insights

- For complete working days, median peak-hour rentals were about 3.15 times
  selected off-peak rentals. This identifies a temporal concentration of
  completed demand, not evidence that riders were unable to rent a bike.
- OPS errors and quantile coverage vary by period. The evening peak has higher
  point error and lower P95 coverage than the nominal 95%; an overall score
  should not obscure that segment risk.
- A system-level fleet estimate cannot locate bikes or determine whether the
  issue is aggregate capacity versus distribution. Instrument failed attempts
  and station stock, then pilot redistribution where measured imbalance exists
  before considering fleet expansion.

The final evaluation guard in the supplied completed project records one run.
Do not delete or bypass it. A fresh end-to-end model evaluation is not part of
ordinary dashboard startup or data/hypothesis reproduction.

#### Development-only model and calibration experiments

Run `05_model_calibration_experiments.py` before making final choices. It reads
only `features.csv` rows before June 1, 2012 and uses expanding chronological
train/calibration/validation folds; it does not load or evaluate the final
September–December test period. It compares the preserved 17-feature OPS set
with the 29-feature OPS set using the same complete rows, RandomForest,
HistGradientBoosting, ExtraTrees, native-quantile HistGradientBoosting, and the
existing seasonal-naive baselines. XGBoost and LightGBM are not added as
dependencies.

The script compares global finite-sample residual offsets with an experimental
peak-aware option. Morning peak is working-day hours 07:00–09:00 and evening
peak is working-day hours 16:00–19:00; local offsets require at least 100
calibration observations per peak, otherwise the global offset is used. It
reports metrics and underprediction by time-of-day, working/non-working day,
and source season, along with quantile crossings before monotone projection.
These validation results guide a recommendation, not a station-level coverage
guarantee. The frozen candidate is the enhanced OPS quantile model with
peak-aware calibration, selected because the case prioritizes rush-hour
underprediction; fold instability and remaining evening undercoverage are
explicitly reported rather than treated as solved.

The saved results include MAE, RMSE, WAPE, R², quantile coverage, pinball loss,
validation-only permutation importance, and forecast plots. Calibration is
marginal and does not guarantee station-level service coverage.

### 4. Operations assistant

The assistant uses deterministic rules to parse questions about dates,
weekdays/weekends, hours, months, weather codes, and demand growth/reduction.
All numeric answers come from `outputs/plan_model.joblib`; no LLM or external
API is used. Weather conditions use month/weather median covariates computed
from the pre-test training period through May 2012.
Every response shows its assumptions. Unsupported or invalid requests return
an explanation rather than an invented result.

Examples:

```powershell
python .\04_ops_assistant.py --demo
python .\04_ops_assistant.py --fleet 300
```

The optional `--fleet` argument is a user-provided current fleet size. The
assistant compares it to a modelled aggregate scenario but does not produce a
definitive buy/don’t-buy verdict.

### 5. Streamlit application

The app has an executive dashboard, demand forecasting, demand analysis,
fleet planning, and a rule-based operations assistant. Forecast controls cover
scenario date/month, day type, hourly range, weather, and demand growth. Fleet
recommendations are conditional and prominently disclose that station-level
data are unavailable. Interactive scenarios use the separate PLAN
calendar/weather-only model because they do not have historical lag inputs; the
dashboard identifies the frozen OPS model and displays its final overall and
time-of-day test metrics.

## Generated outputs

All pipeline reports, figures, model artifacts, and assistant demo output are
written under `outputs/`, including:

- `audit_report.md`, `final_submission_audit.md`, `quality_report.json`,
  `features.csv`
- `data_dictionary.json`
- `fig_01_eda.png`, `fig_01_missing_hours.png`
- `fig_01_outliers.png`
- `hypothesis_results.json`, `experiment_design.md`, `fig_02_hypothesis.png`
- `model_selection_metrics.csv`, `calibration_metrics.csv`, `model_metrics.csv`, `fleet_results.json`
- `fleet_profile_by_hour.csv`, `forecast_test_predictions.csv`
- `evaluation_setup.json`, `validation_feature_importance.csv`
- `development_experiment_config.json`, `development_validation_metrics.csv`
- `development_estimator_comparison.csv`, `development_feature_set_comparison.csv`
- `development_calibration_comparison.csv`, `development_calibration_offsets.csv`
- `development_validation_predictions.csv`
- `frozen_model_config.json`, `final_evaluation_run.json`
- `calibration_offsets.csv`, `ops_test_metrics_by_period.csv`, `baseline_comparison.csv`
- `final_evaluation_report.md`
- `fig_03_model.png`, `plan_model.joblib`, `assistant_demo.md`
- `fig_04_validation_model_comparison.png`, `fig_05_validation_calibration.png`
- `fig_06_validation_hourly_bias.png`
- `baseline_snapshot/` preserves the pre-improvement feature table, metrics,
  predictions, model artifact, and their SHA-256 manifest for comparison.

Generated artifacts are reproducible from the two input CSVs. No model result
or accuracy metric is hardcoded in the application.
