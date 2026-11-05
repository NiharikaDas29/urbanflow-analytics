# Project audit report

## Scope and assignment brief

The original assignment PDF or an equivalent brief was not present in the
repository at the time of this review. The audit therefore uses the two known
requirements supplied for this project:

1. **Data Quality Checks & Feature Engineering**
2. **Formulate a Testable Hypothesis**

No additional assignment criteria are inferred.

## Data quality and feature engineering

The hourly source contains 17,379 rows and 18 columns; the daily source
contains 731 rows and 16 columns. The quality report records source shapes,
per-column missingness and dtypes, numeric distribution summaries, duplicate
checks, range and count consistency checks, timestamp alignment, and source
season/weather code frequencies. The processed feature table is a complete
hourly calendar grid; it distinguishes observed rows from missing timestamps.

Observed zero-rental rows are retained. Missing timestamps remain unknown and
are not imputed as zero demand. Candidate gaps are labeled as potential outage
intervals, not confirmed outages or stockouts. Outliers are flagged and
retained. Feature definitions, PLAN/OPS feature lists, target derivation
assumptions, and leakage boundaries are recorded in
`data_dictionary.json`; row-level engineered values are in `features.csv`.

The estimated concurrent-use target uses Little's Law and assumed 12-minute
registered and 30-minute casual trip durations. The optional 15% fleet buffer
is also an assumption. Neither actual inventory nor failed rental attempts are
present in the source.

## Historical hypothesis

The analysis pairs complete working days and compares the mean recorded
rentals within selected peak and off-peak hours. The one-sided paired
Wilcoxon signed-rank procedure is interpreted as a location-shift test under
an approximate symmetry assumption, not as a test of arithmetic means. The
report now aligns the research question and hypotheses with that test, reports
paired sample size, statistic, p-value, median effects and weekly
block-bootstrap intervals, and includes descriptive paired-difference
symmetry diagnostics.

The recorded association does not establish a shortage or causal effect of
rebalancing. The proposed field experiment requires failed-rental and
station-stock instrumentation.

## Predictive modeling and operational limits

PLAN supports calendar/weather scenarios without demand lags. OPS is a
separate rolling day-ahead forecast with target history cut off at t−24h,
chronological fit/calibration/test periods, quantile prediction, and
finite-sample calibration. OPS evaluation uses realized target-hour weather
covariates, so its held-out metrics are conditional on realized weather; they
do not establish performance using operational weather forecasts.

The frozen final evaluation was completed once. Final test results cannot
establish actual station availability, failed rentals, purchase needs, or
redistribution effectiveness. Review the final report and disclosed
limitations before interpreting the forecasts as operational capacity
decisions.

## Current evidence

- `quality_report.json`: source profiles, data-quality checks, and missing-hour policy.
- `data_dictionary.json`: variable/feature definitions, assumptions, and leakage notes.
- `features.csv`: processed hourly data and engineered features.
- `hypothesis_results.json`, `experiment_design.md`, and `fig_02_hypothesis.png`:
  historical test evidence and interpretation.
- `final_evaluation_report.md`, `frozen_model_config.json`, and
  `final_evaluation_run.json`: frozen OPS model and one-time evaluation evidence.
- `baseline_snapshot/manifest.json`: checksums for the preserved baseline evidence.
