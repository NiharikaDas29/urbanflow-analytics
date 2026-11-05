# Frozen OPS model: final evaluation

**Final test window:** 2012-09-01 through 2012-12-31 (chronological held-out rows).

## Frozen model and calibration

- **Estimator:** HistGradientBoostingRegressor (native quantile); quantiles P50, P90, and P95.
- **Features:** enhanced OPS set (29 leakage-safe features); feature names are recorded in `frozen_model_config.json`.
- **Calibration:** peak_aware; finite-sample residual offsets from June–August 2012. Morning (working-day 07:00–09:00) and evening (working-day 16:00–19:00) offsets are local when each calibration subset has at least 100 observations; otherwise the global offset is used. Off-peak rows use the global offset.
- **Leakage guard:** the model fit ends before June 1, 2012; calibration is June–August; the September–December test was evaluated once and was not used for feature, estimator, or calibration selection. In the rolling OPS test, each target hour may use observed history through t−24h, including earlier test-period hours, as available at forecast issue time.
- **Weather limitation:** OPS test features use realized target-hour weather covariates from the historical dataset. The reported performance is conditional on realized weather and does not establish performance using weather forecasts available at operational forecast issue time.

## Overall OPS test performance

| N | MAE | RMSE | WAPE | R² | P50 coverage | P50 pinball | P90 coverage | P90 pinball | P95 coverage | P95 pinball |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2730 | 12.59 | 19.60 | 0.2078 | 0.880 | 56.4% | 6.29 | 89.3% | 3.38 | 93.5% | 2.18 |

## Performance by operating period

| Segment | N | MAE | RMSE | WAPE | R² | P50 coverage | P50 pinball | P90 coverage | P90 pinball | P95 coverage | P95 pinball |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 2730 | 12.59 | 19.60 | 0.2078 | 0.880 | 56.4% | 6.29 | 89.3% | 3.38 | 93.5% | 2.18 |
| morning_peak | 222 | 16.88 | 22.94 | 0.1751 | 0.700 | 49.5% | 8.44 | 84.2% | 3.76 | 92.3% | 2.19 |
| evening_peak | 299 | 19.82 | 25.21 | 0.1615 | 0.767 | 61.5% | 9.91 | 83.9% | 4.00 | 89.3% | 2.31 |
| off_peak | 2209 | 11.18 | 18.33 | 0.2302 | 0.872 | 56.5% | 5.59 | 90.5% | 3.26 | 94.2% | 2.16 |

## Comparison with preserved baseline

Both models are scored on the same timestamps available in the preserved baseline predictions. The baseline uses its preserved P50/P90/P95 outputs; the frozen model uses the corresponding final predictions.

| Segment | Model | N | MAE | RMSE | WAPE | R² | P50 coverage | P50 pinball | P90 coverage | P90 pinball | P95 coverage | P95 pinball |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | Frozen OPS: enhanced features + peak-aware calibration | 2730 | 12.59 | 19.60 | 0.2078 | 0.880 | 56.4% | 6.29 | 89.3% | 3.38 | 93.5% | 2.18 |
| morning_peak | Frozen OPS: enhanced features + peak-aware calibration | 222 | 16.88 | 22.94 | 0.1751 | 0.700 | 49.5% | 8.44 | 84.2% | 3.76 | 92.3% | 2.19 |
| evening_peak | Frozen OPS: enhanced features + peak-aware calibration | 299 | 19.82 | 25.21 | 0.1615 | 0.767 | 61.5% | 9.91 | 83.9% | 4.00 | 89.3% | 2.31 |
| off_peak | Frozen OPS: enhanced features + peak-aware calibration | 2209 | 11.18 | 18.33 | 0.2302 | 0.872 | 56.5% | 5.59 | 90.5% | 3.26 | 94.2% | 2.16 |
| overall | Preserved baseline OPS | 2730 | 13.06 | 20.13 | 0.2156 | 0.873 | 53.5% | 6.53 | 85.9% | 3.59 | 93.4% | 2.45 |
| morning_peak | Preserved baseline OPS | 222 | 18.14 | 24.81 | 0.1882 | 0.649 | 27.0% | 9.07 | 78.8% | 4.63 | 89.6% | 2.58 |
| evening_peak | Preserved baseline OPS | 299 | 21.04 | 27.25 | 0.1714 | 0.727 | 29.1% | 10.52 | 66.2% | 5.40 | 82.3% | 3.35 |
| off_peak | Preserved baseline OPS | 2209 | 11.47 | 18.40 | 0.2363 | 0.871 | 59.4% | 5.74 | 89.3% | 3.24 | 95.2% | 2.32 |

## Limitations and business recommendations

- The target is estimated average concurrent use from Little’s Law and assumed registered/casual trip durations (12/30 minutes), not observed station inventory.
- Peak-aware calibration is marginal, based on a finite calibration window, and does not guarantee conditional or future-period coverage. Review the evening P90/P95 results before using these intervals for service commitments.
- The UCI data lack station-level stock, failed rental attempts, actual fleet size, and trip durations. These results cannot establish whether complaints require purchasing bikes or redistribution.
- Pair the forecasts with station inventory and failed-rental telemetry; pilot targeted redistribution before making capital decisions.
- No model or calibration changes were made in response to final-test results.
