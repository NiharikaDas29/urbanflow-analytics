# Historical hypothesis result and proposed field experiment

## Research question and test

**Question:** Is the typical daily working-day peak-minus-off-peak rental difference positive, based on a paired signed-rank location test?

**H0:** The paired difference distribution has no positive location shift (signed-rank pseudomedian/location parameter <= 0), under approximate symmetry.

**H1:** The paired difference distribution has a positive location shift (signed-rank pseudomedian/location parameter > 0), under approximate symmetry.

**Comparison groups:** selected working-day peak hours [8, 17, 18]; off-peak
hours [10, 11, 12, 13, 14, 15, 21, 22, 23]. Pairing unit: calendar day; complete pairs only.

**Test:** One-sided paired Wilcoxon signed-rank test on complete working days; tests a location shift under the symmetry assumption, not the arithmetic mean.

- Paired observations: **495**
- Wilcoxon statistic: **122738.0**
- One-sided p-value: **4.736e-83**, with significance level α = 0.05
- Median paired difference: **338.67 rentals/hour** (weekly block-bootstrap
  95% CI **315.11–360.67**)
- Median peak/off-peak ratio: **3.15×** (weekly block-bootstrap
  95% CI **3.12–3.18**)

The test addresses a signed-rank location shift under approximate symmetry; it
is **not** a direct test of arithmetic means. The median difference and ratio
are descriptive effect measures. Statistical significance does not establish
operational causation or actual bike shortage.

## Paired-difference symmetry diagnostic

- Sample skewness: **-0.090**
- Bowley skewness: **-0.039**
- Symmetry flag from selected summaries: **False**
- These selected summaries do not strongly flag asymmetry; they do not prove the distribution is symmetric.

These are descriptive diagnostics, not a formal symmetry test. Daily pairs may
also be serially dependent: the weekly block bootstrap accounts for some
within-week dependence in the effect intervals, but the Wilcoxon p-value
assumes independent pairs. Interpret its p-value cautiously.

## What the historical data can establish

The hourly rental dataset can show whether recorded working-day rentals are
concentrated in the selected peak hours. It does **not** contain failed rental
attempts, station inventory, or current fleet size, so it cannot establish that
customers encountered stock-outs or that rebalancing would solve them.

## Causal question and hypotheses

**H0:** Rebalancing does not reduce peak-hour failed-rental rate versus current
practice. **H1:** Rebalancing reduces peak-hour failed-rental rate.

## Design

- **Unit:** zone (a group of stations), to reduce interference from bikes
  moving between neighboring stations.
- **Assignment:** pair-match zones using pre-period demand and geography, then
  randomize one zone in each pair to treatment.
- **Treatment:** reposition existing bikes before selected AM/PM peaks using
  a prespecified forecast and operational rules. **Control:** current practice.
- **Primary outcome:** failed rental attempts / all rental attempts during
  peak hours; instrumentation must capture unsuccessful attempts.
- **Secondary outcomes:** completed trips, empty-station minutes, full-dock
  events, rebalancing cost, and off-peak idle inventory.
- **Analysis:** intention-to-treat difference between arms, adjusted for
  baseline and matched pairs, with inference clustered at the randomized zone.
  Pre-register one primary outcome and the analysis plan.
- **Duration:** choose from power calculations and operational constraints;
  cover weekdays, weekends, and a range of weather.

## Sample-size scenarios

Rates and intracluster correlation below are planning assumptions, not measured
dataset results. Recalculate with an instrumented pre-period baseline and
zone-level variance before launching.

 assumed_baseline_failed_rental_rate  relative_reduction_to_detect  independent_attempts_per_arm  assumed_cluster_size  assumed_icc  design_effect  approx_attempts_per_arm_with_clustering
                                0.03                           0.2                         11452                   200         0.05          10.95                                   125400
                                0.03                           0.3                          4812                   200         0.05          10.95                                    52692
                                0.05                           0.2                          6743                   200         0.05          10.95                                    73836
                                0.05                           0.3                          2836                   200         0.05          10.95                                    31055
                                0.08                           0.2                          4094                   200         0.05          10.95                                    44830
                                0.08                           0.3                          1724                   200         0.05          10.95                                    18878

## Decision guardrails

Do not infer that a purchase is warranted from rental peaks alone. Consider a
fleet expansion only if measured unmet demand persists after a feasible
redistribution intervention and the incremental economics support it. If
shortages are not measured, first instrument rental attempts and station stock.
