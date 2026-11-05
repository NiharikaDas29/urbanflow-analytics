"""Paired historical peak/off-peak analysis and a proposed causal field experiment."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common import OUT_DIR

PEAK = [8, 17, 18]
OFFPEAK = [10, 11, 12, 13, 14, 15, 21, 22, 23]
BOOTSTRAP_REPS = 3000
RANDOM_STATE = 2026


def _block_bootstrap_medians(daily, repetitions=BOOTSTRAP_REPS):
    """Weekly block bootstrap to reduce sensitivity to within-week dependence."""
    data = daily.sort_index()
    blocks = [
        block[["difference", "ratio"]].to_numpy()
        for _, block in data.groupby(data.index.to_period("W-SUN"))
    ]
    rng = np.random.default_rng(RANDOM_STATE)
    difference_medians, ratio_medians = [], []
    for _ in range(repetitions):
        sampled = []
        sampled_rows = 0
        while sampled_rows < len(data):
            block = blocks[int(rng.integers(0, len(blocks)))]
            sampled.append(block)
            sampled_rows += len(block)
        sample = np.concatenate(sampled, axis=0)[:len(data)]
        difference_medians.append(float(np.median(sample[:, 0])))
        ratio_medians.append(float(np.median(sample[:, 1])))
    return {
        "median_difference_ci95": np.percentile(difference_medians, [2.5, 97.5]).tolist(),
        "median_ratio_ci95": np.percentile(ratio_medians, [2.5, 97.5]).tolist(),
    }


def paired_difference_symmetry_diagnostic(differences):
    """Summarize distributional symmetry without treating diagnostics as a formal test."""
    values = np.asarray(differences, dtype=float)
    median = float(np.median(values))
    quantiles = np.quantile(values, [0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95])
    q05, q10, q25, q40, q50, q60, q75, q90, q95 = map(float, quantiles)
    iqr = q75 - q25
    bowley = (q75 + q25 - 2 * q50) / iqr if iqr else None
    skewness = float(stats.skew(values, bias=False)) if len(values) > 2 else None
    likely_departure = (
        (skewness is not None and abs(skewness) >= 1.0)
        or (bowley is not None and abs(bowley) >= 0.25)
    )
    return {
        "n": int(len(values)),
        "mean_difference": float(np.mean(values)),
        "median_difference": median,
        "sample_skewness": skewness,
        "bowley_skewness": float(bowley) if bowley is not None else None,
        "lower_upper_quantile_distances_from_median": {
            "p05_vs_p95": {"lower": median - q05, "upper": q95 - median},
            "p10_vs_p90": {"lower": median - q10, "upper": q90 - median},
            "p25_vs_p75": {"lower": median - q25, "upper": q75 - median},
            "p40_vs_p60": {"lower": median - q40, "upper": q60 - median},
        },
        "summary_flags_potential_asymmetry": bool(likely_departure),
        "interpretation": (
            "Skewness and matched quantile distances indicate potential asymmetry; "
            "the signed-rank p-value should be interpreted cautiously."
            if likely_departure
            else "These selected summaries do not strongly flag asymmetry; they do not prove the distribution is symmetric."
        ),
        "formal_symmetry_test_performed": False,
    }


def part_a(g):
    """Test whether daily working-day peak demand exceeds off-peak demand."""
    valid = g[g["is_observed"] & g["outage_day"].eq(0) & g["workingday"].eq(1)].copy()
    daily = pd.DataFrame({
        "peak": valid[valid["hr"].isin(PEAK)].groupby("dteday")["cnt"].mean(),
        "offpeak": valid[valid["hr"].isin(OFFPEAK)].groupby("dteday")["cnt"].mean(),
        "peak_hours_observed": valid[valid["hr"].isin(PEAK)].groupby("dteday")["cnt"].count(),
        "offpeak_hours_observed": valid[valid["hr"].isin(OFFPEAK)].groupby("dteday")["cnt"].count(),
    })
    daily = daily[
        daily["peak_hours_observed"].eq(len(PEAK))
        & daily["offpeak_hours_observed"].eq(len(OFFPEAK))
    ].dropna()
    if len(daily) < 10:
        raise ValueError("Too few complete working days for a meaningful paired comparison.")
    daily["difference"] = daily["peak"] - daily["offpeak"]
    daily["ratio"] = daily["peak"] / daily["offpeak"].replace(0, np.nan)
    daily = daily.replace([np.inf, -np.inf], np.nan).dropna(subset=["ratio"])
    if daily.empty:
        raise ValueError("No complete days have a nonzero off-peak mean.")

    # Wilcoxon signed-rank tests a positive location shift under a symmetric-difference model.
    test = stats.wilcoxon(
        daily["peak"], daily["offpeak"], alternative="greater",
        zero_method="wilcox", method="auto",
    )
    intervals = _block_bootstrap_medians(daily)
    symmetry = paired_difference_symmetry_diagnostic(daily["difference"])
    daily["week"] = daily.index.to_period("W-SUN").astype(str)
    yearly = daily.assign(year=daily.index.year).groupby("year")["ratio"].median()
    monthly = daily.assign(month=daily.index.month).groupby("month")["ratio"].median()
    report = {
        "question": (
            "Is the typical daily working-day peak-minus-off-peak rental difference "
            "positive, based on a paired signed-rank location test?"
        ),
        "H0": (
            "The paired difference distribution has no positive location shift "
            "(signed-rank pseudomedian/location parameter <= 0), under approximate symmetry."
        ),
        "H1": (
            "The paired difference distribution has a positive location shift "
            "(signed-rank pseudomedian/location parameter > 0), under approximate symmetry."
        ),
        "test": (
            "One-sided paired Wilcoxon signed-rank test on complete working days; "
            "tests a location shift under the symmetry assumption, not the arithmetic mean."
        ),
        "effect_measure": (
            "Median of paired daily differences and median daily peak/off-peak ratio; "
            "these are descriptive effect measures, distinct from the signed-rank statistic."
        ),
        "test_statistic": float(test.statistic),
        "p_value": float(test.pvalue),
        "alpha": 0.05,
        "statistically_significant_at_alpha": bool(test.pvalue < 0.05),
        "assumption_checks": {
            "unit_of_pairing": "calendar day",
            "required_observed_peak_hours": PEAK,
            "required_observed_offpeak_hours": OFFPEAK,
            "complete_days_only": True,
            "paired_observations": int(len(daily)),
            "signed_rank_symmetry_assumption": (
                "Wilcoxon signed-rank location inference assumes the paired differences "
                "are approximately symmetric around a common location. Distribution "
                "skewness, Bowley skewness, and matched quantile distances are reported "
                "as diagnostics, not as a formal symmetry test."
            ),
            "paired_difference_symmetry_diagnostic": symmetry,
            "serial_dependence": (
                "Daily observations may be serially dependent; confidence intervals use "
                "resampled week blocks, while the Wilcoxon p-value assumes independent pairs."
            ),
        },
        "peak_hours": PEAK,
        "offpeak_hours": OFFPEAK,
        "median_peak_rentals_per_hour": float(daily["peak"].median()),
        "median_offpeak_rentals_per_hour": float(daily["offpeak"].median()),
        "median_paired_difference_rentals_per_hour": float(daily["difference"].median()),
        "median_peak_to_offpeak_ratio": float(daily["ratio"].median()),
        "median_difference_block_bootstrap_ci95": [
            float(value) for value in intervals["median_difference_ci95"]
        ],
        "median_ratio_block_bootstrap_ci95": [
            float(value) for value in intervals["median_ratio_ci95"]
        ],
        "practical_significance_interpretation": (
            "Use the estimated absolute difference and relative ratio with their intervals "
            "to judge operational relevance; statistical significance alone does not establish "
            "a shortage or justify fleet purchase."
        ),
        "median_ratio_by_year": {str(int(k)): float(v) for k, v in yearly.items()},
        "median_ratio_by_month": {str(int(k)): float(v) for k, v in monthly.items()},
        "share_of_days_peak_at_least_twice_offpeak": float((daily["ratio"] >= 2).mean()),
        "historical_evidence_limit": (
            "Rental counts do not record failed rental attempts, station inventory, or fleet size. "
            "This is descriptive historical evidence, not a causal estimate of rebalancing effects."
        ),
    }
    return report, daily


def n_per_arm(p0, relative_mde, alpha=0.05, power=0.8):
    """Approximate independent-arm sample size for a binary rental-failure rate."""
    if not 0 < p0 < 1 or not 0 < relative_mde < 1:
        raise ValueError("p0 and relative_mde must be between zero and one.")
    p1 = p0 * (1 - relative_mde)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    variance = p0 * (1 - p0) + p1 * (1 - p1)
    return int(np.ceil(((z_alpha + z_power) ** 2 * variance) / (p0 - p1) ** 2))


def part_b():
    """Scenario planning; baseline failed-rental rates and clustering are assumptions."""
    rows = []
    for baseline in (0.03, 0.05, 0.08):
        for relative_reduction in (0.2, 0.3):
            independent_n = n_per_arm(baseline, relative_reduction)
            mean_cluster_size, assumed_icc = 200, 0.05
            design_effect = 1 + (mean_cluster_size - 1) * assumed_icc
            rows.append({
                "assumed_baseline_failed_rental_rate": baseline,
                "relative_reduction_to_detect": relative_reduction,
                "independent_attempts_per_arm": independent_n,
                "assumed_cluster_size": mean_cluster_size,
                "assumed_icc": assumed_icc,
                "design_effect": round(design_effect, 2),
                "approx_attempts_per_arm_with_clustering": int(np.ceil(independent_n * design_effect)),
            })
    return pd.DataFrame(rows)


DESIGN = """# Historical hypothesis result and proposed field experiment

## Research question and test

**Question:** {question}

**H0:** {H0}

**H1:** {H1}

**Comparison groups:** selected working-day peak hours {peak_hours}; off-peak
hours {offpeak_hours}. Pairing unit: calendar day; complete pairs only.

**Test:** {test}

- Paired observations: **{n}**
- Wilcoxon statistic: **{statistic:.1f}**
- One-sided p-value: **{p_value:.4g}**, with significance level α = {alpha:.2f}
- Median paired difference: **{difference:.2f} rentals/hour** (weekly block-bootstrap
  95% CI **{difference_low:.2f}–{difference_high:.2f}**)
- Median peak/off-peak ratio: **{ratio:.2f}×** (weekly block-bootstrap
  95% CI **{ratio_low:.2f}–{ratio_high:.2f}**)

The test addresses a signed-rank location shift under approximate symmetry; it
is **not** a direct test of arithmetic means. The median difference and ratio
are descriptive effect measures. Statistical significance does not establish
operational causation or actual bike shortage.

## Paired-difference symmetry diagnostic

- Sample skewness: **{sample_skewness:.3f}**
- Bowley skewness: **{bowley_skewness:.3f}**
- Symmetry flag from selected summaries: **{symmetry_flag}**
- {symmetry_interpretation}

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

{table}

## Decision guardrails

Do not infer that a purchase is warranted from rental peaks alone. Consider a
fleet expansion only if measured unmet demand persists after a feasible
redistribution intervention and the incremental economics support it. If
shortages are not measured, first instrument rental attempts and station stock.
"""


if __name__ == "__main__":
    features_path = os.path.join(OUT_DIR, "features.csv")
    if not os.path.exists(features_path):
        raise FileNotFoundError("Run 01_data_quality_and_features.py before this script.")
    features = pd.read_csv(features_path, parse_dates=["ts", "dteday"])
    results, daily = part_a(features)
    sample_sizes = part_b()
    results["sample_size_scenarios"] = sample_sizes.to_dict("records")
    with open(os.path.join(OUT_DIR, "hypothesis_results.json"), "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    table = sample_sizes.to_string(index=False)
    with open(os.path.join(OUT_DIR, "experiment_design.md"), "w", encoding="utf-8") as file:
        file.write(DESIGN.format(
            question=results["question"],
            H0=results["H0"],
            H1=results["H1"],
            test=results["test"],
            peak_hours=results["peak_hours"],
            offpeak_hours=results["offpeak_hours"],
            n=results["assumption_checks"]["paired_observations"],
            statistic=results["test_statistic"],
            ratio=results["median_peak_to_offpeak_ratio"],
            ratio_low=results["median_ratio_block_bootstrap_ci95"][0],
            ratio_high=results["median_ratio_block_bootstrap_ci95"][1],
            difference=results["median_paired_difference_rentals_per_hour"],
            difference_low=results["median_difference_block_bootstrap_ci95"][0],
            difference_high=results["median_difference_block_bootstrap_ci95"][1],
            p_value=results["p_value"],
            alpha=results["alpha"],
            sample_skewness=results["assumption_checks"][
                "paired_difference_symmetry_diagnostic"
            ]["sample_skewness"],
            bowley_skewness=results["assumption_checks"][
                "paired_difference_symmetry_diagnostic"
            ]["bowley_skewness"],
            symmetry_flag=results["assumption_checks"][
                "paired_difference_symmetry_diagnostic"
            ]["summary_flags_potential_asymmetry"],
            symmetry_interpretation=results["assumption_checks"][
                "paired_difference_symmetry_diagnostic"
            ]["interpretation"],
            table=table,
        ))

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    axes[0].hist(daily["difference"], bins=35, color="#4472C4", alpha=0.85)
    axes[0].axvline(0, color="black", linestyle="--", label="No difference")
    axes[0].axvline(
        results["median_paired_difference_rentals_per_hour"],
        color="#C44E52",
        label="Median paired difference",
    )
    axes[0].set(
        title=f"Paired daily differences (n={len(daily)})",
        xlabel="Peak minus off-peak (rentals/hour)",
        ylabel="Working days",
    )
    axes[0].legend(fontsize=8)

    diagnostic = results["assumption_checks"]["paired_difference_symmetry_diagnostic"]
    distance_map = diagnostic["lower_upper_quantile_distances_from_median"]
    lower_distances = [
        distance_map[key]["lower"] for key in ("p05_vs_p95", "p10_vs_p90", "p25_vs_p75", "p40_vs_p60")
    ]
    upper_distances = [
        distance_map[key]["upper"] for key in ("p05_vs_p95", "p10_vs_p90", "p25_vs_p75", "p40_vs_p60")
    ]
    max_distance = max(lower_distances + upper_distances)
    axes[1].scatter(lower_distances, upper_distances, color="#55A868")
    axes[1].plot([0, max_distance], [0, max_distance], color="black", linestyle="--", label="Symmetry")
    for label, x, y in zip(("5/95", "10/90", "25/75", "40/60"), lower_distances, upper_distances):
        axes[1].annotate(label, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=8)
    axes[1].set(
        title="Matched tail distances around median",
        xlabel="Lower-tail distance",
        ylabel="Upper-tail distance",
    )
    axes[1].legend(fontsize=8)
    axes[1].text(
        0.02,
        0.98,
        f"Skew={diagnostic['sample_skewness']:.2f}; Bowley={diagnostic['bowley_skewness']:.2f}",
        transform=axes[1].transAxes,
        va="top",
        fontsize=8,
    )

    monthly_ratio = daily.assign(month=daily.index.month).groupby("month")["ratio"].median()
    monthly_ratio.plot.bar(ax=axes[2], color="#55A868")
    axes[2].axhline(1, color="black", linestyle="--", label="Equal demand")
    axes[2].set(
        title="Median peak/off-peak ratio by month",
        xlabel="Month",
        ylabel="Median ratio",
    )
    axes[2].legend(fontsize=8)
    fig.suptitle(
        f"Paired Wilcoxon: W={results['test_statistic']:.0f}, "
        f"one-sided p={results['p_value']:.3g}, α={results['alpha']:.2f}",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_02_hypothesis.png"), dpi=160)
    plt.close(fig)
    print(json.dumps(results, indent=2))
    print(sample_sizes.to_string(index=False))
