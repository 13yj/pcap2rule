"""Statistical analysis — Bootstrap confidence intervals and McNemar's test.

Section IV-I: Bootstrap resampling (10,000 samples) for 95% CI.
McNemar's test on paired binary outcomes (correct detection vs. miss).
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy import stats


def bootstrap_confidence_interval(
    values: List[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, Tuple[float, float]]:
    """Compute bootstrap confidence interval for a metric.

    Args:
        values: List of per-sample metric values.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (default 0.95 for 95% CI).
        seed: Random seed.

    Returns:
        (mean, (lower_bound, upper_bound)) tuple.
    """
    rng = np.random.RandomState(seed)
    arr = np.array(values)
    n = len(arr)
    mean = arr.mean()

    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        means.append(sample.mean())

    alpha = (1.0 - confidence) / 2.0
    lower = np.percentile(means, alpha * 100)
    upper = np.percentile(means, (1 - alpha) * 100)

    return mean, (lower, upper)


def mcnemar_test(
    method_a_correct: List[bool],
    method_b_correct: List[bool],
) -> Tuple[float, float]:
    """Perform McNemar's test on paired binary outcomes.

    Compares two methods' detection results on the SAME samples.
    Null hypothesis: both methods have the same error rate.

    Args:
        method_a_correct: Whether method A correctly detected each sample.
        method_b_correct: Whether method B correctly detected each sample.

    Returns:
        (chi_squared_statistic, p_value) tuple.
    """
    # Contingency table:
    #           B correct  B wrong
    # A correct    n_11      n_10
    # A wrong      n_01      n_00

    n_10 = sum(1 for a, b in zip(method_a_correct, method_b_correct) if a and not b)
    n_01 = sum(1 for a, b in zip(method_a_correct, method_b_correct) if not a and b)

    # McNemar's test statistic: (b - c)^2 / (b + c)
    if n_10 + n_01 == 0:
        return (0.0, 1.0)

    chi2 = (n_10 - n_01) ** 2 / (n_10 + n_01)
    p_value = 1.0 - stats.chi2.cdf(chi2, df=1)

    return (chi2, p_value)


def run_all_comparisons(
    pcap2rule_results: List[bool],
    baseline_results: dict,
) -> dict:
    """Run McNemar's test comparing Pcap2Rule against all baselines.

    Args:
        pcap2rule_results: Per-sample detection outcomes for Pcap2Rule.
        baseline_results: Dict mapping baseline name → per-sample outcomes.

    Returns:
        Dict mapping baseline name → (chi2, p-value) tuple.
    """
    results = {}
    for name, baseline in baseline_results.items():
        n = min(len(pcap2rule_results), len(baseline))
        chi2, p = mcnemar_test(
            pcap2rule_results[:n],
            baseline[:n],
        )
        results[name] = (chi2, p)
    return results


def compute_mean_std_across_seeds(
    metrics_per_seed: List[dict],
) -> dict:
    """Compute mean and standard deviation of each metric across seeds.

    Args:
        metrics_per_seed: List of {metric_name: value} dicts, one per seed.

    Returns:
        Dict mapping metric_name → (mean, std) tuple.
    """
    if not metrics_per_seed:
        return {}

    keys = metrics_per_seed[0].keys()
    result = {}
    for key in keys:
        values = [m[key] for m in metrics_per_seed if key in m]
        result[key] = (np.mean(values), np.std(values))
    return result
