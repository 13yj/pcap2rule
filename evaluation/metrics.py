"""Evaluation metrics — SV, DC, VC, FPR, F1 (Section IV-C).

All metrics as defined in the paper:
  SV  = % of rules that pass suricata -T
  DC  = % of attack instances correctly detected
  VC  = % of attack variants detected by generalized rule
  FPR = % of benign flows incorrectly matched
  F1  = 2 * DC * (1-FPR) / (DC + (1-FPR))
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from ..utils.suricata_utils import SuricataRule


def syntactic_validity(rules: List[SuricataRule]) -> float:
    """Compute syntactic validity (SV) — fraction of rules that are syntactically valid.

    Args:
        rules: List of generated SuricataRule objects (None = invalid).

    Returns:
        SV ∈ [0, 100] as percentage.
    """
    if not rules:
        return 0.0
    valid = sum(1 for r in rules if r is not None)
    return (valid / len(rules)) * 100.0


def detection_coverage(
    rules: List[SuricataRule],
    detection_results: List[bool],
) -> float:
    """Compute detection coverage (DC) — % of attack instances detected.

    Args:
        rules: List of generated rules.
        detection_results: Whether each rule successfully detected its attack.

    Returns:
        DC ∈ [0, 100] as percentage.
    """
    if not detection_results:
        return 0.0
    return (sum(detection_results) / len(detection_results)) * 100.0


def variant_coverage(
    generalized_rules: List[SuricataRule],
    variant_results: List[List[bool]],
) -> float:
    """Compute variant coverage (VC) — % of attack variants detected.

    Each generalized rule is tested against m attack variants.
    VC = (1/|R|) * sum_i (1/m * sum_j delta_r(P'_j))

    Args:
        generalized_rules: List of generalized Suricata rules.
        variant_results: List of lists, each inner list has m bools for variant detection.

    Returns:
        VC ∈ [0, 100] as percentage.
    """
    if not variant_results:
        return 0.0
    total_variants = 0
    detected_variants = 0
    for results in variant_results:
        total_variants += len(results)
        detected_variants += sum(results)
    if total_variants == 0:
        return 0.0
    return (detected_variants / total_variants) * 100.0


def false_positive_rate(
    rules: List[SuricataRule],
    benign_matches: List[int],
    benign_total: int,
) -> float:
    """Compute false positive rate (FPR) — % of benign flows matched.

    FPR(r) = |{P ∈ B : delta_r(P) = 1}| / |B|

    Args:
        rules: List of rules.
        benign_matches: Number of benign flows each rule matched.
        benign_total: Total number of benign flows per rule.

    Returns:
        FPR ∈ [0, 100] as percentage.
    """
    if not benign_matches or benign_total == 0:
        return 0.0
    return (np.mean(benign_matches) / benign_total) * 100.0


def f1_score(dc: float = None, fpr: float = None,
             precision: float = None, recall: float = None) -> float:
    """Compute F1-score.

    Can be called as:
      f1_score(dc=87, fpr=3.7)  — from DC and FPR (both in [0, 100])
      f1_score(precision=0.9, recall=0.85)  — from precision and recall (both in [0, 1])

    Returns:
        F1 ∈ [0, 100] if using DC/FPR, or ∈ [0, 1] if using precision/recall.
    """
    if precision is not None and recall is not None:
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    dc = dc or 0
    fpr = fpr or 0
    dc_norm = dc / 100.0
    fpr_norm = fpr / 100.0
    prec = 1.0 - fpr_norm
    if dc_norm + prec == 0:
        return 0.0
    f1 = 2 * dc_norm * prec / (dc_norm + prec)
    return f1 * 100.0


def compute_all_metrics(
    rules: List[SuricataRule],
    generalized_rules: List[SuricataRule],
    detection_results: List[bool],
    variant_results: List[List[bool]],
    benign_matches: List[int],
    benign_total: int,
) -> Dict[str, float]:
    """Compute all five primary metrics.

    Returns:
        Dict with keys: SV, DC, VC, FPR, F1.
    """
    sv = syntactic_validity(rules)
    dc = detection_coverage(rules, detection_results)
    vc = variant_coverage(generalized_rules, variant_results)
    fpr = false_positive_rate(rules, benign_matches, benign_total)
    f1 = f1_score(dc, fpr)
    return {
        'SV': sv,
        'DC': dc,
        'VC': vc,
        'FPR': fpr,
        'F1': f1,
    }


def per_attack_dc(
    rules: List[SuricataRule],
    detection_results: List[bool],
    attack_types: List[str],
) -> Dict[str, float]:
    """Compute detection coverage per attack type (Table V).

    Args:
        rules: List of generated rules.
        detection_results: Per-sample detection outcomes.
        attack_types: Attack type label for each sample (same length).

    Returns:
        Dict mapping attack_type → DC percentage.
    """
    by_type = defaultdict(list)
    for rule, detected, atype in zip(rules, detection_results, attack_types):
        by_type[atype].append(detected)

    return {
        atype: (sum(results) / len(results)) * 100.0
        for atype, results in by_type.items()
    }


def metrics_mean_std(metrics_list: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    """Aggregate metrics across multiple seeds → (mean, std).

    Args:
        metrics_list: List of metric dicts from different seeds.

    Returns:
        Dict mapping metric_name -> (mean, std) tuple.
    """
    keys = metrics_list[0].keys()
    result = {}
    for key in keys:
        values = [m[key] for m in metrics_list]
        result[key] = (np.mean(values), np.std(values))
    return result
