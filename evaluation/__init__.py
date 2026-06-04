"""Evaluation module — metrics, statistical analysis, robustness, and ablation."""

from .metrics import (
    syntactic_validity,
    detection_coverage,
    variant_coverage,
    false_positive_rate,
    f1_score,
    compute_all_metrics,
    per_attack_dc,
    metrics_mean_std,
)
from .statistical import (
    bootstrap_confidence_interval,
    mcnemar_test,
    run_all_comparisons,
    compute_mean_std_across_seeds,
)
from .robustness import (
    perturb_payload_obfuscation,
    perturb_timing,
    perturb_fragmentation,
    perturb_protocol_mimicry,
    apply_all_perturbations,
    evaluate_robustness,
)
from .ablation import (
    ABLATION_CONFIGS,
    run_ablation_experiment,
    run_all_ablations,
    print_ablation_table,
)
from .human_eval import (
    compute_fleiss_kappa,
    compute_approval_rates,
    split_by_rule_type,
)

__all__ = [
    'syntactic_validity',
    'detection_coverage',
    'variant_coverage',
    'false_positive_rate',
    'f1_score',
    'compute_all_metrics',
    'per_attack_dc',
    'metrics_mean_std',
    'bootstrap_confidence_interval',
    'mcnemar_test',
    'run_all_comparisons',
    'compute_mean_std_across_seeds',
    'perturb_payload_obfuscation',
    'perturb_timing',
    'perturb_fragmentation',
    'perturb_protocol_mimicry',
    'apply_all_perturbations',
    'evaluate_robustness',
    'ABLATION_CONFIGS',
    'run_ablation_experiment',
    'run_all_ablations',
    'print_ablation_table',
    'compute_fleiss_kappa',
    'compute_approval_rates',
    'split_by_rule_type',
]
