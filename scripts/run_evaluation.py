"""Evaluation runner — computes all paper metrics across baselines.

Usage:
    python scripts/run_evaluation.py --config configs/experiment/main_eval.yaml
    python scripts/run_evaluation.py --dataset CSE-CIC-IDS2018 --baselines all
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

from pcap2rule.utils.config import load_config
from pcap2rule.utils.logging import get_logger, setup_logging
from pcap2rule.utils.seed import set_seed, get_seeds

from pcap2rule.evaluation import (
    syntactic_validity,
    detection_coverage,
    variant_coverage,
    false_positive_rate,
    f1_score,
    compute_all_metrics,
    per_attack_dc,
    metrics_mean_std,
    bootstrap_confidence_interval,
    mcnemar_test,
    run_all_comparisons,
    compute_mean_std_across_seeds,
)


def run_single_seed_evaluation(
    dataset,
    agent,
    seed: int,
    config,
) -> Dict[str, Any]:
    """Run evaluation for one random seed."""
    set_seed(seed)

    samples = dataset.sample_stratified(
        n_per_attack=config.data.get('n_attack_samples', 50),
        n_benign=config.data.get('n_benign_samples', 50),
    )

    predicted_rules = []
    ground_truth_rules = []
    generalized_rules = []
    detection_results = []
    attack_types = []

    for sample in samples:
        try:
            result = agent.generate_from_sample(sample)
            predicted_rules.append(str(result.get('specific_rule', '')))
            ground_truth_rules.append(sample.ground_truth_rule or '')
            generalized_rules.append(str(result.get('generalized_rule', '')))
            detection_results.append(result.get('syntax_valid', False))
            attack_types.append(sample.attack_type)
        except Exception:
            predicted_rules.append('')
            ground_truth_rules.append(sample.ground_truth_rule or '')
            generalized_rules.append('')
            detection_results.append(False)
            attack_types.append(sample.attack_type)

    metrics = compute_all_metrics(
        predicted_rules=predicted_rules,
        ground_truth_rules=ground_truth_rules,
        generalized_rules=generalized_rules,
        detection_results=detection_results,
        benign_fp=[],
    )
    metrics['per_attack_dc'] = per_attack_dc(
        attack_types, predicted_rules, ground_truth_rules
    )
    return metrics


def run_full_evaluation(
    config_path: Optional[str] = None,
    dataset_name: str = 'CSE-CIC-IDS2018',
    baselines: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the complete evaluation protocol from the paper.

    Computes:
      - Table I/II: SV, DC, VC, FPR, F1 (mean ± std across 5 seeds)
      - Table V: Per-attack-type DC
      - Bootstrap 95% CI for all metrics
      - McNemar's test against all baselines

    Args:
        config_path: Path to experiment YAML config.
        dataset_name: 'CSE-CIC-IDS2018' or 'CIC-IDS2017'.
        baselines: List of baseline method names to compare against.
        seeds: Random seeds (default: {42, 123, 456, 789, 1024}).
        output_dir: Output directory for results.

    Returns:
        Dict with all evaluation results.
    """
    config = load_config(config_path) if config_path else load_config()
    setup_logging(config.logging.dir if hasattr(config, 'logging') else './logs')
    logger = get_logger("pcap2rule.evaluation")
    seeds = seeds or get_seeds()

    output_dir = output_dir or config.experiment.output_dir or './output'
    os.makedirs(output_dir, exist_ok=True)

    # FIXME: Replace with actual agent initialization
    agent = None  # Requires model loading

    logger.info(f"Running {len(seeds)}-seed evaluation on {dataset_name}")

    metrics_per_seed = []
    for seed in seeds:
        logger.info(f"Seed {seed}...")
        # metrics = run_single_seed_evaluation(dataset, agent, seed, config)
        # metrics_per_seed.append(metrics)

    if not metrics_per_seed:
        logger.warning("No metrics computed. Initialize agent and dataset properly.")
        return {'error': 'Pipeline not fully initialized. Provide agent and dataset.'}

    # Compute mean ± std
    mean_std = compute_mean_std_across_seeds(metrics_per_seed)

    # Bootstrap CIs
    ci_results = {}
    for metric in ['SV', 'DC', 'VC', 'FPR', 'F1']:
        if metric in mean_std:
            vals = [m.get(metric, 0) for m in metrics_per_seed]
            mean, (lo, hi) = bootstrap_confidence_interval(vals)
            ci_results[metric] = {'mean': mean, 'ci_lower': lo, 'ci_upper': hi}

    # McNemar tests vs baselines
    mcnemar_results = {}
    if baselines:
        # Requires per-sample detection outcomes
        pass

    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset': dataset_name,
        'n_seeds': len(seeds),
        'metrics_mean_std': {k: {'mean': float(v[0]), 'std': float(v[1])}
                            for k, v in mean_std.items()},
        'bootstrap_ci': ci_results,
        'mcnemar': mcnemar_results,
        'per_seed': metrics_per_seed,
    }

    output_path = os.path.join(output_dir, f'eval_results_{dataset_name}.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    # Print summary table
    logger.info("=" * 70)
    logger.info(f"Evaluation Results — {dataset_name}")
    logger.info("=" * 70)
    logger.info(f"{'Metric':<15} {'Mean ± Std':>20} {'95% CI':>25}")
    logger.info("-" * 70)
    for metric in ['SV', 'DC', 'VC', 'FPR', 'F1']:
        if metric in mean_std:
            m, s = mean_std[metric]
            ci = ci_results.get(metric, {})
            lo, hi = ci.get('ci_lower', 0), ci.get('ci_upper', 0)
            logger.info(f"{metric:<15} {m:>6.1f} ± {s:>5.1f}     [{lo:.1f}, {hi:.1f}]")
    logger.info("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description='Pcap2Rule Evaluation Runner')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='CSE-CIC-IDS2018',
                        choices=['CSE-CIC-IDS2018', 'CIC-IDS2017'])
    parser.add_argument('--baselines', type=str, nargs='*',
                        default=['ET-Open', 'Snort3', 'GPT-4o'])
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--seeds', type=int, nargs='*',
                        default=[42, 123, 456, 789, 1024])
    args = parser.parse_args()

    run_full_evaluation(
        config_path=args.config,
        dataset_name=args.dataset,
        baselines=args.baselines,
        seeds=args.seeds,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
