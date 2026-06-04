"""Ablation study orchestrator (Table III).

Runs Pcap2Rule under 8 configurations, each removing one component:
  full, -flow_only, -payload_only, -rag, -validation,
  -generalization, -lora, -cot

For each configuration, computes SV, DC, VC, FPR with mean ± std
across 5 random seeds.
"""

from typing import Dict, List, Any, Optional, Callable
import numpy as np
from copy import deepcopy

from ..utils.seed import get_seeds, set_seed
from ..utils.logging import get_logger


ABLATION_CONFIGS = {
    "full": {
        "dual_channel": True, "rag": True, "validation": True,
        "generalization": True, "lora": True, "cot": True,
    },
    "-flow_only": {
        "dual_channel": False, "rag": True, "validation": True,
        "generalization": True, "lora": True, "cot": True,
        "input_mode": "flow",
    },
    "-payload_only": {
        "dual_channel": False, "rag": True, "validation": True,
        "generalization": True, "lora": True, "cot": True,
        "input_mode": "payload",
    },
    "-rag": {
        "dual_channel": True, "rag": False, "validation": True,
        "generalization": True, "lora": True, "cot": True,
    },
    "-validation": {
        "dual_channel": True, "rag": True, "validation": False,
        "generalization": True, "lora": True, "cot": True,
    },
    "-generalization": {
        "dual_channel": True, "rag": True, "validation": True,
        "generalization": False, "lora": True, "cot": True,
    },
    "-lora": {
        "dual_channel": True, "rag": True, "validation": True,
        "generalization": True, "lora": False, "cot": True,
    },
    "-cot": {
        "dual_channel": True, "rag": True, "validation": True,
        "generalization": True, "lora": True, "cot": False,
    },
}


def run_ablation_experiment(
    config_name: str,
    config: Dict[str, Any],
    build_agent_fn: Callable,
    evaluate_fn: Callable,
    seeds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Run one ablation configuration across multiple seeds.

    Args:
        config_name: Name of the configuration (e.g., "-rag").
        config: Ablation parameter dict.
        build_agent_fn: Function that takes config and returns a Pcap2RuleAgent.
        evaluate_fn: Function that takes an agent and returns metrics dict.
        seeds: List of random seeds (default: {42, 123, 456, 789, 1024}).

    Returns:
        Dict with 'name', 'metrics_mean', 'metrics_std', 'per_seed' keys.
    """
    seeds = seeds or get_seeds()
    logger = get_logger("pcap2rule.ablation")
    logger.info(f"Running ablation: {config_name}")

    all_metrics = []
    for seed in seeds:
        set_seed(seed)
        agent = build_agent_fn(config, seed)
        metrics = evaluate_fn(agent, seed)
        all_metrics.append(metrics)

    # Compute mean and std
    keys = all_metrics[0].keys()
    means, stds = {}, {}
    for key in keys:
        vals = [m[key] for m in all_metrics]
        means[key] = np.mean(vals)
        stds[key] = np.std(vals)

    return {
        'name': config_name,
        'metrics_mean': means,
        'metrics_std': stds,
        'per_seed': all_metrics,
    }


def run_all_ablations(
    build_agent_fn: Callable,
    evaluate_fn: Callable,
    configs: Optional[Dict[str, Dict]] = None,
    seeds: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Run all ablation configurations and return sorted results.

    Returns:
        List of result dicts (sorted by DC, descending).
    """
    configs = configs or ABLATION_CONFIGS
    results = []

    for config_name, config in configs.items():
        result = run_ablation_experiment(
            config_name, config, build_agent_fn, evaluate_fn, seeds
        )
        results.append(result)

    # Sort by DC (descending) — full should be first
    results.sort(
        key=lambda r: r['metrics_mean'].get('DC', 0),
        reverse=True,
    )
    return results


def print_ablation_table(results: List[Dict[str, Any]]):
    """Print ablation results in paper Table III format."""
    header = f"{'Configuration':<30} {'SV (%)':>10} {'DC (%)':>10} {'VC (%)':>10} {'FPR (%)':>10}"
    sep = "-" * 80
    print(sep)
    print(header)
    print(sep)

    for r in results:
        m = r['metrics_mean']
        s = r['metrics_std']
        print(
            f"{r['name']:<30} "
            f"{m['SV']:>5.1f}±{s['SV']:.1f}  "
            f"{m['DC']:>5.1f}±{s['DC']:.1f}  "
            f"{m['VC']:>5.1f}±{s['VC']:.1f}  "
            f"{m['FPR']:>5.1f}±{s['FPR']:.1f}"
        )
    print(sep)
