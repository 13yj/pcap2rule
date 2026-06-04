"""Ablation study runner (Table III).

Runs all 8 ablation configurations across 5 seeds each:
  full, -flow_only, -payload_only, -rag, -validation,
  -generalization, -lora, -cot

Usage:
    python scripts/run_ablation.py --config configs/experiment/ablation.yaml
    python scripts/run_ablation.py --dataset CSE-CIC-IDS2018 --output-dir ./results
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Any
from datetime import datetime

from pcap2rule.utils.config import load_config
from pcap2rule.utils.logging import get_logger, setup_logging
from pcap2rule.utils.seed import set_seed, get_seeds

from pcap2rule.evaluation.ablation import (
    ABLATION_CONFIGS,
    run_ablation_experiment,
    run_all_ablations,
    print_ablation_table,
)


def run_ablation_study(
    config_path: Optional[str] = None,
    dataset_name: str = 'CSE-CIC-IDS2018',
    configs: Optional[Dict[str, Dict]] = None,
    seeds: Optional[List[int]] = None,
    output_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run the complete ablation study.

    Args:
        config_path: Path to ablation YAML config.
        dataset_name: Dataset to evaluate on.
        configs: Override ABLATION_CONFIGS (default from ablation.py).
        seeds: Random seeds (default: 5 seeds).
        output_dir: Output directory for results.

    Returns:
        List of ablation result dicts (sorted by DC descending).
    """
    config = load_config(config_path) if config_path else load_config()
    setup_logging(config.logging.dir if hasattr(config, 'logging') else './logs')
    logger = get_logger("pcap2rule.ablation_study")
    seeds = seeds or get_seeds()

    output_dir = output_dir or config.experiment.output_dir or './output'
    os.makedirs(output_dir, exist_ok=True)

    configs = configs or ABLATION_CONFIGS

    logger.info(f"Ablation study: {len(configs)} configurations × {len(seeds)} seeds")
    logger.info(f"Configurations: {list(configs.keys())}")

    # FIXME: Provide actual build_agent_fn and evaluate_fn
    # These require model loading and full pipeline components
    def build_agent_fn(ablation_config: dict, seed: int):
        """Build agent with ablation config applied."""
        set_seed(seed)
        # TODO: Initialize agent with specific components disabled
        raise NotImplementedError(
            "Provide build_agent_fn that creates Pcap2RuleAgent with ablation config"
        )

    def evaluate_fn(agent, seed: int):
        """Evaluate agent and return metrics dict."""
        # TODO: Run agent on test set and compute metrics
        raise NotImplementedError(
            "Provide evaluate_fn that runs agent and returns {SV, DC, VC, FPR}"
        )

    results = run_all_ablations(build_agent_fn, evaluate_fn, configs, seeds)

    # Print table
    print_ablation_table(results)

    # Save results
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'dataset': dataset_name,
        'n_configs': len(configs),
        'n_seeds': len(seeds),
        'results': [
            {
                'name': r['name'],
                'metrics_mean': {k: float(v) for k, v in r['metrics_mean'].items()},
                'metrics_std': {k: float(v) for k, v in r['metrics_std'].items()},
            }
            for r in results
        ],
    }
    output_path = os.path.join(output_dir, 'ablation_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Ablation results saved to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description='Pcap2Rule Ablation Study')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='CSE-CIC-IDS2018')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--configs', type=str, nargs='*',
                        help='Specific ablation configs to run (default: all 8)')
    args = parser.parse_args()

    configs = None
    if args.configs:
        configs = {k: ABLATION_CONFIGS[k] for k in args.configs if k in ABLATION_CONFIGS}

    run_ablation_study(
        config_path=args.config,
        dataset_name=args.dataset,
        configs=configs,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
