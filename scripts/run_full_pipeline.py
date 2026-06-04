"""Complete end-to-end pipeline: PCAP -> Suricata Rules.

Orchestrates the full Pcap2Rule inference pipeline (Algorithm 1):
  1. Load PCAP files from dataset
  2. Extract flow features (nfstream, 38-dim)
  3. Extract payload text (scapy, HTTP/DNS/hex)
  4. Predict attack type distribution (XGBoost)
  5. Retrieve top-k exemplar rules (FAISS RAG)
  6. Build cross-modal prompt
  7. Generate rule via 5-step CoT LLM (Qwen2-7B + LoRA)
  8. Closed-loop validation (syntax -> semantic -> FPR)
  9. Generalize rule (IP/port abstraction + PCRE expansion)
  10. Output (specific_rule, generalized_rule)

Usage:
    python scripts/run_full_pipeline.py --config configs/experiment/main_eval.yaml
    python scripts/run_full_pipeline.py --dataset CSE-CIC-IDS2018 --max-samples 10
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from pcap2rule.utils.config import load_config
from pcap2rule.utils.logging import get_logger, setup_logging
from pcap2rule.utils.seed import set_seed
from pcap2rule.utils.suricata_utils import SuricataRule

from pcap2rule.data_pipeline import CSE_CIC_IDS2018, CIC_IDS2017
from pcap2rule.pcap_processor import FlowFeatureExtractor, PayloadExtractor, PromptBuilder
from pcap2rule.retrieval import (
    KnowledgeBase, TypeEmbedding, SigEmbedding,
    FAISSIndex, AttackTypeClassifier, RAGRetriever,
)
from pcap2rule.llm_agent import Pcap2RuleAgent, InferenceEngine
from pcap2rule.validators import (
    SyntaxValidator, SemanticScorer, PerformanceValidator,
    combine_feedback,
)
from pcap2rule.generalization import RuleTransformer
from pcap2rule.evaluation import compute_all_metrics


def build_pipeline(config) -> Dict[str, Any]:
    """Initialize all pipeline components from config.

    Returns:
        Dict of {'component_name': instance} for all pipeline stages.
    """
    logger = get_logger("pcap2rule.pipeline")

    logger.info("Initializing pipeline components...")

    # PCAP processors
    flow_extractor = FlowFeatureExtractor()
    payload_extractor = PayloadExtractor()
    prompt_builder = PromptBuilder()

    # RAG
    kb = KnowledgeBase(config.data.kb_path)
    type_emb = TypeEmbedding()
    sig_emb = SigEmbedding()
    faiss_type = FAISSIndex(dim=768)
    faiss_sig = FAISSIndex(dim=384)
    classifier = AttackTypeClassifier()
    retriever = RAGRetriever(type_emb, sig_emb, faiss_type, faiss_sig, kb, classifier)

    # LLM
    inference_engine = InferenceEngine.from_config(config)
    agent = Pcap2RuleAgent.from_config(config, inference_engine)

    # Validators
    syntax_validator = SyntaxValidator(config.validation.suricata_binary)
    semantic_scorer = SemanticScorer()
    perf_validator = PerformanceValidator(config.validation.benign_set_path)

    # Generalization
    rule_transformer = RuleTransformer()

    components = {
        'flow_extractor': flow_extractor,
        'payload_extractor': payload_extractor,
        'prompt_builder': prompt_builder,
        'retriever': retriever,
        'agent': agent,
        'syntax_validator': syntax_validator,
        'semantic_scorer': semantic_scorer,
        'perf_validator': perf_validator,
        'rule_transformer': rule_transformer,
    }

    logger.info("All components initialized")
    return components


def run_single_sample(
    sample,
    components: Dict[str, Any],
    config,
) -> Dict[str, Any]:
    """Run the full pipeline on a single TrafficSample."""
    logger = get_logger("pcap2rule.pipeline")

    fe = components['flow_extractor']
    pe = components['payload_extractor']
    pb = components['prompt_builder']
    retriever = components['retriever']
    agent = components['agent']

    # Step 1-2: Extract features
    flow_features = fe.extract(sample.pcap_path)
    payload_text = pe.extract(sample.pcap_path)

    # Step 3-5: RAG retrieval
    exemplars = retriever.retrieve(flow_features, payload_text, k=config.rag.k)

    # Step 6: Build prompt
    prompt = pb.build(flow_features, payload_text, exemplars,
                      attack_type=sample.attack_type)

    # Step 7-8: Generate and validate
    result = agent.generate(
        prompt=prompt,
        flow_features=flow_features,
        payload_text=payload_text,
        validator=components['syntax_validator'],
        semantic_scorer=components['semantic_scorer'],
        performance_validator=components['perf_validator'],
    )

    # Step 9: Generalize
    specific_rule = result.get('specific_rule')
    generalized_rule = components['rule_transformer'].transform(specific_rule)

    return {
        'sample_id': sample.sample_id,
        'attack_type': sample.attack_type,
        'specific_rule': str(specific_rule) if specific_rule else None,
        'generalized_rule': str(generalized_rule) if generalized_rule else None,
        'syntax_valid': result.get('syntax_valid', False),
        'semantic_score': result.get('semantic_score', 0.0),
        'fpr': result.get('fpr', 1.0),
        'iterations': result.get('iterations', 0),
        'ground_truth': sample.ground_truth_rule,
    }


def run_full_pipeline(
    config_path: Optional[str] = None,
    dataset_name: str = 'CSE-CIC-IDS2018',
    max_samples: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Main entry point for the full Pcap2Rule pipeline.

    Args:
        config_path: Path to experiment YAML config.
        dataset_name: 'CSE-CIC-IDS2018' or 'CIC-IDS2017'.
        max_samples: Limit number of samples (useful for testing).
        output_dir: Output directory for rules and metrics.

    Returns:
        Dict with aggregated results and per-sample details.
    """
    config = load_config(config_path) if config_path else load_config()
    setup_logging(config.logging.dir if hasattr(config, 'logging') else './logs',
                  level='INFO')
    logger = get_logger("pcap2rule.full_pipeline")
    set_seed(config.experiment.seed)

    output_dir = output_dir or config.experiment.output_dir or './output'
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    dataset_cls = CSE_CIC_IDS2018 if '2018' in dataset_name else CIC_IDS2017
    dataset = dataset_cls(data_dir=config.data.data_dir)
    samples = dataset.sample_stratified(
        n_per_attack=config.data.get('n_attack_samples', 50),
        n_benign=config.data.get('n_benign_samples', 50),
    )
    if max_samples:
        samples = samples[:max_samples]

    logger.info(f"Running pipeline on {len(samples)} samples from {dataset_name}")

    # Build pipeline
    components = build_pipeline(config)

    # Process all samples
    results = []
    for i, sample in enumerate(samples):
        logger.info(f"[{i+1}/{len(samples)}] Processing {sample.sample_id}")
        try:
            result = run_single_sample(sample, components, config)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed on {sample.sample_id}: {e}")
            results.append({
                'sample_id': sample.sample_id,
                'attack_type': sample.attack_type,
                'error': str(e),
            })

    # Compute metrics
    metrics = compute_all_metrics(
        predicted_rules=[r.get('specific_rule', '') for r in results],
        ground_truth_rules=[r.get('ground_truth', '') for r in results],
        generalized_rules=[r.get('generalized_rule', '') for r in results],
        detection_results=[r.get('syntax_valid', False) for r in results],
        benign_fp=[r.get('fpr', 0.0) for r in results],
    )

    # Save results
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'dataset': dataset_name,
        'n_samples': len(results),
        'metrics': {k: float(v) if hasattr(v, '__float__') else v
                    for k, v in metrics.items()},
        'per_sample': results,
    }
    output_path = os.path.join(output_dir, f'pipeline_results_{dataset_name}.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print summary
    logger.info("=" * 60)
    logger.info("Pipeline Complete")
    logger.info("=" * 60)
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {metric}: {value:.2f}")
        else:
            logger.info(f"  {metric}: {value}")
    logger.info(f"Results saved to {output_path}")

    return output_data


def main():
    parser = argparse.ArgumentParser(description='Pcap2Rule Full Pipeline')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to experiment config YAML')
    parser.add_argument('--dataset', type=str, default='CSE-CIC-IDS2018',
                        choices=['CSE-CIC-IDS2018', 'CIC-IDS2017'])
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_full_pipeline(
        config_path=args.config,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
