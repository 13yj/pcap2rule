#!/usr/bin/env python
"""Systematic module import verification for Pcap2Rule.

Runs as: python tests/verify_imports.py
"""
import sys, os, traceback

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

MODULES = [
    # Stage 1: Utils
    "pcap2rule.utils.config",
    "pcap2rule.utils.seed",
    "pcap2rule.utils.logging",
    "pcap2rule.utils.suricata_utils",
    "pcap2rule.utils.pcap_utils",

    # Stage 2: Datasets
    "pcap2rule.data_pipeline.dataset_base",
    "pcap2rule.data_pipeline.cse_cic_ids2018",
    "pcap2rule.data_pipeline.cic_ids2017",

    # Stage 3: PCAP Processor
    "pcap2rule.pcap_processor.flow_extractor",
    "pcap2rule.pcap_processor.payload_extractor",
    "pcap2rule.pcap_processor.protocol_parser",
    "pcap2rule.pcap_processor.prompt_builder",

    # Stage 4: RAG Retrieval
    "pcap2rule.retrieval.knowledge_base",
    "pcap2rule.retrieval.embeddings",
    "pcap2rule.retrieval.faiss_index",
    "pcap2rule.retrieval.attack_classifier",
    "pcap2rule.retrieval.retriever",

    # Stage 5: LLM Agent
    "pcap2rule.llm_agent.model",
    "pcap2rule.llm_agent.cot_prompts",
    "pcap2rule.llm_agent.agent",
    "pcap2rule.llm_agent.inference",
    "pcap2rule.llm_agent.lora_trainer",

    # Stage 6: Validators
    "pcap2rule.validators.syntax_validator",
    "pcap2rule.validators.semantic_scorer",
    "pcap2rule.validators.performance_validator",
    "pcap2rule.validators.feedback_formatter",

    # Stage 7: Generalization
    "pcap2rule.generalization.ip_port_abstractor",
    "pcap2rule.generalization.pcre_generalizer",
    "pcap2rule.generalization.sqli_patterns",
    "pcap2rule.generalization.rule_transformer",

    # Stage 8: Evaluation
    "pcap2rule.evaluation.metrics",
    "pcap2rule.evaluation.statistical",
    "pcap2rule.evaluation.robustness",
    "pcap2rule.evaluation.ablation",
    "pcap2rule.evaluation.human_eval",

    # Stage 9: Visualization
    "pcap2rule.visualization.sensitivity_curves",
    "pcap2rule.visualization.tsne_plot",
    "pcap2rule.visualization.attention_heatmap",
    "pcap2rule.visualization.confusion_matrix",

    # Stage 10: Training
    "pcap2rule.training.data_collator",
    "pcap2rule.training.train_lora",
    "pcap2rule.training.train_bi_encoder",
    "pcap2rule.training.train_xgboost",

    # Stage 11: Synthetic data
    "pcap2rule.data_pipeline.synthetic_data",
]

passed = 0
failed = 0
errors = []

for mod in MODULES:
    try:
        __import__(mod)
        print(f"  [OK] {mod}")
        passed += 1
    except Exception as e:
        err_msg = f"  [FAIL] {mod}: {e}"
        print(err_msg)
        tb = traceback.format_exc()
        errors.append((mod, str(e), tb))
        failed += 1

print(f"\n{'='*60}")
print(f"Results: {passed} passed, {failed} failed out of {len(MODULES)} modules")

if errors:
    print(f"\nFailed modules details:")
    for mod, err, _ in errors:
        print(f"  - {mod}: {err.split(chr(10))[0]}")

sys.exit(0 if failed == 0 else 1)
