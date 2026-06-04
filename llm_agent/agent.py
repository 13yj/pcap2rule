"""Pcap2Rule LLM Agent — main agentic loop (Algorithm 1).

Orchestrates the complete 5-step CoT generation with closed-loop validation.
The agent generates a specific rule and a generalized variant for each
traffic sample, iteratively refining based on validator feedback.
"""

from typing import Optional, List, Tuple, Dict, Any
import numpy as np

from .model import load_model_vllm, load_model_hf
from .cot_prompts import build_cot_prompt, STEP1_ATTACK_IDENTIFICATION
from .inference import InferenceEngine
from ..pcap_processor.prompt_builder import PromptBuilder
from ..utils.suricata_utils import SuricataRule, parse_suricata_rule
from ..utils.logging import get_logger


class Pcap2RuleAgent:
    """LLM Agent for end-to-end Suricata rule generation (Algorithm 1).

    Integrates: dual-channel prompt construction → 5-step CoT generation →
    closed-loop validation with up to N_max refinement iterations.

    Args:
        prompt_builder: PromptBuilder for structured input construction.
        inference_engine: InferenceEngine for LLM generation.
        validators: Dict of validator components (syntax, semantic, performance).
        generalizer: Rule generalization module.
        max_iterations: Maximum validation-refinement iterations (N_max).
        semantic_threshold: Minimum semantic score for acceptance (θ_s).
        fpr_threshold: Maximum FPR for acceptance.
        use_cot: Whether to use structured 5-step CoT.
        use_rag: Whether to include RAG exemplars.
    """

    def __init__(
        self,
        prompt_builder: PromptBuilder,
        inference_engine: InferenceEngine,
        validators: Optional[Dict[str, Any]] = None,
        generalizer: Any = None,
        max_iterations: int = 3,
        semantic_threshold: float = 0.6,
        fpr_threshold: float = 0.05,
        use_cot: bool = True,
        use_rag: bool = True,
    ):
        self.prompt_builder = prompt_builder
        self.inference = inference_engine
        self.validators = validators or {}
        self.generalizer = generalizer
        self.max_iterations = max_iterations
        self.semantic_threshold = semantic_threshold
        self.fpr_threshold = fpr_threshold
        self.use_cot = use_cot
        self.use_rag = use_rag
        self.logger = get_logger("pcap2rule.agent")

    def generate(
        self,
        flow_features: np.ndarray,
        payload_text: str,
        exemplars: Optional[List[SuricataRule]] = None,
        cti: Any = None,
    ) -> Dict[str, Any]:
        """Generate Suricata rules for a traffic sample (Algorithm 1).

        Args:
            flow_features: (4, 38) aggregated flow statistics.
            payload_text: Extracted payload content text.
            exemplars: RAG-retrieved exemplar rules (or None if RAG disabled).
            cti: StructuredCTI object (Analyst-View 3D CTI representation).

        Returns:
            Dict with keys:
                'specific_rule': SuricataRule for the specific attack instance.
                'generalized_rule': SuricataRule for variant detection.
                'syntax_valid': bool.
                'semantic_score': float.
                'fpr': float.
                'iterations': number of refinement iterations used.
                'feedback_history': list of feedback messages.
        """
        exemplars = exemplars if self.use_rag else None

        # Build initial prompt
        prompt = self.prompt_builder.build(
            flow_features=flow_features,
            payload_text=payload_text,
            exemplars=exemplars,
            cti=cti,
            include_task=True,
        )

        # Step 1-5: Structured CoT Generation
        generated = self.inference.generate(prompt)
        rules = self.inference.extract_rules(generated)

        specific_rule = parse_suricata_rule(rules.get('specific', ''))
        generalized_rule = parse_suricata_rule(rules.get('generalized', ''))

        feedback_history = []
        iterations = 1

        # Closed-loop refinement (Algorithm 1, lines 13-19)
        syntax_ok, sem_ok, fpr_ok = False, False, False
        sem_score = 0.0
        fpr = 1.0

        for i in range(self.max_iterations):
            iterations = i + 1
            feedback_parts = []

            # Stage 1: Syntax validation
            syntax_ok, syn_fb = self._validate_syntax(specific_rule)
            if not syntax_ok:
                feedback_parts.append(syn_fb)

            # Stage 2: Semantic validation
            sem_score, sem_ok, sem_fb = self._validate_semantic(
                specific_rule, flow_features, payload_text
            )
            if not sem_ok:
                feedback_parts.append(sem_fb)

            # Stage 3: Performance validation (FPR)
            fpr, fpr_ok, perf_fb = self._validate_performance(specific_rule)
            if not fpr_ok:
                feedback_parts.append(perf_fb)

            # Check convergence
            if syntax_ok and sem_ok and fpr_ok:
                self.logger.info(f"Rule passed all validators in {iterations} iteration(s)")
                break

            # Apply generalizations
            if self.generalizer and generalized_rule:
                generalized_rule = self.generalizer.transform(
                    specific_rule if specific_rule else generalized_rule
                )

            # Refine: re-generate with feedback
            if i < self.max_iterations - 1 and feedback_parts:
                feedback_text = "\n".join(feedback_parts)
                feedback_history.append(feedback_text)

                refinement_prompt = self._build_refinement_prompt(
                    generated, feedback_text
                )
                generated = self.inference.generate(refinement_prompt)
                rules = self.inference.extract_rules(generated)

                new_specific = parse_suricata_rule(rules.get('specific', ''))
                new_generalized = parse_suricata_rule(rules.get('generalized', ''))

                if new_specific:
                    specific_rule = new_specific
                if new_generalized:
                    generalized_rule = new_generalized

        return {
            'specific_rule': specific_rule,
            'generalized_rule': generalized_rule,
            'syntax_valid': syntax_ok,
            'semantic_score': sem_score,
            'fpr': fpr,
            'iterations': iterations,
            'feedback_history': feedback_history,
            'raw_output': generated,
        }

    def generate_batch(
        self,
        batch: List[Tuple[np.ndarray, str, Optional[List[SuricataRule]]]],
    ) -> List[Dict[str, Any]]:
        """Generate rules for a batch of traffic samples.

        Args:
            batch: List of (flow_features, payload_text, exemplars) tuples.

        Returns:
            List of result dicts (see generate() for format).
        """
        results = []
        for flow_feat, payload, exemplars in batch:
            result = self.generate(flow_feat, payload, exemplars)
            results.append(result)
        return results

    def _validate_syntax(self, rule: SuricataRule) -> Tuple[bool, str]:
        """Stage 1: syntax validation."""
        validator = self.validators.get('syntax')
        if validator is None:
            return True, ""
        return validator.check(rule)

    def _validate_semantic(
        self,
        rule: SuricataRule,
        flow_features: np.ndarray,
        payload_text: str,
    ) -> Tuple[float, bool, str]:
        """Stage 2: semantic alignment validation."""
        validator = self.validators.get('semantic')
        if validator is None:
            return 1.0, True, ""

        score, feedback = validator.score(rule, flow_features, payload_text)
        ok = score >= self.semantic_threshold
        return score, ok, feedback if not ok else ""

    def _validate_performance(self, rule: SuricataRule) -> Tuple[float, bool, str]:
        """Stage 3: false positive rate check."""
        validator = self.validators.get('performance')
        if validator is None:
            return 0.0, True, ""

        fpr, feedback = validator.check(rule)
        ok = fpr <= self.fpr_threshold
        return fpr, ok, feedback if not ok else ""

    def _build_refinement_prompt(
        self,
        original_output: str,
        feedback: str,
    ) -> str:
        """Build a refinement prompt incorporating validation feedback."""
        return f"""[REFINEMENT REQUEST]

Your previous rule generation had the following issues:

{feedback}

Original generation:
{original_output[:1000]}

Please revise the rule to address ALL issues mentioned above. Output the corrected rules:

<specific_rule>
...
</specific_rule>

<generalized_rule>
...
</generalized_rule>"""
