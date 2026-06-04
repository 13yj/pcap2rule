"""Stage 2: Bi-Encoder Semantic Scorer (Section III-F.2, Eq.3-5).

Dual-tower architecture with shared weights, trained via contrastive learning.
Computes cosine similarity between rule embeddings and traffic feature embeddings.

The Bi-Encoder is fine-tuned with contrastive learning on (rule, traffic_features)
pairs. For each positive pair, N_neg = 8 negative rules are sampled.
"""

import numpy as np
from typing import Tuple, Optional, List

from ..utils.suricata_utils import SuricataRule
from ..utils.logging import get_logger


class SemanticScorer:
    """Bi-Encoder semantic alignment scorer.

    Encodes rules and traffic features into a shared 768-dim space and
    computes cosine similarity as the semantic alignment score.

    In production, this loads a fine-tuned sentence-transformer model.
    For initialization/testing, falls back to a rule-based heuristic.

    Args:
        model_name: Sentence-transformer model to use.
        embedding_dim: Output embedding dimension (768).
        threshold: Minimum score for acceptance (θ_s = 0.6).
        device: Device for model inference.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        embedding_dim: int = 768,
        threshold: float = 0.6,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.threshold = threshold
        self.device = device
        self._model = None
        self.logger = get_logger("pcap2rule.semantic")

    def _ensure_model(self):
        """Lazy-load the sentence-transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except ImportError:
                self.logger.warning(
                    "sentence-transformers not available. Using heuristic scorer."
                )
                self._model = None

    def score(
        self,
        rule: SuricataRule,
        flow_features: np.ndarray,
        payload_text: str,
    ) -> Tuple[float, str]:
        """Compute semantic alignment score between rule and traffic features.

        Args:
            rule: Generated SuricataRule.
            flow_features: (4, 38) flow statistics.
            payload_text: Extracted payload text.

        Returns:
            (score, feedback) tuple. Score ∈ [-1, 1]. Feedback is empty if
            score >= threshold, otherwise contains suggestions.
        """
        if rule is None:
            return 0.0, "[SEMANTIC ERROR] Null rule — cannot compute semantic score."

        if self._model is not None:
            score = self._compute_embedding_score(rule, flow_features, payload_text)
        else:
            score = self._compute_heuristic_score(rule, flow_features, payload_text)

        if score >= self.threshold:
            return score, ""
        else:
            feedback = self._build_feedback(rule, score, flow_features, payload_text)
            return score, feedback

    def _compute_embedding_score(
        self,
        rule: SuricataRule,
        flow_features: np.ndarray,
        payload_text: str,
    ) -> float:
        """Compute score using real sentence-transformer embeddings (Eq.3-4).

        h_r = Enc_rule(r)
        h_f = Enc_feat(Phi(P))
        s(r, P) = cos(h_r, h_f)
        """
        self._ensure_model()
        if self._model is None:
            return self._compute_heuristic_score(rule, flow_features, payload_text)

        # Encode rule
        rule_text = f"{rule.msg}. {rule.to_string()}"
        rule_emb = self._model.encode([rule_text], convert_to_numpy=True)[0]

        # Encode traffic features
        feat_text = self._features_to_text(flow_features, payload_text)
        feat_emb = self._model.encode([feat_text], convert_to_numpy=True)[0]

        # Cosine similarity (Eq.4)
        score = np.dot(rule_emb, feat_emb) / (
            np.linalg.norm(rule_emb) * np.linalg.norm(feat_emb) + 1e-8
        )
        return float(score)

    def _compute_heuristic_score(
        self,
        rule: SuricataRule,
        flow_features: np.ndarray,
        payload_text: str,
    ) -> float:
        """Heuristic semantic scorer as fallback when no model is available.

        Scores based on:
        - Content keyword overlap with payload (40%)
        - Protocol match (20%)
        - Structural completeness (20%)
        - Flow direction/condition appropriateness (20%)
        """
        scores = []

        # 1. Content overlap with payload
        payload_lower = payload_text.lower()
        content_hits = 0
        for content in rule.content:
            if content.lower() in payload_lower:
                content_hits += 1
        content_score = min(content_hits / max(len(rule.content), 1), 1.0) if rule.content else 0.5
        scores.append(content_score * 0.4)

        # 2. Protocol match
        mean = flow_features[0]
        proto_score = 0.5
        if len(mean) > 31 and mean[31] > 0.5:  # HTTP indicator
            proto_score = 1.0 if rule.protocol == 'http' else 0.3
        elif len(mean) > 32 and mean[32] > 0.5:  # DNS indicator
            proto_score = 1.0 if rule.protocol == 'dns' else 0.3
        scores.append(proto_score * 0.2)

        # 3. Structural completeness
        struct_score = 0.0
        if rule.msg:
            struct_score += 0.25
        if rule.flow:
            struct_score += 0.2
        if rule.content:
            struct_score += 0.2
        if rule.pcre:
            struct_score += 0.15
        if rule.classtype and rule.classtype != 'trojan-activity':
            struct_score += 0.1
        if rule.sid >= 100000:
            struct_score += 0.1
        scores.append(struct_score * 0.2)

        # 4. Flow direction appropriateness
        direction_score = 0.5
        if rule.flow:
            if 'to_server' in rule.flow:
                direction_score = 0.8
            elif 'to_client' in rule.flow:
                direction_score = 0.6
        scores.append(direction_score * 0.2)

        return sum(scores)

    def _features_to_text(
        self,
        flow_features: np.ndarray,
        payload_text: str,
    ) -> str:
        """Convert features to a text representation for the Bi-Encoder."""
        mean = flow_features[0]
        parts = [payload_text[:512]]

        if len(mean) > 0:
            parts.append(f"duration: {mean[0]:.1f}ms")
        if len(mean) > 10:
            parts.append(f"packets: {mean[10]:.0f}|{mean[11]:.0f}")
        if len(mean) > 12:
            parts.append(f"bytes: {mean[12]:.0f}|{mean[13]:.0f}")

        return " ".join(parts)

    def _build_feedback(
        self,
        rule: SuricataRule,
        score: float,
        flow_features: np.ndarray,
        payload_text: str,
    ) -> str:
        """Build structured feedback for rules below the semantic threshold."""
        lines = [
            f"[SEMANTIC WARNING] Alignment score {score:.3f} < threshold {self.threshold}:",
        ]

        # Content keyword analysis
        payload_lower = payload_text.lower()
        missing_content = []
        for content in rule.content:
            if content.lower() not in payload_lower:
                missing_content.append(content)
        if missing_content:
            lines.append(
                f"  Content keywords not found in traffic: {missing_content}"
            )
            lines.append(
                "  Suggestion: Verify that content matches correspond to "
                "actual byte sequences in the observed traffic."
            )

        if not rule.content and not rule.pcre:
            lines.append(
                "  Rule has no content or pcre conditions — consider adding "
                "specific byte patterns from the payload."
            )

        if score < 0.3:
            lines.append(
                "  Suggestion: The rule appears semantically unrelated to the "
                "traffic. Consider regenerating with more exemplar rules."
            )

        return "\n".join(lines)
