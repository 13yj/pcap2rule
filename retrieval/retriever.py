"""RAG Retriever — Top-k exemplar retrieval with weighted similarity.

Implements the core retrieval logic from Section III-D, Eq.2:
  sim(q, r_i) = α·cos(e_type_q, e_type_i) + (1-α)·cos(e_sig_q, e_sig_i)

where e_type_q is the type-weighted query embedding and e_sig_q is
extracted from the input traffic features.
"""

import numpy as np
from typing import List, Optional, Tuple

from .knowledge_base import KnowledgeBase
from .embeddings import TypeEmbedding, SigEmbedding
from .faiss_index import FAISSIndex
from .attack_classifier import AttackTypeClassifier
from ..utils.suricata_utils import SuricataRule
from ..pcap_processor.protocol_parser import normalize_payload


class RAGRetriever:
    """RAG retrieval module with dual-embedding weighted similarity.

    Args:
        kb: KnowledgeBase instance with loaded rules.
        type_embedding: TypeEmbedding encoder.
        sig_embedding: SigEmbedding encoder.
        faiss_index: FAISSIndex for ANN search.
        classifier: AttackTypeClassifier for type prediction.
        alpha: Weight between type and signature similarity (default 0.6).
        k: Number of exemplars to retrieve.
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        type_embedding: TypeEmbedding,
        sig_embedding: SigEmbedding,
        faiss_index: FAISSIndex,
        classifier: AttackTypeClassifier,
        alpha: float = 0.6,
        k: int = 3,
    ):
        self.kb = kb
        self.type_embedding = type_embedding
        self.sig_embedding = sig_embedding
        self.faiss_index = faiss_index
        self.classifier = classifier
        self.alpha = alpha
        self.k = k

    def build_index(self):
        """Compute embeddings for all KB rules and build FAISS indices."""
        if len(self.kb) == 0:
            raise RuntimeError("Knowledge base is empty. Load rules first.")

        rules = self.kb.rules
        type_emb = self.type_embedding.encode_rules(rules)
        sig_emb = self.sig_embedding.encode_rules(rules)

        self.kb.set_embeddings(type_emb, sig_emb)
        self.faiss_index.build(type_emb, sig_emb)

    def retrieve(
        self,
        flow_features: np.ndarray,
        payload_text: str,
    ) -> List[SuricataRule]:
        """Retrieve top-k exemplar rules.

        Args:
            flow_features: (4, 38) aggregated flow statistics.
            payload_text: Extracted payload text.

        Returns:
            List of top-k SuricataRule exemplars.
        """
        if not self.kb.is_indexed:
            return []

        # Step 1: Predict attack type distribution (Eq.2)
        type_probs = self.classifier.predict_single_proba(flow_features)

        # Step 2: Build type-weighted query embedding
        type_query = self._build_type_query(type_probs)

        # Step 3: Build signature query embedding from traffic features
        sig_query = self._build_sig_query(flow_features, payload_text)

        # Step 4: Search both indices
        type_dists, type_ids = self.faiss_index.search_type(type_query, self.k * 2)
        sig_dists, sig_ids = self.faiss_index.search_sig(sig_query, self.k * 2)

        # Step 5: Weighted similarity fusion (Eq.2)
        scores = self._fuse_scores(type_dists[0], type_ids[0],
                                   sig_dists[0], sig_ids[0])

        # Step 6: Select top-k
        top_k = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:self.k]
        exemplars = [self.kb.rules[idx] for idx, _ in top_k]

        return exemplars

    def _build_type_query(self, type_probs: np.ndarray) -> np.ndarray:
        """Build type-weighted query embedding.

        e_type_q = Σ p_j · e_type_centroid,j

        Uses the probability distribution over attack types to compute
        a weighted combination of type centroids.

        Args:
            type_probs: (K,) probability vector from classifier.

        Returns:
            (1, 768) query embedding.
        """
        attack_types = self.classifier.attack_types
        if not attack_types:
            return np.zeros((1, self.type_embedding.dim), dtype=np.float32)

        # For each attack type, create a text description and encode
        centroid_texts = [f"A {at} attack in network traffic" for at in attack_types]
        centroids = self.type_embedding.encode_batch(centroid_texts)  # (K, 768)

        if type_probs.ndim == 1:
            type_probs = type_probs.reshape(1, -1)

        query = type_probs @ centroids  # (1, 768)
        return query.astype(np.float32)

    def _build_sig_query(
        self,
        flow_features: np.ndarray,
        payload_text: str,
    ) -> np.ndarray:
        """Build signature structure query embedding from traffic features.

        Encodes the flow statistics and payload into a text representation
        that captures structural patterns, then encodes with the Char-CNN.

        Args:
            flow_features: (4, 38) flow stats.
            payload_text: Extracted payload.

        Returns:
            (1, 384) query embedding.
        """
        # Build a structural representation
        text = normalize_payload(payload_text)
        # Add flow structure info
        mean = flow_features[0]
        struct_hints = []
        if len(mean) > 31 and mean[31] > 0.5:  # HTTP indicator
            struct_hints.append("protocol:HTTP")
        if len(mean) > 37 and mean[37] == 6:
            struct_hints.append("transport:TCP")
        else:
            struct_hints.append("transport:UDP")

        combined = text[:256] + " | " + " ".join(struct_hints)
        emb = self.sig_embedding.encode_text(combined)
        return emb.reshape(1, -1).astype(np.float32)

    def _fuse_scores(
        self,
        type_dists: np.ndarray,
        type_ids: np.ndarray,
        sig_dists: np.ndarray,
        sig_ids: np.ndarray,
    ) -> dict:
        """Fuse type and signature similarity scores via weighted sum.

        sim(q, r_i) = α·cos(e_type_q, e_type_i) + (1-α)·cos(e_sig_q, e_sig_i)

        Args:
            type_dists: FAISS distances from type search.
            type_ids: Corresponding rule indices.
            sig_dists: FAISS distances from sig search.
            sig_ids: Corresponding rule indices.

        Returns:
            Dict mapping rule index → fused score.
        """
        scores = {}

        # Type scores (FAISS returns inner product = cosine for normalized vecs)
        for dist, idx in zip(type_dists, type_ids):
            if idx >= 0:
                scores[idx] = scores.get(idx, 0.0) + self.alpha * float(dist)

        # Sig scores
        for dist, idx in zip(sig_dists, sig_ids):
            if idx >= 0:
                scores[idx] = scores.get(idx, 0.0) + (1 - self.alpha) * float(dist)

        return scores

    def retrieve_by_type(
        self,
        attack_type: str,
        k: int = 3,
    ) -> List[SuricataRule]:
        """Retrieve exemplars for a specific attack type (bypasses classifier).

        Args:
            attack_type: Canonical attack type name.
            k: Number of exemplars.

        Returns:
            List of matching rules.
        """
        candidates = self.kb.get_by_attack_type(attack_type)
        if len(candidates) <= k:
            return candidates

        # If we have type embeddings, rank by centroid similarity
        if self.kb.type_embeddings is not None:
            query_text = f"A {attack_type} attack in network traffic"
            query = self.type_embedding.encode_text(query_text)
            query = query / np.linalg.norm(query)

            scores = []
            for rule in candidates:
                idx = self.kb._rule_index.get(rule.sid)
                if idx is not None and idx < len(self.kb.type_embeddings):
                    emb = self.kb.type_embeddings[idx]
                    emb = emb / (np.linalg.norm(emb) + 1e-8)
                    scores.append((rule, float(np.dot(query, emb))))
                else:
                    scores.append((rule, 0.0))

            scores.sort(key=lambda x: x[1], reverse=True)
            return [r for r, _ in scores[:k]]

        return candidates[:k]
