"""Dual embedding generators for RAG retrieval (Section III-D).

Two embedding types:
1. Attack-Type Semantic Embedding (768-dim): sentence-transformer fine-tuned on
   attack descriptions. Captures semantic category of the threat.
2. Signature Structure Embedding (384-dim): character-level CNN that encodes
   the detection logic (content keywords, pcre, thresholds) of each rule.

Reference: all-mpnet-base-v2 for type embedding; custom Char-CNN for sig embedding.
"""

import numpy as np
from typing import List, Optional, Tuple

from ..utils.suricata_utils import SuricataRule


class TypeEmbedding:
    """Attack-type semantic embedding generator using sentence-transformers.

    Encodes attack type descriptions and rule messages into 768-dim vectors
    using the all-mpnet-base-v2 model (fine-tuned on security text).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def dim(self) -> int:
        return 768

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            if self.device != 'cpu':
                self._model = self._model.to(self.device)

    def encode_rules(self, rules: List[SuricataRule]) -> np.ndarray:
        """Encode a list of Suricata rules into type embeddings.

        Uses the rule's msg field + metadata as the text representation,
        since the msg describes the attack type. For richer encoding,
        concatenates attack type metadata when available.

        Args:
            rules: List of SuricataRule objects.

        Returns:
            (N, 768) embedding matrix.
        """
        self._ensure_model()
        texts = []
        for rule in rules:
            # Build descriptive text from rule msg and metadata
            desc = rule.msg
            for meta in rule.metadata:
                if 'attack_type:' in meta:
                    desc = f"{meta.split(':', 1)[1]}: {rule.msg}"
                    break
            texts.append(desc)

        embeddings = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text string.

        Args:
            text: Attack description or traffic summary text.

        Returns:
            (768,) embedding vector.
        """
        self._ensure_model()
        embedding = self._model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embedding[0].astype(np.float32)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of text strings.

        Args:
            texts: List of text strings.

        Returns:
            (N, 768) embedding matrix.
        """
        self._ensure_model()
        embeddings = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)


class SigEmbedding:
    """Signature structure embedding using a character-level CNN.

    Architecture (as inferred from paper description):
        Character embedding (128-dim)
        → Conv1D(kernel=3, filters=128)
        → Conv1D(kernel=5, filters=128)
        → Conv1D(kernel=7, filters=128)
        → GlobalMaxPool per kernel
        → Concatenate → Dense(384) → L2 normalize

    This encodes the syntactic/structural properties of each rule:
    content keywords, pcre patterns, threshold conditions, protocol,
    and other detection logic elements.

    Args:
        embedding_dim: Output embedding dimension (384).
        char_vocab_size: Number of unique characters to embed.
        device: Device string ('cpu' or 'cuda').
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        char_vocab_size: int = 128,
        device: str = "cpu",
    ):
        self.embedding_dim = embedding_dim
        self.char_vocab_size = char_vocab_size
        self.device = device
        self._model = None
        self._char_to_idx = None

    @property
    def dim(self) -> int:
        return self.embedding_dim

    def _ensure_model(self):
        if self._model is not None:
            return

        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class CharCNN(nn.Module):
            def __init__(self, vocab_size, embed_dim=128, output_dim=384):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self.conv3 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
                self.conv5 = nn.Conv1d(embed_dim, 128, kernel_size=5, padding=2)
                self.conv7 = nn.Conv1d(embed_dim, 128, kernel_size=7, padding=3)
                self.bn = nn.BatchNorm1d(128 * 3)
                self.fc = nn.Linear(128 * 3, output_dim)

            def forward(self, x):
                # x: (B, seq_len) character indices
                x = self.embed(x)                     # (B, seq_len, embed_dim)
                x = x.transpose(1, 2)                 # (B, embed_dim, seq_len)
                c3 = F.relu(self.conv3(x))            # (B, 128, seq_len)
                c5 = F.relu(self.conv5(x))
                c7 = F.relu(self.conv7(x))
                c3 = F.adaptive_max_pool1d(c3, 1).squeeze(-1)  # (B, 128)
                c5 = F.adaptive_max_pool1d(c5, 1).squeeze(-1)
                c7 = F.adaptive_max_pool1d(c7, 1).squeeze(-1)
                x = torch.cat([c3, c5, c7], dim=1)   # (B, 384)
                x = self.bn(x)
                x = self.fc(x)                        # (B, output_dim)
                return F.normalize(x, p=2, dim=1)

        self._model = CharCNN(self.char_vocab_size, 128, self.embedding_dim)
        if self.device != 'cpu':
            self._model = self._model.to(self.device)
        self._model.eval()

        # Build character vocabulary (printable ASCII)
        chars = ['<PAD>', '<UNK>'] + [chr(i) for i in range(32, 127)]
        self._char_to_idx = {c: i for i, c in enumerate(chars[:self.char_vocab_size])}

    def _text_to_tensor(self, texts: List[str], max_len: int = 1024):
        """Convert list of rule strings to character-index tensors."""
        import torch

        batch_size = len(texts)
        indices = torch.zeros(batch_size, max_len, dtype=torch.long)

        for b, text in enumerate(texts):
            for pos, ch in enumerate(text[:max_len]):
                idx = self._char_to_idx.get(ch, 1)  # 1 = <UNK>
                if idx < self.char_vocab_size:
                    indices[b, pos] = idx

        if self.device != 'cpu':
            indices = indices.to(self.device)
        return indices

    def encode_rules(self, rules: List[SuricataRule]) -> np.ndarray:
        """Encode rules into signature structure embeddings.

        Uses the full rule string (serialized) as input to the Char-CNN,
        which captures structural patterns in the rule syntax.

        Args:
            rules: List of SuricataRule objects.

        Returns:
            (N, 384) embedding matrix.
        """
        self._ensure_model()
        texts = [rule.to_string() for rule in rules]
        return self.encode_text_batch(texts)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text into signature embedding.

        Args:
            text: Rule string or traffic feature text.

        Returns:
            (384,) embedding vector.
        """
        return self.encode_text_batch([text])[0]

    def encode_text_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts.

        Args:
            texts: List of text strings.

        Returns:
            (N, 384) embedding matrix.
        """
        self._ensure_model()
        import torch

        batch_size = 32
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            with torch.no_grad():
                tensor = self._text_to_tensor(batch)
                emb = self._model(tensor)
                all_embeddings.append(emb.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0).astype(np.float32)

    def save(self, path: str):
        """Save the Char-CNN model weights."""
        import torch
        self._ensure_model()
        torch.save(self._model.state_dict(), path)

    def load(self, path: str):
        """Load Char-CNN model weights."""
        import torch
        self._ensure_model()
        self._model.load_state_dict(torch.load(path, map_location=self.device))
        self._model.eval()
