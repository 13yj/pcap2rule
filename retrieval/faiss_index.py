"""FAISS IVF-PQ index for approximate nearest neighbor search.

Implements the FAISS-based retrieval index described in Section III-D:
  - IVFPQ with 256 centroids (nlist)
  - 64-byte PQ codes (M)
  - Cosine similarity via inner product on L2-normalized vectors
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class FAISSIndex:
    """FAISS IVF-PQ index for dual-embedding similarity search.

    Two separate indices for type embeddings (768-dim) and signature
    embeddings (384-dim).

    Args:
        type_dim: Type embedding dimension (768).
        sig_dim: Signature embedding dimension (384).
        nlist: Number of IVF centroids (256 per paper).
        m: Number of PQ sub-quantizers (64-byte codes → m depends on dim).
        nprobe: Number of clusters to probe during search.
        index_dir: Directory to save/load index files.
    """

    def __init__(
        self,
        type_dim: int = 768,
        sig_dim: int = 384,
        nlist: int = 256,
        m: int = 64,
        nprobe: int = 8,
        index_dir: str = "./knowledge_base/faiss_index",
    ):
        if not FAISS_AVAILABLE:
            raise ImportError(
                "faiss is required for the RAG index. "
                "Install with: pip install faiss-cpu (or faiss-gpu)"
            )
        self.type_dim = type_dim
        self.sig_dim = sig_dim
        self.nlist = nlist
        self.m = m
        self.nprobe = nprobe
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.type_index: Optional[faiss.Index] = None
        self.sig_index: Optional[faiss.Index] = None

    def build(self, type_embeddings: np.ndarray, sig_embeddings: np.ndarray):
        """Build both indices from embedding matrices.

        Args:
            type_embeddings: (N, 768) float32 type embeddings.
            sig_embeddings: (N, 384) float32 signature embeddings.
        """
        self.type_index = self._build_index(type_embeddings, self.type_dim)
        self.sig_index = self._build_index(sig_embeddings, self.sig_dim)

    def _build_index(self, embeddings: np.ndarray, dim: int) -> faiss.Index:
        """Build an IVF-PQ index for the given embeddings.

        Uses inner product (cosine similarity on L2-normalized vectors).
        """
        n = embeddings.shape[0]
        # Ensure float32 contiguous
        x = np.ascontiguousarray(embeddings.astype(np.float32))

        # L2 normalize for cosine via inner product
        faiss.normalize_L2(x)

        # Determine PQ parameters
        # m must divide dim; use 64-byte codes → m = dim / 2 for 2-byte sub-vectors
        effective_m = min(self.m, dim // 2)
        while dim % effective_m != 0:
            effective_m -= 1
        if effective_m < 1:
            effective_m = 1

        nlist = min(self.nlist, n // 5) if n > 0 else self.nlist
        nlist = max(nlist, 1)

        # Quantizer
        quantizer = faiss.IndexFlatIP(dim)

        # IVF-PQ index
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, effective_m, 8)

        # Train
        if n > nlist:
            index.train(x)
        index.add(x)
        index.nprobe = min(self.nprobe, nlist)

        return index

    def search_type(self, query: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Search the type embedding index.

        Args:
            query: (1, 768) or (batch, 768) query vector(s).
            k: Number of results per query.

        Returns:
            (distances, indices) tuple, each of shape (batch, k).
        """
        if self.type_index is None:
            raise RuntimeError("Type index not built. Call build() first.")
        q = np.ascontiguousarray(query.astype(np.float32))
        if q.ndim == 1:
            q = q.reshape(1, -1)
        faiss.normalize_L2(q)
        return self.type_index.search(q, k)

    def search_sig(self, query: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Search the signature embedding index."""
        if self.sig_index is None:
            raise RuntimeError("Sig index not built. Call build() first.")
        q = np.ascontiguousarray(query.astype(np.float32))
        if q.ndim == 1:
            q = q.reshape(1, -1)
        faiss.normalize_L2(q)
        return self.sig_index.search(q, k)

    def save(self):
        """Persist both indices to disk."""
        if self.type_index is not None:
            faiss.write_index(self.type_index, str(self.index_dir / "type_index.faiss"))
        if self.sig_index is not None:
            faiss.write_index(self.sig_index, str(self.index_dir / "sig_index.faiss"))

    def load(self):
        """Load both indices from disk."""
        type_path = self.index_dir / "type_index.faiss"
        sig_path = self.index_dir / "sig_index.faiss"
        if type_path.exists():
            self.type_index = faiss.read_index(str(type_path))
            self.type_index.nprobe = self.nprobe
        if sig_path.exists():
            self.sig_index = faiss.read_index(str(sig_path))
            self.sig_index.nprobe = self.nprobe
