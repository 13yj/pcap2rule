"""Bi-Encoder contrastive training (Section IV-C).

Trains the semantic scorer Bi-Encoder with contrastive loss:
  - Shared-weight dual-tower architecture
  - Temperature-scaled InfoNCE loss (tau=0.07)
  - N_neg=8 hard negatives per anchor
  - Cosine similarity as scoring function
  - Cosine annealing LR, batch size 32, 10 epochs

The trained Bi-Encoder is used in Stage 2 of the closed-loop validation
pipeline to score the semantic alignment between generated rules and
attack traffic patterns.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from ..utils.logging import get_logger
from ..utils.seed import set_seed
from ..validators.semantic_scorer import SemanticScorer


@dataclass
class BiEncoderTrainingArgs:
    """Bi-Encoder training hyperparameters."""
    output_dir: str = './checkpoints/bi_encoder'
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    temperature: float = 0.07
    num_negatives: int = 8
    embedding_dim: int = 768
    warmup_ratio: float = 0.1
    max_seq_length: int = 512
    logging_steps: int = 50
    eval_steps: int = 500
    seed: int = 42


class BiEncoderModel(nn.Module):
    """Shared-weight Bi-Encoder for rule-traffic semantic scoring.

    Architecture:
      Traffic Tower: sentence-transformers → Dense(768) → L2 norm
      Rule Tower:    sentence-transformers → Dense(768) → L2 norm
                     (shared weights with Traffic Tower)
    """

    def __init__(
        self,
        model_name: str = 'all-mpnet-base-v2',
        embedding_dim: int = 768,
        max_seq_length: int = 512,
    ):
        super().__init__()
        self.encoder = SentenceTransformer(model_name)
        self.encoder.max_seq_length = max_seq_length
        self.projection = nn.Linear(
            self.encoder.get_sentence_embedding_dimension(),
            embedding_dim,
        )

    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode texts to normalized embeddings."""
        with torch.no_grad():
            features = self.encoder.encode(
                texts, convert_to_tensor=True, show_progress_bar=False,
            )
        features = self.projection(features)
        return F.normalize(features, p=2, dim=-1)

    def forward(
        self,
        anchors: List[str],
        positives: List[str],
        negatives: List[List[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning embeddings for all triplets.

        Returns:
            (anchor_emb, positive_emb, negative_embs) where
            anchor_emb: (B, D), positive_emb: (B, D), negative_embs: (B, N_neg, D)
        """
        anchor_emb = self.encode(anchors)
        positive_emb = self.encode(positives)
        B, N = len(negatives), len(negatives[0])
        neg_flat = [n for neg_list in negatives for n in neg_list]
        neg_emb_flat = self.encode(neg_flat)
        negative_embs = neg_emb_flat.view(B, N, -1)
        return anchor_emb, positive_emb, negative_embs


def contrastive_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    negative_mask: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE contrastive loss with multiple negatives.

    Args:
        anchor: (B, D) anchor embeddings.
        positive: (B, D) positive embeddings.
        negatives: (B, N, D) negative embeddings.
        negative_mask: (B, N) mask (1 = valid negative, 0 = padding).
        temperature: Temperature scaling parameter tau.

    Returns:
        Scalar loss.
    """
    B = anchor.size(0)
    N = negatives.size(1)

    # Positive logits: (B, 1)
    pos_sim = (anchor * positive).sum(dim=-1) / temperature  # (B,)
    pos_logits = pos_sim.unsqueeze(1)  # (B, 1)

    # Negative logits: (B, N)
    neg_sim = (anchor.unsqueeze(1) * negatives).sum(dim=-1) / temperature  # (B, N)

    # Combine logits: (B, 1 + N)
    logits = torch.cat([pos_logits, neg_sim], dim=1)  # (B, 1+N)

    # Mask for valid negatives (negatives with non-zero mask get logit, padding gets -inf)
    full_mask = torch.cat([
        torch.ones(B, 1, device=negative_mask.device),
        negative_mask,
    ], dim=1)

    logits = logits - (1 - full_mask) * 1e9

    # Labels: first position is the positive
    labels = torch.zeros(B, dtype=torch.long, device=anchor.device)

    loss = F.cross_entropy(logits, labels)
    return loss


def train_bi_encoder(
    train_dataset: Any,
    eval_dataset: Optional[Any],
    args: Optional[BiEncoderTrainingArgs] = None,
    model: Optional[BiEncoderModel] = None,
) -> BiEncoderModel:
    """Train the Bi-Encoder model with contrastive learning.

    Args:
        train_dataset: Dataset yielding {anchor, positive, negatives} dicts.
        eval_dataset: Optional validation dataset.
        args: Training hyperparameters.
        model: Pre-initialized BiEncoderModel (creates new if None).

    Returns:
        Trained BiEncoderModel.
    """
    args = args or BiEncoderTrainingArgs()
    logger = get_logger("pcap2rule.train_bi_encoder")
    set_seed(args.seed)

    model = model or BiEncoderModel(embedding_dim=args.embedding_dim,
                                     max_seq_length=args.max_seq_length)
    model.train()

    # Only the projection layer is trainable (encoder is frozen)
    optimizer = AdamW(
        model.projection.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda batch: batch,
    )

    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)

    os.makedirs(args.output_dir, exist_ok=True)
    global_step = 0

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"Bi-Encoder Epoch {epoch + 1}/{args.num_epochs}")

        for step, batch in enumerate(progress):
            anchors = [item['anchor'] for item in batch]
            positives = [item['positive'] for item in batch]
            negatives = [item.get('negatives', [''] * args.num_negatives) for item in batch]
            neg_mask = torch.ones(len(batch), args.num_negatives)

            # Set mask for padding
            for i, neg_list in enumerate(negatives):
                for j, n in enumerate(neg_list):
                    if not n or n == '':
                        neg_mask[i, j] = 0

            anchor_emb, positive_emb, negative_embs = model(anchors, positives, negatives)
            loss = contrastive_loss(
                anchor_emb, positive_emb, negative_embs, neg_mask,
                temperature=args.temperature,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.projection.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            epoch_loss += loss.item()
            progress.set_postfix({'loss': f'{loss.item():.4f}'})

            if global_step % args.logging_steps == 0:
                logger.info(
                    f"Step {global_step}: loss={loss.item():.4f}, "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} complete: avg_loss={avg_loss:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(args.output_dir, f'epoch_{epoch + 1}')
        os.makedirs(ckpt_path, exist_ok=True)
        torch.save(model.projection.state_dict(),
                   os.path.join(ckpt_path, 'projection.pt'))
        model.encoder.save(os.path.join(ckpt_path, 'encoder'))

    final_path = os.path.join(args.output_dir, 'final')
    os.makedirs(final_path, exist_ok=True)
    torch.save(model.projection.state_dict(), os.path.join(final_path, 'projection.pt'))
    model.encoder.save(os.path.join(final_path, 'encoder'))
    logger.info(f"Bi-Encoder training complete. Model saved to {final_path}")

    return model


def load_bi_encoder(path: str, embedding_dim: int = 768) -> BiEncoderModel:
    """Load a trained Bi-Encoder from checkpoint.

    Args:
        path: Path to the checkpoint directory.
        embedding_dim: Dimension of the projection layer.

    Returns:
        Loaded BiEncoderModel.
    """
    model = BiEncoderModel(embedding_dim=embedding_dim)
    model.projection.load_state_dict(
        torch.load(os.path.join(path, 'projection.pt'), map_location='cpu')
    )
    model.encoder = SentenceTransformer(os.path.join(path, 'encoder'))
    return model
