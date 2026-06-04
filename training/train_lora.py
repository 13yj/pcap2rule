"""LoRA fine-tuning script for Qwen2-7B (Section IV-B).

Training data: D = D1 (synthetic 2000) + D2 (real 1500) = 3500 samples.
Configuration:
  - LoRA r=16, alpha=32 on all attention + FFN linear layers
  - 4-bit QLoRA (NF4 quantization, double-quantization)
  - lr=2e-4, cosine schedule, 3 epochs
  - Batch size 4, gradient accumulation 4 (effective batch 16)
  - Max sequence length 2048 tokens
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from ..utils.logging import get_logger
from ..utils.seed import set_seed
from ..utils.config import Config
from ..llm_agent.lora_trainer import LoRAConfig, setup_lora_model, save_lora_checkpoint
from ..data_pipeline.dataset_base import BaseDataset, TrafficSample
from ..pcap_processor.prompt_builder import PromptBuilder


@dataclass
class LoRATrainingArgs:
    """Paper-specified training hyperparameters."""
    output_dir: str = './checkpoints/lora_adapters'
    num_epochs: int = 3
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_seq_length: int = 2048
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 500
    max_grad_norm: float = 1.0
    fp16: bool = True
    seed: int = 42


class SFTDataset(Dataset):
    """Supervised fine-tuning dataset from D1 + D2."""

    def __init__(
        self,
        samples: List[Any],
        prompt_builder: PromptBuilder,
        tokenizer: Any,
        max_length: int = 2048,
    ):
        self.samples = samples
        self.prompt_builder = prompt_builder
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        sample = self.samples[idx]

        # Build prompt from flow features + payload
        if hasattr(sample, 'flow_features') and hasattr(sample, 'payload_text'):
            prompt = self.prompt_builder.build(
                flow_features=sample.flow_features,
                payload_text=sample.payload_text,
                exemplars=[],
                attack_type=sample.attack_type if hasattr(sample, 'attack_type') else None,
            )
        else:
            prompt = str(sample)

        target = sample.ground_truth_rule if hasattr(sample, 'ground_truth_rule') else ''

        return {'prompt': prompt, 'target': target}


def load_training_dataset(
    dataset: BaseDataset,
    synthetic_samples: Optional[List[Any]] = None,
) -> SFTDataset:
    """Load and combine D1 + D2 training data.

    Args:
        dataset: Real traffic dataset (D2).
        synthetic_samples: Synthetic samples from D1 generation.

    Returns:
        SFTDataset combining both sources.
    """
    real_samples = dataset.get_split('train')
    all_samples = list(real_samples)

    if synthetic_samples:
        all_samples.extend(synthetic_samples)

    return all_samples


def train_lora(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Optional[Any],
    args: Optional[LoRATrainingArgs] = None,
    lora_config: Optional[LoRAConfig] = None,
    config: Optional[Config] = None,
) -> Any:
    """Run LoRA fine-tuning of Qwen2-7B.

    Args:
        model: Loaded base model (HuggingFace AutoModelForCausalLM).
        tokenizer: Tokenizer for the base model.
        train_dataset: PyTorch Dataset of training samples.
        eval_dataset: Optional validation dataset.
        args: Training hyperparameters.
        lora_config: LoRA adapter configuration.
        config: Global configuration.

    Returns:
        Trained PEFT model.
    """
    args = args or LoRATrainingArgs()
    lora_config = lora_config or LoRAConfig()
    logger = get_logger("pcap2rule.train_lora")

    set_seed(args.seed)

    # Apply LoRA
    model = setup_lora_model(model, lora_config)

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    eval_loader = None
    if eval_dataset is not None:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.per_device_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)

    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    global_step = 0
    model.train()

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")

        for step, batch in enumerate(progress):
            batch = {k: v.to(model.device) for k, v in batch.items()
                    if isinstance(v, torch.Tensor)}

            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item()
            progress.set_postfix({'loss': f'{loss.item():.4f}'})

            if global_step > 0 and global_step % args.logging_steps == 0:
                avg_loss = epoch_loss / (step + 1)
                logger.info(
                    f"Step {global_step}: loss={avg_loss:.4f}, "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

            if global_step > 0 and global_step % args.save_steps == 0:
                save_lora_checkpoint(
                    model,
                    os.path.join(args.output_dir, f'checkpoint-{global_step}'),
                    tokenizer,
                )

        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} complete: avg_loss={avg_epoch_loss:.4f}")

        # Evaluation
        if eval_loader is not None:
            eval_loss = _evaluate(model, eval_loader)
            logger.info(f"Epoch {epoch + 1} eval_loss={eval_loss:.4f}")

    # Final save
    final_path = os.path.join(args.output_dir, 'final')
    save_lora_checkpoint(model, final_path, tokenizer)
    logger.info(f"Training complete. Model saved to {final_path}")

    return model


def _evaluate(model: Any, eval_loader: DataLoader) -> float:
    """Compute evaluation loss."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(model.device) for k, v in batch.items()
                    if isinstance(v, torch.Tensor)}
            outputs = model(**batch)
            total_loss += outputs.loss.item()

    model.train()
    return total_loss / len(eval_loader)
