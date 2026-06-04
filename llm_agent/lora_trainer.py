"""LoRA fine-tuning wrapper for Qwen2-7B.

Uses PEFT + bitsandbytes 4-bit QLoRA:
  - r=16, alpha=32
  - Target: all linear layers in attention + FFN
  - 4-bit NF4 quantization
  - Double-quantization for memory efficiency
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader

from ..utils.logging import get_logger
from ..utils.config import Config


@dataclass
class LoRAConfig:
    """LoRA hyperparameters from paper Section III-E."""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: tuple = (
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj',
    )
    bias: str = 'none'
    task_type: str = 'CAUSAL_LM'


def create_lora_config(base_config: Optional[Config] = None) -> LoRAConfig:
    """Create LoRAConfig from YAML config or defaults."""
    if base_config and hasattr(base_config, 'lora'):
        cfg = base_config.lora
        return LoRAConfig(
            r=cfg.get('r', 16),
            lora_alpha=cfg.get('alpha', 32),
            lora_dropout=cfg.get('dropout', 0.1),
        )
    return LoRAConfig()


def setup_lora_model(
    base_model: Any,
    lora_config: Optional[LoRAConfig] = None,
) -> Any:
    """Apply LoRA adapters to a HuggingFace model.

    Args:
        base_model: Loaded transformers model (AutoModelForCausalLM).
        lora_config: LoRA configuration. Uses defaults if None.

    Returns:
        PEFT model with LoRA adapters attached.
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    lora_config = lora_config or LoRAConfig()

    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=list(lora_config.target_modules),
        bias=lora_config.bias,
        task_type=lora_config.task_type,
    )

    # Prepare 4-bit model for training
    base_model = prepare_model_for_kbit_training(base_model)
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    return model


def save_lora_checkpoint(
    model: Any,
    output_dir: str,
    tokenizer: Optional[Any] = None,
):
    """Save LoRA adapter weights and tokenizer."""
    model.save_pretrained(output_dir)
    if tokenizer:
        tokenizer.save_pretrained(output_dir)


def load_lora_for_inference(
    base_model: Any,
    adapter_path: str,
):
    """Load trained LoRA adapter for inference.

    Args:
        base_model: Base HuggingFace model (without LoRA applied).
        adapter_path: Path to saved LoRA adapter.

    Returns:
        Model with LoRA adapter loaded for inference.
    """
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    return model
