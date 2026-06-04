"""LLM model loading and serving interface.

Supports two inference backends:
1. vLLM (PagedAttention, batched inference) — recommended for production
2. HuggingFace transformers (with model.generate) — for environments without GPU
"""

import os
from typing import Optional, List, Dict, Any


def load_model_vllm(
    model_name: str = "Qwen/Qwen2-7B-Instruct",
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.90,
    **kwargs,
):
    """Load a model with vLLM for efficient batched inference.

    Args:
        model_name: HuggingFace model ID or local path.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        max_model_len: Maximum sequence length.
        gpu_memory_utilization: Fraction of GPU memory to use.

    Returns:
        vLLM LLM instance.
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError(
            "vLLM is required for this backend. "
            "Install with: pip install vllm"
        )

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        **kwargs,
    )
    return llm


def load_model_hf(
    model_name: str = "Qwen/Qwen2-7B-Instruct",
    load_in_4bit: bool = True,
    device_map: str = "auto",
    **kwargs,
):
    """Load a model with HuggingFace transformers (optionally 4-bit quantized).

    Args:
        model_name: HuggingFace model ID or local path.
        load_in_4bit: Use 4-bit quantization (bitsandbytes required).
        device_map: Device mapping strategy.

    Returns:
        (model, tokenizer) tuple.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_kwargs = {
        "trust_remote_code": True,
        "device_map": device_map,
        **kwargs,
    }

    if load_in_4bit:
        model_kwargs.update({
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True,
        })

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_lora_adapter(
    base_model,
    tokenizer,
    adapter_path: str,
):
    """Load a LoRA adapter onto a base model.

    Args:
        base_model: Base HuggingFace model.
        tokenizer: Associated tokenizer.
        adapter_path: Path to LoRA adapter weights.

    Returns:
        Model with LoRA adapter loaded.
    """
    from peft import PeftModel

    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model
