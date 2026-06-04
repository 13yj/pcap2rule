"""LLM agent module — model loading, CoT prompts, agent loop, inference, and LoRA training."""

from .model import load_model_vllm, load_model_hf, load_lora_adapter
from .cot_prompts import build_cot_prompt, STEP1_ATTACK_IDENTIFICATION
from .agent import Pcap2RuleAgent
from .inference import InferenceEngine
from .lora_trainer import (
    LoRAConfig,
    create_lora_config,
    setup_lora_model,
    save_lora_checkpoint,
    load_lora_for_inference,
)

__all__ = [
    'load_model_vllm',
    'load_model_hf',
    'load_lora_adapter',
    'build_cot_prompt',
    'Pcap2RuleAgent',
    'InferenceEngine',
    'LoRAConfig',
    'create_lora_config',
    'setup_lora_model',
    'save_lora_checkpoint',
    'load_lora_for_inference',
]
