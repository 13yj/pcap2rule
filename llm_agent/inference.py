"""LLM Agent inference interface — batch generation with configurable sampling."""

import re
from typing import List, Optional, Dict, Any


class InferenceEngine:
    """Unified inference interface supporting both vLLM and HuggingFace backends.

    Args:
        backend: 'vllm' or 'hf' (HuggingFace).
        model: The loaded model object (vLLM LLM or HF model).
        tokenizer: HuggingFace tokenizer (HF backend only).
        temperature: Sampling temperature.
        max_tokens: Maximum generation length.
        top_p: Nucleus sampling parameter.
    """

    def __init__(
        self,
        backend: str = "vllm",
        model: Any = None,
        tokenizer: Any = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
        top_p: float = 0.95,
    ):
        self.backend = backend
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

    def generate(self, prompt: str) -> str:
        """Generate a single completion."""
        results = self.batch_generate([prompt])
        return results[0] if results else ""

    def batch_generate(self, prompts: List[str]) -> List[str]:
        """Generate completions for a batch of prompts."""
        if self.backend == "vllm":
            return self._generate_vllm(prompts)
        else:
            return self._generate_hf(prompts)

    def _generate_vllm(self, prompts: List[str]) -> List[str]:
        """vLLM batched generation."""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            stop=["</specific_rule>", "</generalized_rule>"],
        )

        outputs = self.model.generate(prompts, sampling_params)
        return [o.outputs[0].text for o in outputs]

    def _generate_hf(self, prompts: List[str]) -> List[str]:
        """HuggingFace model.generate based inference."""
        import torch

        results = []
        for prompt in prompts:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=self.temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
            )
            results.append(text)

        return results

    def extract_rules(self, generated_text: str) -> Dict[str, str]:
        """Parse generated text to extract specific and generalized rules.

        Args:
            generated_text: Raw LLM output.

        Returns:
            Dict with 'specific' and 'generalized' keys.
        """
        result = {'specific': '', 'generalized': ''}

        # Extract specific rule
        specific_match = re.search(
            r'<specific_rule>\s*(.*?)\s*</specific_rule>',
            generated_text, re.DOTALL | re.IGNORECASE
        )
        if specific_match:
            result['specific'] = specific_match.group(1).strip()

        # Extract generalized rule
        gen_match = re.search(
            r'<generalized_rule>\s*(.*?)\s*</generalized_rule>',
            generated_text, re.DOTALL | re.IGNORECASE
        )
        if gen_match:
            result['generalized'] = gen_match.group(1).strip()

        # Fallback: find "alert ..." patterns
        if not result['specific']:
            alert_matches = re.findall(
                r'alert\s+\w+\s+.*?;\s*\)',
                generated_text, re.DOTALL
            )
            if alert_matches:
                result['specific'] = alert_matches[0].strip()
                if len(alert_matches) > 1:
                    result['generalized'] = alert_matches[-1].strip()

        return result
