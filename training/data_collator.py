"""Data collators for supervised fine-tuning and contrastive learning."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import torch


@dataclass
class SFTCollator:
    """Collates prompt-rule pairs for causal LM fine-tuning.

    For each sample: prompt (instruction + flow + payload + exemplars) is the input,
    and the target Suricata rule is the label. Uses left-padding for inference.
    """

    tokenizer: Any
    max_length: int = 2048
    pad_to_max_length: bool = True

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        prompts = [item['prompt'] for item in batch]
        targets = [item['target'] for item in batch]

        full_texts = [p + t + self.tokenizer.eos_token for p, t in zip(prompts, targets)]

        tokenized = self.tokenizer(
            full_texts,
            max_length=self.max_length,
            padding='max_length' if self.pad_to_max_length else 'longest',
            truncation=True,
            return_tensors='pt',
        )

        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']

        # Labels: -100 for prompt tokens (ignored in loss)
        labels = input_ids.clone()
        for i, (p, t) in enumerate(zip(prompts, targets)):
            prompt_tokens = self.tokenizer(p, add_special_tokens=False)['input_ids']
            prompt_len = len(prompt_tokens)
            labels[i, :prompt_len] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


@dataclass
class ContrastiveCollator:
    """Collates (anchor, positive, negatives) triplets for Bi-Encoder training.

    Produces: anchor (B, D), positive (B, D), negatives (B, N_neg, D)
    where the Bi-Encoder encodes each item separately.
    """

    max_negatives: int = 8

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        anchors = [item['anchor'] for item in batch]
        positives = [item['positive'] for item in batch]
        negatives = [item.get('negatives', []) for item in batch]

        # Pad negatives to max_negatives
        padded_negatives = []
        negative_mask = []
        for neg_list in negatives:
            n = min(len(neg_list), self.max_negatives)
            padded = neg_list[:n] + [''] * (self.max_negatives - n)
            padded_negatives.append(padded)
            mask = [1] * n + [0] * (self.max_negatives - n)
            negative_mask.append(mask)

        return {
            'anchor': anchors,
            'positive': positives,
            'negatives': padded_negatives,
            'negative_mask': torch.tensor(negative_mask, dtype=torch.float32),
        }
