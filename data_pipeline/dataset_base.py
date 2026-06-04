"""Base dataset class with common PCAP dataset loading, sampling, and splitting.

Provides train/val/test splitting by date, stratified sampling by attack type,
and benign sample collection — matching the paper's experimental protocol.
"""

import os
import random
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrafficSample:
    """A single traffic sample with extracted features and metadata."""

    sample_id: str
    pcap_path: str
    attack_type: str           # e.g. 'DoS', 'Brute Force', 'benign'
    flow_features: Optional[np.ndarray] = None   # (4, 38) segment-level stats
    payload_text: str = ""                         # extracted payload text
    ground_truth_rule: Optional[str] = None        # ideal Suricata rule if available
    metadata: Dict[str, Any] = field(default_factory=dict)  # extra info


class BaseDataset(ABC):
    """Abstract base for PCAP intrusion detection datasets.

    Subclasses must implement _load_sample_list() and _parse_label().
    """

    def __init__(
        self,
        data_dir: str,
        seed: int = 42,
        attack_types: Optional[List[str]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.attack_types = attack_types or []
        self.samples: Dict[str, List[TrafficSample]] = {}  # split -> samples

    @abstractmethod
    def _load_sample_list(self) -> List[TrafficSample]:
        """Load and return the full list of samples from disk.

        Returns:
            List of TrafficSample objects.
        """
        ...

    @abstractmethod
    def _parse_label(self, raw_label: str) -> str:
        """Normalize a raw dataset label to a canonical attack type name.

        Args:
            raw_label: Raw label string from the dataset.

        Returns:
            Canonical attack type name (or 'benign').
        """
        ...

    def load(self) -> List[TrafficSample]:
        """Load all samples and cache them."""
        samples = self._load_sample_list()
        for s in samples:
            s.attack_type = self._parse_label(s.attack_type)
        return samples

    def split_by_date(
        self,
        samples: List[TrafficSample],
        train_dates: List[str],
        test_dates: List[str],
    ) -> Dict[str, List[TrafficSample]]:
        """Split samples into train/val/test by capture date.

        Args:
            samples: List of all loaded samples.
            train_dates: List of date strings for training (e.g. ['2018-02-14']).
            test_dates: List of date strings for testing.

        Returns:
            Dict with 'train', 'val', 'test' keys.
        """
        train, val, test = [], [], []
        for s in samples:
            date = s.metadata.get('date', '')
            if any(d in date for d in train_dates):
                train.append(s)
            elif any(d in date for d in test_dates):
                test.append(s)
            else:
                val.append(s)
        # If no date info, do random split
        if not train and not test:
            return self.split_random(samples)
        self.samples = {'train': train, 'val': val, 'test': test}
        return self.samples

    def split_random(
        self,
        samples: List[TrafficSample],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Dict[str, List[TrafficSample]]:
        """Random stratified split.

        Args:
            samples: List of all loaded samples.
            train_ratio: Fraction for training.
            val_ratio: Fraction for validation.

        Returns:
            Dict with 'train', 'val', 'test' keys.
        """
        indices = list(range(len(samples)))
        self.rng.shuffle(indices)
        n = len(samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        self.samples = {
            'train': [samples[i] for i in indices[:n_train]],
            'val': [samples[i] for i in indices[n_train:n_train + n_val]],
            'test': [samples[i] for i in indices[n_train + n_val:]],
        }
        return self.samples

    def sample_stratified(
        self,
        samples: List[TrafficSample],
        n_per_attack: int = 100,
        n_benign: int = 200,
    ) -> Tuple[List[TrafficSample], List[TrafficSample]]:
        """Stratified sampling of attack and benign samples.

        Args:
            samples: Source sample list.
            n_per_attack: Number of attack samples per attack type.
            n_benign: Total number of benign samples.

        Returns:
            (attack_samples, benign_samples) tuple.
        """
        attack_by_type: Dict[str, List[TrafficSample]] = {}
        benign = []
        for s in samples:
            if s.attack_type == 'benign':
                benign.append(s)
            else:
                attack_by_type.setdefault(s.attack_type, []).append(s)

        # Sample n_per_attack from each type
        attack_samples = []
        for atype, slist in attack_by_type.items():
            if len(slist) > n_per_attack:
                chosen = self.rng.choice(slist, n_per_attack, replace=False)
            else:
                chosen = slist
            attack_samples.extend(chosen)

        # Sample benign
        if len(benign) > n_benign:
            benign = list(self.rng.choice(benign, n_benign, replace=False))

        return attack_samples, benign

    def get_split(self, split: str) -> List[TrafficSample]:
        """Get samples for a specific split."""
        return self.samples.get(split, [])

    @property
    def attack_type_counts(self) -> Dict[str, int]:
        """Count samples per attack type in the loaded dataset."""
        counts: Dict[str, int] = {}
        for samples in self.samples.values():
            for s in samples:
                counts[s.attack_type] = counts.get(s.attack_type, 0) + 1
        return counts
