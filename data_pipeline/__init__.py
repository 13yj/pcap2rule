"""Dataset module — provides dataset loaders, sampling utilities, and synthetic data generation."""

from .dataset_base import BaseDataset, TrafficSample
from .cse_cic_ids2018 import CSE_CIC_IDS2018
from .cic_ids2017 import CIC_IDS2017
from .synthetic_data import (
    SyntheticSample,
    generate_synthetic_dataset,
    save_synthetic_dataset,
    load_synthetic_dataset,
    build_reverse_prompt,
    parse_synthetic_response,
    quality_check_sample,
)

__all__ = [
    'BaseDataset',
    'TrafficSample',
    'CSE_CIC_IDS2018',
    'CIC_IDS2017',
    'SyntheticSample',
    'generate_synthetic_dataset',
    'save_synthetic_dataset',
    'load_synthetic_dataset',
    'build_reverse_prompt',
    'parse_synthetic_response',
    'quality_check_sample',
]
