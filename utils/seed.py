"""Random seed management for reproducibility."""

import random
import os
import numpy as np
from typing import List


SEEDS = [42, 123, 456, 789, 1024]


def set_seed(seed: int):
    """Set all random seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def get_seeds() -> List[int]:
    """Return the list of default random seeds used for multi-seed evaluation."""
    return SEEDS
