"""Seed management for reproducibility."""

import random

import numpy as np
import torch

# Module-level variable to store the current seed
_seed = 42


def set_seed(seed=42):
    """Set random seed for Python, NumPy, and PyTorch."""
    global _seed
    _seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_seed():
    """Return the seed currently set for reproducibility."""
    return _seed

