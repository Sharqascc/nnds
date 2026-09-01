"""Seed management for reproducibility."""
import random

import numpy as np
import torch


def set_seed(seed=42):
    """Set random seed for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_seed():
    return 42
