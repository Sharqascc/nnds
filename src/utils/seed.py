"""Seed management for reproducibility.

`set_seed(seed)` seeds Python's `random`, NumPy, and PyTorch, and enables the
backend flags that make CUDA/cuDNN/cuBLAS deterministic. It also exports
`PYTHONHASHSEED` so subprocesses spawned by the caller inherit the same hash
randomization behavior.

Determinism is best-effort: some CUDA ops have no deterministic implementation.
`deterministic_torch=True` (default) calls
`torch.use_deterministic_algorithms(..., warn_only=True)`, which emits a
warning for those ops rather than raising. Pass `deterministic_torch=False`
for maximum throughput when reproducibility is not required.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

# Module-level variable to store the current seed
_seed = 42


def set_seed(seed: int = 42, *, deterministic_torch: bool = True) -> None:
    """Set random seed for Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed:
        Integer seed. `bool` is explicitly rejected (a bare `True` is almost
        certainly a bug in the caller).
    deterministic_torch:
        If True (default), enable PyTorch's deterministic algorithms and
        disable cuDNN autotuning. This slows some ops but makes results
        reproducible across runs. If False, only seed the RNGs and leave
        backend flags alone.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an integer, got {type(seed).__name__}")

    global _seed
    _seed = seed

    # Affects hash randomization in any subprocess we spawn after this point.
    # (The current process read PYTHONHASHSEED at startup and will not change.)
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        # Required for deterministic cuBLAS matmuls on CUDA (per PyTorch docs).
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # torch < 1.11: warn_only kwarg not accepted
            torch.use_deterministic_algorithms(True)


def get_seed() -> int:
    """Return the seed currently set for reproducibility."""
    return _seed
