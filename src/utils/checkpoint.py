"""Checkpoint loading with a safe-first fallback.

`torch.load(..., weights_only=False)` allows arbitrary code execution via
pickle deserialization. PyTorch 2.6 changed the default to `weights_only=True`
for that reason. Our training scripts historically saved numpy arrays alongside
tensors in checkpoints, which `weights_only=True` cannot deserialize; those
checkpoints therefore need an explicit fallback.

This module centralizes that fallback so it is:

  * the *default* path is `weights_only=True` (safe),
  * the unsafe path only fires when a checkpoint genuinely needs it,
  * there is exactly one `# nosec B614` in the codebase, with documented
    rationale, rather than one per call site.

If you add another `torch.load` call, use this helper. If you must bypass it,
add a rationale at the call site explaining why the checkpoint is trusted.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Union

import torch


def load_checkpoint_safe(
    path: Union[str, Path],
    *,
    map_location: Union[str, torch.device] = "cpu",
) -> Any:
    """Load a PyTorch checkpoint, preferring `weights_only=True`.

    Parameters
    ----------
    path:
        Path to a checkpoint file (typically a ``.pt`` produced by one of the
        training scripts in this repo).
    map_location:
        Passed through to ``torch.load``.

    Returns
    -------
    The deserialized checkpoint, typically a ``dict``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist. (torch.load would raise this anyway; the
        explicit check produces a clearer message.)
    RuntimeError, pickle.UnpicklingError
        If loading fails for a reason unrelated to the weights_only fallback.

    Notes
    -----
    The fallback to ``weights_only=False`` is only triggered by errors that
    indicate the checkpoint contains objects outside the weights_only
    safelist. Any other error propagates so we do not silently downgrade on
    unrelated failures (e.g. truncated files).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")

    try:
        return torch.load(p, map_location=map_location, weights_only=True)
    except (RuntimeError, pickle.UnpicklingError) as exc:
        msg = str(exc).lower()
        # The two reliable markers for "checkpoint contains non-safelisted
        # objects" across PyTorch versions:
        #   * "Unsupported global: <name>" — from _weights_only_unpickler
        #   * messages that explicitly mention weights_only
        if "unsupported global" not in msg and "weights_only" not in msg:
            raise
        # Legacy checkpoint. Path is a local file the caller controls; the
        # user could only attack themselves. Bandit flags this line as B614.
        return torch.load(p, map_location=map_location, weights_only=False)  # nosec B614
