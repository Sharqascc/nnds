"""Tests for src.utils.checkpoint.load_checkpoint_safe."""

from __future__ import annotations

import pickle

import pytest
import torch

from src.utils.checkpoint import load_checkpoint_safe


class _Untrusted:
    """A class not on the weights_only safelist; used to trigger the fallback."""

    def __init__(self, value: int = 42) -> None:
        self.value = value


def test_loads_plain_tensor_checkpoint_via_safe_path(tmp_path):
    """A dict of tensors is deserialized by weights_only=True (no fallback)."""
    ckpt = {"mean": torch.tensor([1.0, 2.0, 3.0]), "std": torch.tensor([0.1, 0.2, 0.3])}
    path = tmp_path / "plain.pt"
    torch.save(ckpt, path)

    loaded = load_checkpoint_safe(path)
    assert torch.equal(loaded["mean"], ckpt["mean"])
    assert torch.equal(loaded["std"], ckpt["std"])


def test_falls_back_for_checkpoint_with_non_safelisted_object(tmp_path):
    """A checkpoint containing a custom instance triggers the fallback path."""
    ckpt = {"obj": _Untrusted(value=7), "mean": torch.tensor([0.0])}
    path = tmp_path / "legacy.pt"
    torch.save(ckpt, path)

    loaded = load_checkpoint_safe(path)
    assert isinstance(loaded["obj"], _Untrusted)
    assert loaded["obj"].value == 7


def test_missing_path_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint_safe(tmp_path / "does-not-exist.pt")


def test_non_weights_only_error_propagates(tmp_path):
    """A truncated file should not silently downgrade to weights_only=False."""
    path = tmp_path / "truncated.pt"
    path.write_bytes(b"not-a-real-checkpoint")
    with pytest.raises((RuntimeError, pickle.UnpicklingError, EOFError)):
        load_checkpoint_safe(path)
