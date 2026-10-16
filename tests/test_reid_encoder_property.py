"""Property-based tests for ReIDEncoder.

The MobileNetV3 model is replaced by torch.nn.Identity so the test does not
download weights. Every property checks only the outer encode_crop logic.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from src.pipeline.reid_encoder import ReIDEncoder


def _make_encoder() -> ReIDEncoder:
    with patch("src.pipeline.reid_encoder.models.mobilenet_v3_small") as m:
        m.return_value = torch.nn.Identity()
        return ReIDEncoder(device="cpu")


def _frame(seed: int = 0, shape=(100, 100, 3)):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, shape, dtype=np.uint8)


def test_embedding_is_1d_and_normalized():
    enc = _make_encoder()
    emb = enc.encode_crop(_frame(), 10, 10, 60, 60)
    assert emb is not None
    assert emb.ndim == 1
    assert emb.dtype == np.float32 or emb.dtype == np.float64
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-4


def test_degenerate_box_returns_none():
    enc = _make_encoder()
    f = _frame()
    assert enc.encode_crop(f, 50, 50, 50, 50) is None
    assert enc.encode_crop(f, 60, 50, 40, 50) is None
    assert enc.encode_crop(f, 50, 60, 50, 40) is None


@given(
    x1=st.integers(-50, 150),
    y1=st.integers(-50, 150),
    x2=st.integers(-50, 150),
    y2=st.integers(-50, 150),
)
def test_valid_bounds_yield_normalized_embedding(x1, y1, x2, y2):
    enc = _make_encoder()
    f = _frame()
    result = enc.encode_crop(f, x1, y1, x2, y2)
    if result is None:
        # either out of bounds or degenerate
        return
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert abs(float(np.linalg.norm(result)) - 1.0) < 1e-4


@given(seed=st.integers(0, 1000))
def test_determinism(seed):
    enc = _make_encoder()
    f = _frame(seed)
    a = enc.encode_crop(f, 20, 20, 70, 70)
    b = enc.encode_crop(f, 20, 20, 70, 70)
    if a is None:
        assert b is None
    else:
        assert b is not None
        assert np.allclose(a, b, atol=1e-6)


@given(
    H=st.integers(10, 200),
    W=st.integers(10, 200),
)
def test_frame_shape_independence(H, W):
    enc = _make_encoder()
    f = _frame(shape=(H, W, 3))
    # Use a crop that stays inside the frame
    x1, y1 = 0, 0
    x2, y2 = min(W, 50), min(H, 50)
    emb = enc.encode_crop(f, x1, y1, x2, y2)
    # Should not raise
    if emb is not None:
        assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-4


def test_crop_clamped_to_frame_bounds():
    enc = _make_encoder()
    f = _frame(shape=(50, 50, 3))
    # Ask for a crop larger than the frame; code clamps
    emb = enc.encode_crop(f, -10, -10, 200, 200)
    assert emb is not None
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-4
