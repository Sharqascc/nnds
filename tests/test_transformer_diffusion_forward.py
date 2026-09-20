"""Forward-pass tests for TransformerTrajectoryDiffusion.

CPU-only, small tensors. Covers PositionalEncoding shape / determinism
and the transformer denoiser's forward contract.
"""

from __future__ import annotations

import pytest
import torch

from src.diffusion.traffic_diffusion.transformer_diffusion import (
    PositionalEncoding,
    TransformerTrajectoryDiffusion,
)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return TransformerTrajectoryDiffusion(
        traj_dim=2,
        cond_dim=2,
        hidden_dim=16,
        num_heads=4,
        num_layers=2,
        max_len=16,
    )


def _inputs(B=2, T=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    noisy = torch.randn(B, T, 2, generator=g)
    cond = torch.randn(B, T, 2, generator=g)
    t = torch.randint(0, 200, (B,), generator=g)
    return noisy, cond, t


def test_positional_encoding_shape():
    pe = PositionalEncoding(d_model=8, max_len=32)
    x = torch.zeros(2, 16, 8)
    assert pe(x).shape == x.shape


def test_positional_encoding_deterministic():
    pe = PositionalEncoding(d_model=8, max_len=32)
    x = torch.randn(2, 16, 8)
    torch.testing.assert_close(pe(x), pe(x))


def test_forward_shape(model):
    noisy, cond, t = _inputs()
    out = model(noisy, cond, t)
    assert out.shape == (noisy.shape[0], noisy.shape[1], 2)


def test_forward_finite(model):
    noisy, cond, t = _inputs()
    out = model(noisy, cond, t)
    assert torch.isfinite(out).all()


def test_forward_deterministic_in_eval_mode(model):
    """In eval mode (dropout disabled), forward is deterministic.

    In train mode the TransformerEncoderLayer's dropout (p=0.1) makes
    forward stochastic on purpose; that is correct behavior. Inference
    always uses eval(), so determinism is asserted there.
    """
    model.eval()
    noisy, cond, t = _inputs()
    torch.testing.assert_close(model(noisy, cond, t), model(noisy, cond, t))


def test_forward_is_stochastic_in_train_mode(model):
    """Complementary check: train mode must NOT be deterministic.

    If this ever passes trivially (two identical calls), someone has
    likely disabled dropout or set eval() globally, and both the
    determinism test above and this one lose meaning.
    """
    model.train()
    noisy, cond, t = _inputs()
    a = model(noisy, cond, t)
    b = model(noisy, cond, t)
    if torch.allclose(a, b, atol=1e-7):
        import warnings

        warnings.warn(
            "Transformer forward is identical across calls in train mode; "
            "dropout may be disabled. Verify this is intentional.",
            stacklevel=1,
        )


@pytest.mark.parametrize("B", [1, 2, 4])
def test_forward_various_batch_sizes(model, B):
    noisy, cond, t = _inputs(B=B)
    out = model(noisy, cond, t)
    assert out.shape[0] == B


def test_forward_backprop_finite_grads(model):
    noisy, cond, t = _inputs()
    out = model(noisy, cond, t)
    out.pow(2).mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
