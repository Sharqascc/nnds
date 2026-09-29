"""Forward-pass tests for TrajectoryDiffusionModel.

CPU-only, small tensors, no training loop. Verifies forward shape and
finiteness, loss backprop, and sampler shape / determinism.
"""

from __future__ import annotations

import pytest
import torch

from src.diffusion.traffic_diffusion.trajectory_diffusion import (
    TrajectoryDiffusionModel,
)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return TrajectoryDiffusionModel(
        traj_shape=(8, 1, 2),  # T=8, N=1 agents, D=2 dims
        cond_dim=4,
        hidden_dim=16,
    )


def _inputs(B=2, Th=8, cond_dim=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, Th, 1, 2, generator=g)
    cond = torch.randn(B, cond_dim, generator=g)
    t = torch.rand(B, 1, generator=g)
    return x, cond, t


def test_forward_shape(model):
    x, cond, t = _inputs()
    out = model(x, cond, t)
    assert out.shape == x.shape


def test_forward_finite(model):
    x, cond, t = _inputs()
    out = model(x, cond, t)
    assert torch.isfinite(out).all()


def test_forward_deterministic(model):
    model.eval()
    x, cond, t = _inputs()
    torch.testing.assert_close(model(x, cond, t), model(x, cond, t))


def test_forward_accepts_1d_t(model):
    x, cond, _ = _inputs()
    t = torch.rand(x.shape[0])
    out = model(x, cond, t)
    assert out.shape == x.shape


def test_forward_t_normalization_branch(model):
    # forward divides t by 100 whenever t.max() > 1.0. Both sides of the
    # branch should produce finite output.
    x, cond, _ = _inputs()
    for t_val in (0.5, 50.0):
        t = torch.full((x.shape[0], 1), t_val)
        out = model(x, cond, t)
        assert torch.isfinite(out).all()


def test_forward_handles_cond_3d(model):
    # The forward reshapes cond to (B, -1) when cond.ndim > 2.
    x, _, t = _inputs()
    cond_3d = torch.randn(x.shape[0], 4, 1)
    out = model(x, cond_3d, t)
    assert out.shape == x.shape


def test_compute_loss_scalar_and_finite(model):
    x, cond, _ = _inputs()
    loss = model.compute_loss(x, cond)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_compute_loss_backprop_produces_finite_grads(model):
    x, cond, _ = _inputs()
    loss = model.compute_loss(x, cond)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_sample_shape(model):
    _, cond, _ = _inputs()
    samples = model.sample(cond, num_steps=4)
    assert samples.shape == (cond.shape[0], 8, 1, 2)


def test_sample_finite(model):
    _, cond, _ = _inputs()
    samples = model.sample(cond, num_steps=4)
    assert torch.isfinite(samples).all()


def test_sample_deterministic_with_seed(model):
    _, cond, _ = _inputs()
    torch.manual_seed(123)
    a = model.sample(cond, num_steps=4)
    torch.manual_seed(123)
    b = model.sample(cond, num_steps=4)
    torch.testing.assert_close(a, b)
