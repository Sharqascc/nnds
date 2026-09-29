"""Tests for LinearNoiseScheduler in src/diffusion/complete_ddpm.py.

CPU-only. No model loads, no GPU, no network. Verifies shape contracts,
the monotonicity of alpha_cumprod, and determinism of both add_noise
and sample_prev_timestep.
"""

from __future__ import annotations

import pytest
import torch

from src.diffusion.complete_ddpm import LinearNoiseScheduler


@pytest.fixture
def scheduler():
    return LinearNoiseScheduler(num_timesteps=50, beta_start=1e-4, beta_end=0.02)


def test_init_shapes(scheduler):
    n = scheduler.num_timesteps
    for name in (
        "betas",
        "alphas",
        "alpha_cumprod",
        "sqrt_alpha_cumprod",
        "sqrt_one_minus_alpha_cumprod",
    ):
        t = getattr(scheduler, name)
        assert t.shape == (n,), f"{name} has wrong shape"


def test_betas_within_bounds(scheduler):
    assert (scheduler.betas > 0).all()
    assert (scheduler.betas <= 0.02 + 1e-6).all()


def test_alpha_cumprod_monotonic_decreasing(scheduler):
    diffs = scheduler.alpha_cumprod[1:] - scheduler.alpha_cumprod[:-1]
    assert (diffs <= 1e-9).all()


def test_alpha_cumprod_in_unit_interval(scheduler):
    assert (scheduler.alpha_cumprod > 0).all()
    assert (scheduler.alpha_cumprod <= 1.0 + 1e-6).all()


def test_add_noise_shape_preserved(scheduler):
    x = torch.randn(4, 8, 1, 2)
    noise = torch.randn_like(x)
    t = torch.tensor([0, 5, 10, 20])
    out = scheduler.add_noise(x, t, noise)
    assert out.shape == x.shape


def test_add_noise_at_t0_with_zero_noise_is_signal(scheduler):
    x = torch.ones(1, 4, 1, 2)
    noise = torch.zeros_like(x)
    t = torch.tensor([0])
    out = scheduler.add_noise(x, t, noise)
    expected = float(scheduler.sqrt_alpha_cumprod[0]) * x
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


def test_add_noise_at_max_t_is_dominated_by_noise(scheduler):
    x = torch.zeros(1, 4, 1, 2)
    noise = torch.ones_like(x)
    t = torch.tensor([scheduler.num_timesteps - 1])
    out = scheduler.add_noise(x, t, noise)
    expected = float(scheduler.sqrt_one_minus_alpha_cumprod[-1]) * noise
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)
    # sqrt(1 - alpha_bar) close to 1 at the last timestep
    assert float(scheduler.sqrt_one_minus_alpha_cumprod[-1]) > 0.5


def test_add_noise_deterministic(scheduler):
    x = torch.randn(2, 4, 1, 2)
    noise = torch.randn_like(x)
    t = torch.tensor([3, 4])
    torch.testing.assert_close(
        scheduler.add_noise(x, t, noise),
        scheduler.add_noise(x, t, noise),
    )


def test_sample_prev_timestep_shape(scheduler):
    x = torch.randn(1, 4, 1, 2)
    noise_pred = torch.randn_like(x)
    t = torch.tensor([10])
    out = scheduler.sample_prev_timestep(x, t, noise_pred)
    assert out.shape == x.shape


def test_sample_prev_timestep_at_t0_adds_no_noise(scheduler):
    # the code path: `if t[0] > 0: x_prev = x_prev + sigma * randn_like(x)`
    # is skipped when t[0] == 0. Two calls at t=0 must be identical.
    x = torch.randn(1, 4, 1, 2)
    noise_pred = torch.randn_like(x)
    t = torch.tensor([0])
    a = scheduler.sample_prev_timestep(x, t, noise_pred)
    b = scheduler.sample_prev_timestep(x, t, noise_pred)
    torch.testing.assert_close(a, b)


def test_sample_prev_timestep_deterministic_with_seed(scheduler):
    x = torch.randn(1, 4, 1, 2)
    noise_pred = torch.randn_like(x)
    t = torch.tensor([10])
    torch.manual_seed(42)
    a = scheduler.sample_prev_timestep(x, t, noise_pred)
    torch.manual_seed(42)
    b = scheduler.sample_prev_timestep(x, t, noise_pred)
    torch.testing.assert_close(a, b)


def test_full_denoise_loop_runs(scheduler):
    # Simulate 10 steps of reverse sampling with a zero-noise oracle.
    # Just verifies: no crash, finite output, shape preserved.
    x = torch.randn(1, 4, 1, 2)
    for step in torch.linspace(scheduler.num_timesteps - 1, 0, 10).long():
        t = torch.tensor([int(step)])
        noise_pred = torch.zeros_like(x)
        x = scheduler.sample_prev_timestep(x, t, noise_pred)
    assert x.shape == (1, 4, 1, 2)
    assert torch.isfinite(x).all()
