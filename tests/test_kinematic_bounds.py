"""Physical plausibility bounds on kinematic computations.

The pipeline computes velocity and acceleration from tracked positions. A
bug in fps handling, the time-delta formula, or the coordinate
transformation can produce values that are mathematically consistent but
physically impossible for road vehicles. These tests use synthetic
trajectories with known kinematics and assert the computed values land
within plausible bounds.

Different from `test_scientific_invariants.py`, which checks statistical
properties of pre-computed output CSVs (and skips in CI when those aren't
present). Here the computations are exercised directly on controlled
inputs, so the tests run everywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.traj_error import acceleration, speed, velocity

# Typical urban vehicle limits, used to bound what a tracker should ever
# produce on smooth input. Hard braking on dry asphalt is ~-8 m/s^2;
# emergency is ~-10. 60 m/s is ~215 km/h, above any road vehicle.
MAX_PLAUSIBLE_ACCEL_MS2 = 10.0
MAX_PLAUSIBLE_SPEED_MS = 60.0


def _constant_velocity_traj(
    speed_ms: float,
    fps: float,
    n_frames: int,
    vx_unit: float = 1.0,
    vy_unit: float = 0.0,
):
    """Synthetic trajectory moving at `speed_ms` in direction (vx_unit, vy_unit),
    sampled at `fps`. Coordinates are in world meters."""
    dt = 1.0 / fps
    norm = (vx_unit**2 + vy_unit**2) ** 0.5
    vx = speed_ms * vx_unit / norm
    vy = speed_ms * vy_unit / norm
    return [(f, vx * f * dt, vy * f * dt) for f in range(n_frames)]


def test_velocity_recovers_known_car_speed():
    """A car at exactly 10 m/s produces exactly 10 m/s from velocity()."""
    traj = _constant_velocity_traj(speed_ms=10.0, fps=30.0, n_frames=5)
    v = velocity(traj, fps=30.0)
    assert len(v) > 0
    for _, vx, vy in v:
        assert float(np.hypot(vx, vy)) == pytest.approx(10.0, rel=1e-9)


def test_fps_consistency_across_sampling_rates():
    """The same physical motion sampled at different frame rates must
    recover the same velocity. If fps is used incorrectly, one of these
    fails - this is the strongest guard against a systematic unit bug."""
    traj_30 = _constant_velocity_traj(speed_ms=15.0, fps=30.0, n_frames=5)
    traj_60 = _constant_velocity_traj(speed_ms=15.0, fps=60.0, n_frames=9)

    v30 = velocity(traj_30, fps=30.0)
    v60 = velocity(traj_60, fps=60.0)

    for _, vx, vy in v30:
        assert float(np.hypot(vx, vy)) == pytest.approx(15.0, rel=1e-9)
    for _, vx, vy in v60:
        assert float(np.hypot(vx, vy)) == pytest.approx(15.0, rel=1e-9)


def test_acceleration_of_constant_velocity_is_physically_zero():
    """At any sampling rate, constant velocity produces zero acceleration."""
    for fps in (10.0, 30.0, 60.0, 120.0):
        traj = _constant_velocity_traj(speed_ms=20.0, fps=fps, n_frames=6)
        a = acceleration(traj, fps=fps)
        assert len(a) > 0
        for _, ax, ay in a:
            assert abs(ax) < 1e-9
            assert abs(ay) < 1e-9


def test_smooth_acceleration_stays_within_car_limits():
    """A trajectory with 3 m/s^2 acceleration produces ~3 m/s^2, not
    amplified by float noise in the central difference."""
    a_true = 3.0
    fps = 30.0
    traj = [(f, 0.5 * a_true * (f / fps) ** 2, 0.0) for f in range(10)]
    a = acceleration(traj, fps=fps)
    assert len(a) > 0
    for _, ax, _ in a:
        assert ax == pytest.approx(a_true, rel=1e-6)


def test_velocity_does_not_exceed_plausible_speed_on_smooth_input():
    """A car at 20 m/s stays below the plausible bound at every frame.
    Guards against systematic fps/unit errors that inflate magnitudes."""
    traj = _constant_velocity_traj(speed_ms=20.0, fps=30.0, n_frames=20)
    v = velocity(traj, fps=30.0)
    for _, vx, vy in v:
        s = float(np.hypot(vx, vy))
        assert s <= MAX_PLAUSIBLE_SPEED_MS, (
            f"Computed speed {s:.2f} m/s exceeds plausible bound "
            f"{MAX_PLAUSIBLE_SPEED_MS} on smooth input"
        )


def test_acceleration_does_not_exceed_plausible_bound_on_smooth_input():
    """A smooth trajectory never produces accelerations above the physical
    limit. Guards against float blow-up in central differences."""
    a_true = 5.0
    fps = 30.0
    traj = [(f, 0.5 * a_true * (f / fps) ** 2, 0.0) for f in range(15)]
    a = acceleration(traj, fps=fps)
    for _, ax, ay in a:
        mag = float(np.hypot(ax, ay))
        assert mag <= MAX_PLAUSIBLE_ACCEL_MS2, (
            f"Computed acceleration {mag:.2f} m/s^2 exceeds plausible "
            f"bound {MAX_PLAUSIBLE_ACCEL_MS2} on smooth input"
        )


def test_small_position_noise_does_not_amplify_in_velocity():
    """Realistic detection noise (1 cm sigma) at 30 fps must not become a
    huge velocity through the central-difference formula."""
    rng = np.random.default_rng(seed=0)
    base = _constant_velocity_traj(speed_ms=10.0, fps=30.0, n_frames=30)
    noisy = [
        (f, x + float(rng.normal(0, 0.01)), y + float(rng.normal(0, 0.01))) for f, x, y in base
    ]
    v_noisy = velocity(noisy, fps=30.0)
    # Noise sigma in velocity: 0.01 m * 30 fps / sqrt(2) ~ 0.2 m/s.
    # Conservative bound: nothing should reach 50 m/s.
    for _, vx, vy in v_noisy:
        s = float(np.hypot(vx, vy))
        assert s < 50.0, f"noise amplified to {s:.2f} m/s"


def test_speed_function_matches_velocity_magnitude():
    """speed() must equal |velocity()| at every frame."""
    traj = _constant_velocity_traj(speed_ms=12.0, fps=30.0, n_frames=8)
    v = velocity(traj, fps=30.0)
    s = speed(traj, fps=30.0)
    v_map = {f: float(np.hypot(vx, vy)) for f, vx, vy in v}
    for f, sp in s:
        assert f in v_map
        assert sp == pytest.approx(v_map[f], rel=1e-12)
