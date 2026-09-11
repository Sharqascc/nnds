import numpy as np
import pytest

from src.analysis.traj_error import (
    acceleration,
    acceleration_metrics,
    speed,
    trajectory_metrics,
    velocity,
    velocity_metrics,
)

FPS = 30.0


def _constant_velocity(vx: float, vy: float, n: int = 10) -> list[tuple[int, float, float]]:
    return [(i, vx * i / FPS, vy * i / FPS) for i in range(n)]


def test_velocity_of_constant_motion():
    traj = _constant_velocity(10.0, 0.0)
    v = velocity(traj, FPS)
    assert len(v) == len(traj)
    # all velocities should be 10 m/s in x, 0 in y
    for _, vx, vy in v:
        assert vx == pytest.approx(10.0, abs=1e-9)
        assert vy == pytest.approx(0.0, abs=1e-9)


def test_speed_of_3_4_5_triangle():
    # v = (3, 4) m/s -> speed 5 m/s
    traj = [(i, 3.0 * i / FPS, 4.0 * i / FPS) for i in range(10)]
    s = speed(traj, FPS)
    assert all(v == pytest.approx(5.0, abs=1e-9) for _, v in s)


def test_velocity_short_traj():
    assert velocity([(0, 0.0, 0.0)], FPS) == []


def test_acceleration_of_constant_accel():
    # x(t) = 0.5 * a * t^2, a = 2 m/s^2
    a = 2.0
    traj = [(i, 0.5 * a * (i / FPS) ** 2, 0.0) for i in range(10)]
    acc = acceleration(traj, FPS)
    # central-difference on constant-acceleration parabola is exact
    for _, ax, ay in acc:
        assert ax == pytest.approx(a, abs=1e-6)
        assert ay == pytest.approx(0.0, abs=1e-9)


def test_acceleration_short_traj():
    assert acceleration([(0, 0.0, 0.0), (1, 1.0, 1.0)], FPS) == []


def test_zero_error_on_identical_traj():
    traj = _constant_velocity(5.0, 3.0)
    vm = velocity_metrics(traj, traj, FPS)
    assert vm["mae"] == pytest.approx(0.0, abs=1e-9)
    assert vm["rmse"] == pytest.approx(0.0, abs=1e-9)


def test_velocity_error_known_offset():
    gt = _constant_velocity(10.0, 0.0)
    pred = _constant_velocity(12.0, 0.0)
    vm = velocity_metrics(pred, gt, FPS)
    assert vm["mae"] == pytest.approx(2.0, abs=1e-6)
    assert vm["rmse"] == pytest.approx(2.0, abs=1e-6)


def test_acceleration_error_known_offset():
    def _parabola(a: float) -> list[tuple[int, float, float]]:
        return [(i, 0.5 * a * (i / FPS) ** 2, 0.0) for i in range(10)]

    am = acceleration_metrics(_parabola(3.0), _parabola(2.0), FPS)
    assert am["mae"] == pytest.approx(1.0, abs=1e-6)


def test_empty_trajectory_metrics():
    m = trajectory_metrics([], [], FPS)
    assert m["velocity"] == {"mae": 0.0, "rmse": 0.0, "n": 0}
    assert m["acceleration"] == {"mae": 0.0, "rmse": 0.0, "n": 0}


def test_fps_invalid_raises():
    with pytest.raises(ValueError):
        velocity(_constant_velocity(1.0, 0.0), 0.0)
    with pytest.raises(ValueError):
        acceleration(_constant_velocity(1.0, 0.0), -30.0)


def test_common_frames_only():
    gt = [(i, 5.0 * i / FPS, 0.0) for i in range(10)]
    pred = [(i, 5.0 * i / FPS, 0.0) for i in range(5, 15)]  # only frames 5-9 overlap
    m = velocity_metrics(pred, gt, FPS)
    # on common frames 5-9, values match -> 0 error
    assert m["mae"] == pytest.approx(0.0, abs=1e-9)
    assert m["n"] == 5
