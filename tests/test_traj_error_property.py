import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.analysis.traj_error import (
    acceleration_metrics,
    speed,
    velocity_metrics,
)

FPS = 30.0


def _linear_traj(vx: float, vy: float, n: int, fps: float = FPS):
    return [(i, vx * i / fps, vy * i / fps) for i in range(n)]


def _parabola_traj(ax: float, ay: float, n: int, fps: float = FPS):
    return [(i, 0.5 * ax * (i / fps) ** 2, 0.5 * ay * (i / fps) ** 2) for i in range(n)]


@given(
    st.floats(-30, 30, allow_nan=False, allow_infinity=False),
    st.floats(-30, 30, allow_nan=False, allow_infinity=False),
    st.integers(3, 20),
)
@settings(max_examples=50)
def test_constant_velocity_zero_error(vx, vy, n):
    traj = _linear_traj(vx, vy, n)
    m = velocity_metrics(traj, traj, FPS)
    assert m["mae"] == pytest.approx(0.0, abs=1e-6)


@given(
    st.floats(-5, 5, allow_nan=False, allow_infinity=False),
    st.floats(-5, 5, allow_nan=False, allow_infinity=False),
    st.integers(4, 20),
)
@settings(max_examples=50)
def test_constant_acceleration_zero_error(ax, ay, n):
    traj = _parabola_traj(ax, ay, n)
    m = acceleration_metrics(traj, traj, FPS)
    assert m["mae"] == pytest.approx(0.0, abs=1e-5)


@given(
    st.floats(-20, 20, allow_nan=False, allow_infinity=False),
    st.floats(-20, 20, allow_nan=False, allow_infinity=False),
    st.integers(3, 15),
)
@settings(max_examples=50)
def test_speed_non_negative(vx, vy, n):
    traj = _linear_traj(vx, vy, n)
    for _, s in speed(traj, FPS):
        assert s >= 0.0


@given(
    st.floats(-20, 20, allow_nan=False, allow_infinity=False),
    st.floats(-20, 20, allow_nan=False, allow_infinity=False),
    st.floats(-20, 20, allow_nan=False, allow_infinity=False),
    st.floats(-20, 20, allow_nan=False, allow_infinity=False),
    st.integers(3, 15),
)
@settings(max_examples=50)
def test_velocity_error_triangle_inequality(vx1, vy1, vx2, vy2, n):
    a = _linear_traj(vx1, vy1, n)
    b = _linear_traj(vx2, vy2, n)
    # error(a, b) == error(b, a) (absolute error is symmetric)
    m1 = velocity_metrics(a, b, FPS)
    m2 = velocity_metrics(b, a, FPS)
    assert m1["mae"] == pytest.approx(m2["mae"], abs=1e-9)


@given(st.integers(3, 10))
@settings(max_examples=30)
def test_zero_length_both_empty(n):
    m = velocity_metrics([], [], FPS)
    assert m["n"] == 0
    assert m["mae"] == 0.0


@given(
    st.floats(0.1, 20.0, allow_nan=False, allow_infinity=False),
    st.integers(3, 15),
)
@settings(max_examples=50)
def test_velocity_scales_with_position_scale(scale, n):
    base = _linear_traj(10.0, 0.0, n)
    scaled = [(f, x * scale, y * scale) for f, x, y in base]
    s_base = speed(base, FPS)
    s_scaled = speed(scaled, FPS)
    for (_, v0), (_, v1) in zip(s_base, s_scaled, strict=False):
        assert v1 == pytest.approx(v0 * scale, abs=1e-6)
