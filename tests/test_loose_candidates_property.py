"""Property tests for paper.analysis.loose_candidates."""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from paper.analysis import loose_candidates as lc


def test_arr_returns_2d_array():
    td = {0: (1.0, 2.0), 1: (3.0, 4.0), 2: (5.0, 6.0)}
    arr = lc._arr(td)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3, 2)


@given(
    x=st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
    y=st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
)
def test_heading_deg_in_valid_range(x, y):
    xy = np.array([[0.0, 0.0], [x, y]])
    h = lc._heading_deg(xy)
    assert -180.0 <= h <= 180.0


def test_speed_mps_returns_finite_number():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    frames = [0, 30, 60]
    s = lc._speed_mps(xy, frames)
    assert np.isfinite(s)
