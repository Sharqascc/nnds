"""Property tests for paper.analysis.tracklet_stitching."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from paper.analysis import tracklet_stitching as ts


@given(
    a=st.floats(-720.0, 720.0, allow_nan=False, allow_infinity=False),
    b=st.floats(-720.0, 720.0, allow_nan=False, allow_infinity=False),
)
def test_angle_diff_deg_in_valid_range(a, b):
    d = ts._angle_diff_deg(a, b)
    assert 0.0 <= d <= 180.0


def test_angle_diff_deg_identical_is_zero():
    assert ts._angle_diff_deg(45.0, 45.0) == pytest.approx(0.0)


def test_angle_diff_deg_opposite_is_180():
    assert ts._angle_diff_deg(0.0, 180.0) == pytest.approx(180.0)


def test_endpoint_returns_dict():
    td = {0: (0.0, 0.0, 0.0, 0.0),
          1: (0.0, 0.0, 1.0, 0.0),
          2: (0.0, 0.0, 2.0, 0.0)}
    e = ts._endpoint(td, [0, 1, 2], at_end=False)
    assert "xy" in e
    assert "speed" in e
    assert "heading" in e


def test_merge_score_returns_none_on_large_gap():
    a = {"start_frame": 0, "end_frame": 10,
         "start_xy": (0.0, 0.0), "end_xy": (0.0, 0.0),
         "start_speed": 1.0, "end_speed": 1.0,
         "start_heading": 0.0, "end_heading": 0.0}
    b = {"start_frame": 100, "end_frame": 110,
         "start_xy": (0.0, 0.0), "end_xy": (0.0, 0.0),
         "start_speed": 1.0, "end_speed": 1.0,
         "start_heading": 0.0, "end_heading": 0.0}
    assert ts._merge_score(a, b) is None


def test_merge_score_returns_dict_on_close_pair():
    a = {"start_frame": 0, "end_frame": 10,
         "start_xy": (0.0, 0.0), "end_xy": (1.0, 0.0),
         "start_speed": 1.0, "end_speed": 1.0,
         "start_heading": 0.0, "end_heading": 0.0}
    b = {"start_frame": 13, "end_frame": 23,
         "start_xy": (1.0, 0.0), "end_xy": (2.0, 0.0),
         "start_speed": 1.0, "end_speed": 1.0,
         "start_heading": 0.0, "end_heading": 0.0}
    m = ts._merge_score(a, b)
    assert m is not None
    assert 0.0 <= m["score"] <= 1.0
