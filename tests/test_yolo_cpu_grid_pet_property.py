"""Property tests for pure geometry helpers in yolo_cpu_grid_pet."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.analysis.grid_trajectory import yolo_cpu_grid_pet as yp

# -------------- _segment_intersection --------------


def test_intersecting_diagonals_returns_center():
    r = yp._segment_intersection((0.0, 0.0), (2.0, 2.0), (0.0, 2.0), (2.0, 0.0))
    assert r is not None
    x, y = r
    assert abs(x - 1.0) < 1e-6
    assert abs(y - 1.0) < 1e-6


def test_parallel_segments_return_none():
    r = yp._segment_intersection((0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 2.0))
    assert r is None


def test_disjoint_collinear_segments_return_none():
    r = yp._segment_intersection((0.0, 0.0), (1.0, 1.0), (5.0, 5.0), (6.0, 6.0))
    assert r is None


def test_non_crossing_segments_return_none():
    r = yp._segment_intersection((0.0, 0.0), (1.0, 0.0), (2.0, 2.0), (3.0, 3.0))
    assert r is None


@given(
    x1=st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
    y1=st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
    x2=st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
    y2=st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
)
def test_segment_intersection_returns_tuple_or_none(x1, y1, x2, y2):
    r = yp._segment_intersection((0.0, 0.0), (1.0, 1.0), (x1, y1), (x2, y2))
    assert r is None or (isinstance(r, tuple) and len(r) == 2)


# -------------- _point_in_square --------------


@given(
    cx=st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
    cy=st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
    hs=st.floats(0.1, 100.0, allow_nan=False, allow_infinity=False),
)
def test_point_at_center_is_inside(cx, cy, hs):
    assert yp._point_in_square(cx, cy, cx, cy, hs)


@given(
    cx=st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
    cy=st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
    hs=st.floats(0.1, 100.0, allow_nan=False, allow_infinity=False),
)
def test_point_on_boundary_is_inside(cx, cy, hs):
    assert yp._point_in_square(cx - hs, cy, cx, cy, hs)
    assert yp._point_in_square(cx + hs, cy, cx, cy, hs)
    assert yp._point_in_square(cx, cy - hs, cx, cy, hs)
    assert yp._point_in_square(cx, cy + hs, cx, cy, hs)


@given(
    cx=st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
    cy=st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
    hs=st.floats(0.1, 100.0, allow_nan=False, allow_infinity=False),
)
def test_point_far_outside_is_outside(cx, cy, hs):
    assert not yp._point_in_square(cx + hs + 1.0, cy, cx, cy, hs)
    assert not yp._point_in_square(cx, cy + hs + 1.0, cx, cy, hs)
    assert not yp._point_in_square(cx - hs - 1.0, cy, cx, cy, hs)


# -------------- _entry_exit_frames --------------


def _pt(frame, x=0.0, y=0.0):
    return yp.TrackPoint(frame=frame, x=x, y=y, cls_id=0, cls_name="person", conf=0.9)


def test_entry_exit_no_points_inside_returns_none():
    pts = [_pt(0, 100.0, 100.0), _pt(1, 200.0, 200.0)]
    assert yp._entry_exit_frames(pts, 0.0, 0.0, 1.0) is None


def test_entry_exit_returns_min_max_frames():
    pts = [_pt(5), _pt(10, 0.5, 0.5), _pt(15)]
    assert yp._entry_exit_frames(pts, 0.0, 0.0, 1.0) == (5, 15)


def test_entry_exit_min_lte_max():
    pts = [_pt(f) for f in [3, 7, 5, 1]]
    entry, exit_ = yp._entry_exit_frames(pts, 0.0, 0.0, 1.0)
    assert entry <= exit_


# -------------- _pair_conflict_point --------------


def test_pair_conflict_none_when_no_intersection():
    a = [_pt(0, 0.0, 0.0), _pt(1, 1.0, 1.0)]
    b = [_pt(0, 5.0, 5.0), _pt(1, 6.0, 6.0)]
    assert yp._pair_conflict_point(a, b) is None


def test_pair_conflict_finds_crossing():
    a = [_pt(0, 0.0, 0.0), _pt(1, 2.0, 2.0)]
    b = [_pt(0, 0.0, 2.0), _pt(1, 2.0, 0.0)]
    r = yp._pair_conflict_point(a, b)
    assert r is not None
    assert abs(r[0] - 1.0) < 1e-6
    assert abs(r[1] - 1.0) < 1e-6
