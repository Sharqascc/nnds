"""Property tests for uvh_coco_fused_grid_pet pure helpers."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from src.analysis.grid_trajectory.uvh_coco_fused_grid_pet import (
    _angle_diff_deg,
    _bbox_overlap,
    _box_area,
    _line_side,
    _point_in_square,
    _segment_bbox,
)

FINITE = st.floats(min_value=-1e5, max_value=1e5, allow_nan=False, allow_infinity=False)
ANGLE = st.floats(min_value=-720.0, max_value=720.0, allow_nan=False, allow_infinity=False)
SMALL_POS = st.floats(min_value=0.001, max_value=1e3, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# _box_area
# ---------------------------------------------------------------------------


@given(box=st.tuples(FINITE, FINITE, FINITE, FINITE))
@settings(max_examples=80, deadline=None)
def test_box_area_non_negative(box):
    assert _box_area(box) >= 0


@given(box=st.tuples(FINITE, FINITE, FINITE, FINITE))
@settings(max_examples=50, deadline=None)
def test_box_area_symmetric_in_endpoints(box):
    x1, y1, x2, y2 = box
    v1 = _box_area((x1, y1, x2, y2))
    v2 = _box_area((x2, y2, x1, y1))
    assert abs(v1 - v2) < 1e-6 * (1.0 + abs(v1))


@given(
    x=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    y=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=30, deadline=None)
def test_box_area_degenerate_is_zero(x, y):
    assert _box_area((x, y, x, y)) == 0.0


# ---------------------------------------------------------------------------
# _angle_diff_deg
# ---------------------------------------------------------------------------


@given(a=ANGLE, b=ANGLE)
@settings(max_examples=100, deadline=None)
def test_angle_diff_in_range(a, b):
    d = _angle_diff_deg(a, b)
    assert d is not None
    assert 0.0 <= d <= 180.0


@given(a=ANGLE)
@settings(max_examples=30, deadline=None)
def test_angle_diff_self_is_zero(a):
    d = _angle_diff_deg(a, a)
    assert d is not None
    assert abs(d) < 1e-6


@given(a=ANGLE, b=ANGLE)
@settings(max_examples=100, deadline=None)
def test_angle_diff_symmetric(a, b):
    d1 = _angle_diff_deg(a, b)
    d2 = _angle_diff_deg(b, a)
    assert d1 is not None and d2 is not None
    assert abs(d1 - d2) < 1e-6


def test_angle_diff_none_inputs():
    assert _angle_diff_deg(None, 10.0) is None
    assert _angle_diff_deg(10.0, None) is None
    assert _angle_diff_deg(None, None) is None


# ---------------------------------------------------------------------------
# _point_in_square
# ---------------------------------------------------------------------------


@given(
    cx=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    cy=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    r=SMALL_POS,
)
@settings(max_examples=50, deadline=None)
def test_point_in_square_center_is_inside(cx, cy, r):
    assert _point_in_square(cx, cy, cx, cy, r) is True


@given(
    cx=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    cy=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    r=st.floats(min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50, deadline=None)
def test_point_in_square_very_far_is_outside(cx, cy, r):
    assert _point_in_square(cx + 1e4, cy, cx, cy, r) is False


# ---------------------------------------------------------------------------
# _segment_bbox
# ---------------------------------------------------------------------------


@given(p1=st.tuples(FINITE, FINITE), p2=st.tuples(FINITE, FINITE))
@settings(max_examples=80, deadline=None)
def test_segment_bbox_contains_endpoints(p1, p2):
    x1, y1, x2, y2 = _segment_bbox(p1, p2)
    for px, py in (p1, p2):
        assert x1 - 1e-6 <= px <= x2 + 1e-6
        assert y1 - 1e-6 <= py <= y2 + 1e-6


@given(p1=st.tuples(FINITE, FINITE), p2=st.tuples(FINITE, FINITE))
@settings(max_examples=50, deadline=None)
def test_segment_bbox_well_ordered(p1, p2):
    x1, y1, x2, y2 = _segment_bbox(p1, p2)
    assert x1 <= x2
    assert y1 <= y2


# ---------------------------------------------------------------------------
# _bbox_overlap
# ---------------------------------------------------------------------------


@given(
    b1=st.tuples(FINITE, FINITE, FINITE, FINITE),
    b2=st.tuples(FINITE, FINITE, FINITE, FINITE),
)
@settings(max_examples=80, deadline=None)
def test_bbox_overlap_non_negative(b1, b2):
    assert _bbox_overlap(b1, b2) >= 0


@given(
    b1=st.tuples(FINITE, FINITE, FINITE, FINITE),
    b2=st.tuples(FINITE, FINITE, FINITE, FINITE),
)
@settings(max_examples=80, deadline=None)
def test_bbox_overlap_symmetric(b1, b2):
    v12 = _bbox_overlap(b1, b2)
    v21 = _bbox_overlap(b2, b1)
    assert abs(v12 - v21) < 1e-6 * (1.0 + abs(v12))


@given(b=st.tuples(FINITE, FINITE, FINITE, FINITE))
@settings(max_examples=50, deadline=None)
def test_bbox_overlap_self_non_negative(b):
    assert _bbox_overlap(b, b) >= 0


@given(x_gap=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=30, deadline=None)
def test_bbox_overlap_far_apart_is_zero(x_gap):
    b1 = (0.0, 0.0, 10.0, 10.0)
    b2 = (10.0 + x_gap, 0.0, 20.0 + x_gap, 10.0)
    assert _bbox_overlap(b1, b2) == 0.0


# ---------------------------------------------------------------------------
# _line_side
# ---------------------------------------------------------------------------


@given(
    p=st.tuples(FINITE, FINITE),
    p1=st.tuples(FINITE, FINITE),
    p2=st.tuples(FINITE, FINITE),
)
@settings(max_examples=80, deadline=None)
def test_line_side_deterministic(p, p1, p2):
    s1 = _line_side(p, p1, p2)
    s2 = _line_side(p, p1, p2)
    assert s1 == s2


@given(p1=st.tuples(FINITE, FINITE), p2=st.tuples(FINITE, FINITE))
@settings(max_examples=30, deadline=None)
def test_line_side_midpoint_is_on_line(p1, p2):
    if p1 == p2:
        return
    mx = (p1[0] + p2[0]) / 2.0
    my = (p1[1] + p2[1]) / 2.0
    s = _line_side((mx, my), p1, p2)
    assert abs(float(s)) < 1e-6
