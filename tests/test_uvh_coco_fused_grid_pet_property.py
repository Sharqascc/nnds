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
BOX = st.tuples(FINITE, FINITE, FINITE, FINITE)


# ---------------------------------------------------------------------------
# _box_area — clamped formula: max(0, x2-x1) * max(0, y2-y1)
# ---------------------------------------------------------------------------


@given(box=BOX)
@settings(max_examples=80, deadline=None)
def test_box_area_matches_clamped_formula(box):
    x1, y1, x2, y2 = box
    expected = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    got = _box_area(box)
    assert abs(got - expected) < 1e-6 * (1.0 + abs(expected))


@given(box=BOX)
@settings(max_examples=80, deadline=None)
def test_box_area_non_negative(box):
    assert _box_area(box) >= 0.0


@given(x=FINITE, y=FINITE)
@settings(max_examples=50, deadline=None)
def test_box_area_degenerate_is_zero(x, y):
    assert _box_area((x, y, x, y)) == 0.0


@given(
    x1=FINITE,
    y1=FINITE,
    w=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    h=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=80, deadline=None)
def test_box_area_positive_on_ordered_box(x1, y1, w, h):
    got = _box_area((x1, y1, x1 + w, y1 + h))
    assert abs(got - w * h) < 1e-6 * (1.0 + abs(w * h))


# ---------------------------------------------------------------------------
# _angle_diff_deg — returns None on None, else float in [0, 180], symmetric
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
    assert abs(d1 - d2) < 1e-9


def test_angle_diff_none_inputs():
    assert _angle_diff_deg(None, 10.0) is None
    assert _angle_diff_deg(10.0, None) is None
    assert _angle_diff_deg(None, None) is None


# ---------------------------------------------------------------------------
# _point_in_square
# ---------------------------------------------------------------------------


@given(cx=FINITE, cy=FINITE, r=SMALL_POS)
@settings(max_examples=50, deadline=None)
def test_point_in_square_center_is_inside(cx, cy, r):
    assert _point_in_square(cx, cy, cx, cy, r) is True


@given(
    cx=FINITE,
    cy=FINITE,
    r=st.floats(min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50, deadline=None)
def test_point_in_square_far_away_is_outside(cx, cy, r):
    assert _point_in_square(cx + 1e4, cy, cx, cy, r) is False


# ---------------------------------------------------------------------------
# _segment_bbox — returns (min_x, min_y, max_x, max_y), well ordered
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
# _bbox_overlap — returns bool
# ---------------------------------------------------------------------------


@given(b1=BOX, b2=BOX)
@settings(max_examples=80, deadline=None)
def test_bbox_overlap_returns_bool(b1, b2):
    assert isinstance(_bbox_overlap(b1, b2), bool)


@given(b1=BOX, b2=BOX)
@settings(max_examples=80, deadline=None)
def test_bbox_overlap_symmetric(b1, b2):
    assert _bbox_overlap(b1, b2) == _bbox_overlap(b2, b1)


@given(b=BOX)
@settings(max_examples=50, deadline=None)
def test_bbox_overlap_self_true_when_non_degenerate(b):
    x1, y1, x2, y2 = b
    if x2 > x1 and y2 > y1:
        assert _bbox_overlap(b, b) is True


@given(x_gap=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=30, deadline=None)
def test_bbox_overlap_far_apart_is_false(x_gap):
    b1 = (0.0, 0.0, 10.0, 10.0)
    b2 = (10.0 + x_gap, 0.0, 20.0 + x_gap, 10.0)
    assert _bbox_overlap(b1, b2) is False


def test_bbox_overlap_pad_extends_reach():
    b1 = (0.0, 0.0, 10.0, 10.0)
    b2 = (12.0, 0.0, 20.0, 10.0)
    assert _bbox_overlap(b1, b2) is False
    assert _bbox_overlap(b1, b2, pad=3.0) is True


# ---------------------------------------------------------------------------
# _line_side — signed scalar
# ---------------------------------------------------------------------------


@given(p=st.tuples(FINITE, FINITE), p1=st.tuples(FINITE, FINITE), p2=st.tuples(FINITE, FINITE))
@settings(max_examples=80, deadline=None)
def test_line_side_deterministic(p, p1, p2):
    assert _line_side(p, p1, p2) == _line_side(p, p1, p2)


@given(p1=st.tuples(FINITE, FINITE), p2=st.tuples(FINITE, FINITE))
@settings(max_examples=30, deadline=None)
def test_line_side_midpoint_is_on_line(p1, p2):
    if p1 == p2:
        return
    mx = (p1[0] + p2[0]) / 2.0
    my = (p1[1] + p2[1]) / 2.0
    s = _line_side((mx, my), p1, p2)
    assert abs(float(s)) < 1e-6


@given(p1=st.tuples(FINITE, FINITE), p2=st.tuples(FINITE, FINITE))
@settings(max_examples=50, deadline=None)
def test_line_side_endpoints_are_on_line(p1, p2):
    assert abs(float(_line_side(p1, p1, p2))) < 1e-6
    assert abs(float(_line_side(p2, p1, p2))) < 1e-6
