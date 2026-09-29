import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.analysis.detection_metrics import (
    Detection,
    GroundTruth,
    ap_at_iou,
    iou,
    map_at_iou_range,
    per_class_recall,
)

box_st = st.tuples(
    st.floats(0, 100, allow_nan=False, allow_infinity=False),
    st.floats(0, 100, allow_nan=False, allow_infinity=False),
    st.floats(0, 100, allow_nan=False, allow_infinity=False),
    st.floats(0, 100, allow_nan=False, allow_infinity=False),
).map(
    lambda b: (
        min(b[0], b[2]),
        min(b[1], b[3]),
        max(b[0], b[2]) + 1.0,
        max(b[1], b[3]) + 1.0,
    )
)

items_st = (
    st.lists(st.tuples(st.integers(0, 5), box_st), min_size=1, max_size=15)
    .map(lambda xs: list(dict.fromkeys(xs)))
    .filter(lambda xs: len(xs) >= 1)
)


def _dets(items):
    return [Detection(f, b, "car", 0.9) for f, b in items]


def _gts(items):
    return [GroundTruth(f, b, "car") for f, b in items]


@given(box_st, box_st)
@settings(max_examples=50)
def test_iou_in_unit_interval(b1, b2):
    v = iou(b1, b2)
    assert 0.0 <= v <= 1.0


@given(box_st, box_st)
@settings(max_examples=50)
def test_iou_symmetric(b1, b2):
    assert iou(b1, b2) == pytest.approx(iou(b2, b1))


@given(box_st)
@settings(max_examples=50)
def test_iou_self_is_one_when_positive_area(b):
    if (b[2] - b[0]) > 0 and (b[3] - b[1]) > 0:
        assert iou(b, b) == pytest.approx(1.0)


@given(items_st)
@settings(max_examples=50)
def test_per_class_recall_in_unit(items):
    r = per_class_recall(_dets(items), _gts(items), 0.5)
    assert 0.0 <= r.get("car", 0.0) <= 1.0


@given(items_st)
@settings(max_examples=50)
def test_ap_monotone_decreasing_in_iou(items):
    d, g = _dets(items), _gts(items)
    assert ap_at_iou(d, g, "car", 0.5) + 1e-9 >= ap_at_iou(d, g, "car", 0.75)


@given(items_st)
@settings(max_examples=50)
def test_map50_95_bounded_by_extremes(items):
    d, g = _dets(items), _gts(items)
    m = map_at_iou_range(d, g)
    ap_95 = ap_at_iou(d, g, "car", 0.95)
    assert ap_95 - 1e-9 <= m["mAP50:95"] <= m["mAP50"] + 1e-9


@given(items_st)
@settings(max_examples=50)
def test_perfect_predictions_give_ap_1(items):
    assert ap_at_iou(_dets(items), _gts(items), "car", 0.5) == pytest.approx(1.0)


@given(items_st)
@settings(max_examples=50)
def test_duplicate_detection_does_not_increase_ap(items):
    d, g = _dets(items), _gts(items)
    ap0 = ap_at_iou(d, g, "car", 0.5)
    d2 = d + [Detection(f, b, "car", 0.1) for f, b in items]
    ap1 = ap_at_iou(d2, g, "car", 0.5)
    assert ap1 <= ap0 + 1e-9


@given(items_st)
@settings(max_examples=30)
def test_empty_detections_ap_zero(items):
    assert ap_at_iou([], _gts(items), "car", 0.5) == 0.0
