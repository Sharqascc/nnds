import pytest

from src.analysis.detection_metrics import (
    Detection,
    GroundTruth,
    ap_at_iou,
    ap_by_size,
    average_precision_voc,
    box_area,
    iou,
    map_at_iou_range,
    per_class_recall,
)


def _det(frame, box, cls="car", conf=0.9):
    return Detection(frame, box, cls, conf)


def _gt(frame, box, cls="car"):
    return GroundTruth(frame, box, cls)


def test_box_area_positive():
    assert box_area((0, 0, 10, 20)) == pytest.approx(200.0)


def test_box_area_degenerate():
    assert box_area((10, 10, 5, 20)) == 0.0


def test_iou_identical_boxes():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_no_overlap():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_half_overlap():
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1.0 / 3.0)


def test_iou_zero_area():
    assert iou((0, 0, 0, 0), (0, 0, 10, 10)) == 0.0


def test_ap_voc_trivial():
    import numpy as np

    recalls = np.array([0.5, 1.0])
    precisions = np.array([1.0, 1.0])
    assert average_precision_voc(recalls, precisions) == pytest.approx(1.0)


def test_perfect_predictions_ap_1():
    dets = [_det(0, (0, 0, 10, 10)), _det(0, (20, 20, 30, 30))]
    gts = [_gt(0, (0, 0, 10, 10)), _gt(0, (20, 20, 30, 30))]
    assert ap_at_iou(dets, gts, "car", 0.5) == pytest.approx(1.0)


def test_no_detections_ap_0():
    gts = [_gt(0, (0, 0, 10, 10))]
    assert ap_at_iou([], gts, "car", 0.5) == 0.0


def test_no_gts_ap_0():
    dets = [_det(0, (0, 0, 10, 10))]
    assert ap_at_iou(dets, [], "car", 0.5) == 0.0


def test_wrong_class_does_not_match():
    dets = [_det(0, (0, 0, 10, 10), cls="pedestrian")]
    gts = [_gt(0, (0, 0, 10, 10), cls="car")]
    assert ap_at_iou(dets, gts, "car", 0.5) == 0.0


def test_different_frames_do_not_match():
    dets = [_det(0, (0, 0, 10, 10))]
    gts = [_gt(1, (0, 0, 10, 10))]
    assert ap_at_iou(dets, gts, "car", 0.5) == 0.0


def test_map_range_perfect():
    dets = [_det(0, (0, 0, 10, 10))]
    gts = [_gt(0, (0, 0, 10, 10))]
    m = map_at_iou_range(dets, gts)
    assert m["mAP50"] == pytest.approx(1.0)
    assert m["mAP75"] == pytest.approx(1.0)
    assert m["mAP50:95"] == pytest.approx(1.0)


def test_map_range_empty():
    m = map_at_iou_range([], [])
    assert m == {"mAP50": 0.0, "mAP75": 0.0, "mAP50:95": 0.0}


def test_per_class_recall_perfect():
    dets = [_det(0, (0, 0, 10, 10), cls="car")]
    gts = [_gt(0, (0, 0, 10, 10), cls="car")]
    r = per_class_recall(dets, gts)
    assert r == {"car": pytest.approx(1.0)}


def test_ap_by_size_buckets():
    assert ap_by_size(
        [_det(0, (0, 0, 10, 10))],
        [_gt(0, (0, 0, 10, 10))],
    )["APs"] == pytest.approx(1.0)
    assert ap_by_size(
        [_det(0, (0, 0, 50, 50))],
        [_gt(0, (0, 0, 50, 50))],
    )["APm"] == pytest.approx(1.0)
    assert ap_by_size(
        [_det(0, (0, 0, 200, 200))],
        [_gt(0, (0, 0, 200, 200))],
    )["APl"] == pytest.approx(1.0)


def test_ap_by_size_mixed_set_bounded():
    dets = [
        _det(0, (0, 0, 10, 10)),
        _det(0, (0, 0, 50, 50)),
        _det(0, (0, 0, 200, 200)),
    ]
    gts = [
        _gt(0, (0, 0, 10, 10)),
        _gt(0, (0, 0, 50, 50)),
        _gt(0, (0, 0, 200, 200)),
    ]
    ap = ap_by_size(dets, gts)
    assert ap["APs"] == pytest.approx(1.0)
    assert 0.0 <= ap["APm"] <= 1.0
    assert 0.0 <= ap["APl"] <= 1.0


def test_ap_by_size_missing_bucket_zero():
    dets = [_det(0, (0, 0, 10, 10))]
    gts = [_gt(0, (0, 0, 10, 10))]
    ap = ap_by_size(dets, gts)
    assert ap["APs"] == pytest.approx(1.0)
    assert ap["APm"] == 0.0
    assert ap["APl"] == 0.0
