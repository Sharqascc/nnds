"""Regression: duplicate detections must be collapsed before tracking."""

from src.pipeline.custom_tracker import CustomTracker, Detection


def _det(cx, cy, cls_id=0, cls_name="car", w=40, h=40, conf=0.9, frame=0):
    return Detection(
        frame=frame,
        x1=cx - w / 2,
        y1=cy - h / 2,
        x2=cx + w / 2,
        y2=cy + h / 2,
        cx=cx,
        cy=cy,
        cls_id=cls_id,
        cls_name=cls_name,
        conf=conf,
        source="test",
    )


def test_dedup_collapses_same_class_overlap():
    trk = CustomTracker(max_age=10, min_hits=1, iou_threshold=0.2)
    # Two boxes on one car, 95% overlap, different confidences
    a = _det(100, 100, conf=0.9, frame=0)
    b = _det(101, 101, conf=0.7, frame=0)
    kept_idx, kept_dets = trk._dedup_detections([a, b], iou_thr=0.9)
    assert len(kept_dets) == 1
    assert kept_dets[0].conf == 0.9  # higher confidence survives
    assert kept_idx == [0]  # original index of `a`


def test_dedup_keeps_different_classes():
    trk = CustomTracker(max_age=10, min_hits=1, iou_threshold=0.2)
    a = _det(100, 100, cls_id=0, cls_name="car", frame=0)
    b = _det(100, 100, cls_id=1, cls_name="pedestrian", frame=0)
    kept_idx, kept_dets = trk._dedup_detections([a, b], iou_thr=0.9)
    assert len(kept_dets) == 2  # cross-class not collapsed
    assert kept_idx == [0, 1]


def test_dedup_keeps_distant_boxes():
    trk = CustomTracker(max_age=10, min_hits=1, iou_threshold=0.2)
    a = _det(100, 100, frame=0)
    b = _det(300, 300, frame=0)
    kept_idx, kept_dets = trk._dedup_detections([a, b], iou_thr=0.9)
    assert len(kept_dets) == 2
    assert kept_idx == [0, 1]


def test_dedup_empty_and_single():
    trk = CustomTracker(max_age=10, min_hits=1, iou_threshold=0.2)
    idx, dets = trk._dedup_detections([])
    assert idx == [] and dets == []
    idx, dets = trk._dedup_detections([_det(100, 100)])
    assert idx == [0] and len(dets) == 1


def test_dedup_creates_single_track_from_duplicates():
    trk = CustomTracker(max_age=10, min_hits=1, iou_threshold=0.2)
    a = _det(100, 100, conf=0.9, frame=0)
    b = _det(101, 101, conf=0.7, frame=0)
    matched = trk.update([a, b], frame=0)
    # one detection kept -> one track
    assert len(trk.tracks) == 1
