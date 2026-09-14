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


def test_dedup_remaps_indices_correctly_with_middle_drop():
    """Middle-drop case: exercises the index-remap bug class directly.

    Pre-fix behavior: dedup returned a re-indexed list, so a match at
    position 1 in the deduped list got attributed to the wrong detection
    row in the caller. With 3 detections where #1 is dropped, the caller
    must receive indices [0, 2], not [0, 1].
    """
    trk = CustomTracker(max_age=10, min_hits=1, iou_threshold=0.2)
    a = _det(100, 100, conf=0.9, frame=0)  # index 0, keep
    b = _det(101, 101, conf=0.5, frame=0)  # index 1, dup of a, drop
    c = _det(500, 500, conf=0.8, frame=0)  # index 2, keep
    kept_idx, kept_dets = trk._dedup_detections([a, b, c], iou_thr=0.9)
    assert kept_idx == [0, 2], f"expected [0,2], got {kept_idx}"
    assert len(kept_dets) == 2
    assert kept_dets[0].conf == 0.9
    assert kept_dets[1].conf == 0.8


def test_dedup_remap_preserves_track_assignment():
    """End-to-end: a duplicate must not shift which detection gets which track."""
    trk = CustomTracker(max_age=10, min_hits=1, iou_threshold=0.2)
    # Frame 0: two real objects (car A at 100, car B at 500)
    a0 = _det(100, 100, conf=0.9, frame=0)
    b0 = _det(500, 500, conf=0.9, frame=0)
    m0 = trk.update([a0, b0], frame=0)
    assert set(m0.keys()) == {0, 1}  # both detections matched
    id_a = m0[0]
    id_b = m0[1]
    assert id_a != id_b

    # Frame 1: car A now emitted TWICE (duplicate), car B once
    a1 = _det(102, 102, conf=0.9, frame=1)  # index 0, real
    a1_dup = _det(103, 103, conf=0.6, frame=1)  # index 1, dup
    b1 = _det(502, 502, conf=0.9, frame=1)  # index 2, real
    m1 = trk.update([a1, a1_dup, b1], frame=1)
    # Detections 0 and 2 must map to the existing tracks; 1 must not be returned
    assert 1 not in m1, "duplicate detection should not be matched"
    assert m1[0] == id_a, "car A's real detection should keep its track ID"
    assert m1[2] == id_b, "car B's detection should keep its track ID"
