
import numpy as np
import cv2
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.pipeline.custom_tracker import Detection, KalmanTrack, CustomTracker


def make_detection(frame=0, x1=0, y1=0, x2=10, y2=10, cls_id=0, cls_name="car", conf=0.9, source="uvh26", hist=None, embedding=None):
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return Detection(
        frame=frame, x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
        cx=cx, cy=cy, cls_id=cls_id, cls_name=cls_name, conf=conf, source=source,
        hist=hist, embedding=embedding,
    )


# ---------------- KalmanTrack tests ----------------

def test_kalman_track_init():
    det = make_detection(x1=5, y1=5, x2=15, y2=20)
    track = KalmanTrack(det, 1)
    assert track.id == 1
    assert track.cls_name == "car"
    assert track.age == 0
    assert track.time_since_update == 0
    assert track.hist is None
    assert track.embedding is None
    assert track.center == (det.cx, det.cy)  # statePost initial


def test_kalman_track_predict():
    det = make_detection()
    track = KalmanTrack(det, 2)
    track.predict()
    assert track.age == 1
    assert track.time_since_update == 1


def test_kalman_track_update_no_hist_no_embedding():
    det = make_detection()
    track = KalmanTrack(det, 3)
    det2 = make_detection(x1=20, y1=20, x2=30, y2=30)
    track.predict()
    track.update(det2)
    assert track.time_since_update == 0
    assert track.hist is None
    assert track.embedding is None


def test_kalman_track_update_hist_first():
    hist = np.array([0.1, 0.2, 0.3])
    det = make_detection(hist=hist)
    track = KalmanTrack(det, 4)
    det2 = make_detection(hist=hist)
    track.update(det2)
    assert track.hist is not None
    # should be the det2.hist (since track.hist was None, then becomes det.hist; then weighted average when updated)
    # Actually first update sets track.hist = det2.hist because track.hist was None? Wait initial det had hist, so track.hist is not None.
    # track.hist = det.hist in constructor, so update with det2.hist -> weighted avg.
    assert np.allclose(track.hist, 0.8 * hist + 0.2 * hist)


def test_kalman_track_update_hist_second():
    hist = np.array([0.5, 0.5, 0.5])
    det = make_detection()  # hist None initially
    track = KalmanTrack(det, 5)
    det2 = make_detection(hist=hist)
    track.update(det2)
    assert track.hist is hist  # exactly det2.hist because track.hist was None


def test_kalman_track_update_embedding_first():
    emb = np.array([0.1, 0.2])
    det = make_detection(embedding=emb)
    track = KalmanTrack(det, 6)
    det2 = make_detection(embedding=emb)
    track.update(det2)
    assert track.embedding is not None
    assert np.allclose(track.embedding, 0.8 * emb + 0.2 * emb)


def test_kalman_track_update_embedding_second():
    emb = np.array([0.8, 0.1])
    det = make_detection()
    track = KalmanTrack(det, 7)
    det2 = make_detection(embedding=emb)
    track.update(det2)
    assert track.embedding is emb


def test_kalman_track_box_property():
    det = make_detection(x1=10, y1=10, x2=20, y2=30)
    track = KalmanTrack(det, 8)
    box = track.box
    assert box[0] < box[2]
    assert box[1] < box[3]
    # center should be within box
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    assert abs(cx - det.cx) < 1e-6
    assert abs(cy - det.cy) < 1e-6


# ---------------- CustomTracker tests ----------------

def test_custom_tracker_init_no_log():
    tracker = CustomTracker()
    assert tracker.log_handle is None
    assert tracker.next_id == 1
    assert tracker.tracks == {}


def test_custom_tracker_init_log(tmp_path):
    log_path = tmp_path / "overlap.log"
    tracker = CustomTracker(log_overlaps=True, overlap_log_path=str(log_path))
    assert tracker.log_handle is not None
    tracker.log_handle.close()


def test_iou_overlap():
    tracker = CustomTracker()
    iou = tracker._iou((0, 0, 10, 10), (5, 5, 15, 15))
    assert iou == pytest.approx(25 / 175)  # intersection 25, union 100+100-25=175


def test_iou_no_overlap():
    tracker = CustomTracker()
    assert tracker._iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_zero_area():
    tracker = CustomTracker()
    assert tracker._iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


def test_cosine_similarity_none():
    tracker = CustomTracker()
    assert tracker._cosine_similarity(None, np.array([1, 2])) == 0.0
    assert tracker._cosine_similarity(np.array([1, 2]), None) == 0.0


def test_cosine_similarity_valid():
    tracker = CustomTracker()
    a = np.array([1, 0])
    b = np.array([1, 0])
    assert tracker._cosine_similarity(a, b) == pytest.approx(1.0)


def test_appearance_cost_embedding():
    tracker = CustomTracker()
    emb = np.array([1, 0])
    cost = tracker._appearance_cost(None, None, emb, emb)
    assert cost < 0.01


def test_appearance_cost_hist():
    tracker = CustomTracker()
    hist = np.array([0.2, 0.3, 0.5])
    cost = tracker._appearance_cost(hist, hist, None, None)
    assert cost < 0.01


def test_appearance_cost_none():
    tracker = CustomTracker()
    assert tracker._appearance_cost(None, None, None, None) == 0.0


def test_update_empty_detections():
    tracker = CustomTracker()
    result = tracker.update([])
    assert result == {}


def test_update_no_tracks_creates_new():
    tracker = CustomTracker()
    det = make_detection()
    result = tracker.update([det])
    assert len(result) == 1
    assert 0 in result
    assert result[0] == 1
    assert 1 in tracker.tracks


def test_update_iou_matching():
    tracker = CustomTracker()
    det1 = make_detection(x1=0, y1=0, x2=20, y2=20)
    tracker.update([det1])  # creates track 1

    det2 = make_detection(x1=1, y1=1, x2=21, y2=21)
    matched = tracker.update([det2])
    assert 0 in matched
    assert matched[0] == 1


def test_update_stage2_matching():
    tracker = CustomTracker()
    det1 = make_detection(x1=0, y1=0, x2=20, y2=20, hist=np.array([0.1, 0.2, 0.3]))
    tracker.update([det1])  # track 1

    # New detection far away but with same hist -> stage 2 should match
    det2 = make_detection(x1=50, y1=50, x2=70, y2=70, hist=np.array([0.1, 0.2, 0.3]))
    matched = tracker.update([det2])
    assert len(matched) == 1


def test_update_dead_track_removal():
    tracker = CustomTracker(max_age=1)
    det1 = make_detection(x1=0, y1=0, x2=20, y2=20)
    tracker.update([det1])  # track 1 with time_since_update=0
    # Far detection so no match, then dead track removal runs
    det2 = make_detection(x1=1000, y1=1000, x2=1020, y2=1020)
    result = tracker.update([det2])
    # Old track should be removed (time_since_update = 1 >= max_age)
    assert 1 not in tracker.tracks
    # New track created
    assert len(tracker.tracks) == 1


def test_update_log_overlaps(tmp_path):
    log_path = tmp_path / "overlap.log"
    tracker = CustomTracker(log_overlaps=True, overlap_log_path=str(log_path))
    det1 = make_detection(x1=0, y1=0, x2=20, y2=20)
    det2 = make_detection(x1=1, y1=1, x2=21, y2=21)
    tracker.update([det1, det2])
    tracker.update([det1])  # existing tracks
    assert tracker.log_handle is not None
    # Close and check file has content
    tracker.log_handle.close()
    content = log_path.read_text()
    assert len(content) > 0


def test_update_log_overlaps_no_overlap(tmp_path):
    log_path = tmp_path / "overlap.log"
    tracker = CustomTracker(log_overlaps=True, overlap_log_path=str(log_path))
    det1 = make_detection(x1=0, y1=0, x2=10, y2=10)
    det2 = make_detection(x1=100, y1=100, x2=110, y2=110)
    tracker.update([det1, det2])
    tracker.update([det1])
    tracker.log_handle.close()
    content = log_path.read_text()
    # No overlap, so only header line
    lines = content.strip().splitlines()
    assert len(lines) == 1


def test_update_reid_encoder_called():
    tracker = CustomTracker(reid_encoder=MagicMock(return_value=None))
    # Actually reid_encoder should be object with encode_crop method
    mock_encoder = MagicMock()
    mock_encoder.encode_crop.return_value = np.array([0.1, 0.2, 0.3])
    tracker = CustomTracker(reid_encoder=mock_encoder)
    det = make_detection(x1=0, y1=0, x2=20, y2=20)
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    tracker.update([det], frame_img=frame)
    # After first update, track exists. Now add new detection with no embedding, trigger stage2
    det2 = make_detection(x1=50, y1=50, x2=70, y2=70)
    matched = tracker.update([det2], frame_img=frame)
    # Encoder should have been called at least once (for det2 in stage2)
    assert mock_encoder.encode_crop.called
