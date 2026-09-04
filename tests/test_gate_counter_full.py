from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pandas as pd
import pytest

from src.analysis.gate_counter import RobustTracker, TrafficVolumeCounter, VirtualGate

# ---------- VirtualGate ----------


def make_gate(**kwargs):
    defaults = dict(name="G1", p1=(0, 0), p2=(10, 0), entry_side="left")
    defaults.update(kwargs)
    return VirtualGate(**defaults)


def test_virtual_gate_post_init_min_frames():
    gate = make_gate(min_frames_between_crossings=0)
    assert gate.min_frames_between_crossings == 1
    gate2 = make_gate(min_frames_between_crossings=5)
    assert gate2.min_frames_between_crossings == 5


def test_direction_nonzero():
    gate = make_gate(p1=(0, 0), p2=(10, 0))
    d = gate.direction()
    assert np.allclose(d, [1, 0])


def test_direction_zero():
    gate = make_gate(p1=(0, 0), p2=(0, 0))
    d = gate.direction()
    assert np.allclose(d, [1, 0])


def test_signed_distance():
    gate = make_gate(p1=(0, 0), p2=(10, 0))
    assert gate.signed_distance((5, 5)) < 0
    assert gate.signed_distance((5, -5)) > 0
    assert gate.signed_distance((5, 0)) == 0.0


def test_check_crossing_none():
    gate = make_gate()
    assert gate.check_crossing(None, (1, 1), 1, 10) is None
    assert gate.check_crossing((0, 0), None, 1, 10) is None


def test_check_crossing_near_zero():
    gate = make_gate()
    # One side exactly zero -> None
    assert gate.check_crossing((0, 0), (0, 1), 1, 10) is None  # prev side zero
    assert gate.check_crossing((0, 1), (0, 0), 1, 10) is None  # curr side zero


def test_check_crossing_same_side():
    gate = make_gate()
    assert gate.check_crossing((5, 1), (5, 2), 1, 10) is None


def test_check_crossing_too_soon():
    gate = make_gate(min_frames_between_crossings=10)
    gate.history[1] = 5
    assert gate.check_crossing((5, -1), (5, 1), 1, 10) is None


def test_check_crossing_entry_left():
    gate = make_gate(entry_side="left")
    # signed_distance = -y, so negative side is y>0, positive side is y<0
    res = gate.check_crossing((5, 1), (5, -1), 1, 10)
    assert res == "entry"
    assert gate.entry_count == 1
    assert gate.exit_count == 0


def test_check_crossing_exit_left():
    gate = make_gate(entry_side="left")
    res = gate.check_crossing((5, -1), (5, 1), 1, 10)
    assert res == "exit"
    assert gate.exit_count == 1


def test_check_crossing_entry_right():
    gate = make_gate(entry_side="right")
    # For right: neg->pos = exit, pos->neg = entry
    res = gate.check_crossing((5, -1), (5, 1), 1, 10)
    assert res == "entry"
    assert gate.entry_count == 1


def test_check_crossing_exit_right():
    gate = make_gate(entry_side="right")
    res = gate.check_crossing((5, 1), (5, -1), 1, 10)
    assert res == "exit"
    assert gate.exit_count == 1


# ---------- RobustTracker ----------


def test_robust_tracker_update_new_detections():
    tracker = RobustTracker()
    dets = [{"centroid": (10, 10)}]
    out = tracker.update(dets, frame_idx=1)
    assert 1 in out
    assert out[1]["track_id"] == 1
    assert out[1]["centroid"] == (10, 10)
    assert out[1]["missed"] == 0


def test_robust_tracker_existing_track_prev_centroid():
    tracker = RobustTracker()
    det1 = {"track_id": 1, "centroid": (0, 0)}
    tracker.update([det1], frame_idx=1)
    det2 = {"track_id": 1, "centroid": (5, 5)}
    out = tracker.update([det2], frame_idx=2)
    assert out[1]["prev_centroid"] == (0, 0)
    assert out[1]["centroid"] == (5, 5)


def test_robust_tracker_missing_frames():
    tracker = RobustTracker(max_missing=2)
    tracker.update([{"track_id": 1, "centroid": (0, 0)}], frame_idx=1)
    # No detections for this track
    out = tracker.update([], frame_idx=2)
    assert 1 in out
    assert out[1]["missed"] == 1
    # Second missed frame
    out = tracker.update([], frame_idx=3)
    assert 1 in out
    assert out[1]["missed"] == 2
    # Third missed frame -> dropped
    out = tracker.update([], frame_idx=4)
    assert 1 not in out


def test_robust_tracker_cleanup_old_tracks():
    tracker = RobustTracker(max_track_age_frames=50)
    tracker._frame_counter = 200
    # Track last seen 100 frames ago
    tracker.tracks = {1: {"last_seen_frame": 100, "centroid": (0, 0), "missed": 0}}
    tracker._cleanup_old_tracks()
    assert 1 not in tracker.tracks


# ---------- TrafficVolumeCounter ----------


def make_counter(tmp_path, gate_config=None, classes=None, **kwargs):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    kwargs.setdefault("videopath", str(video))
    kwargs.setdefault("gate_config", gate_config)
    kwargs.setdefault("classes_of_interest", classes)
    return TrafficVolumeCounter(**kwargs)


def test_traffic_volume_init_missing_video(tmp_path):
    with pytest.raises(FileNotFoundError):
        TrafficVolumeCounter(videopath=str(tmp_path / "nonexistent.mp4"))


def test_traffic_volume_default_classes(tmp_path):
    counter = make_counter(tmp_path)
    assert "car" in counter._class_whitelist
    assert counter.gates == {}


def test_traffic_volume_load_gates(tmp_path):
    config = tmp_path / "gates.yaml"
    config.write_text("""
gates:
  - name: West_Gate
    start: [10, 20]
    end: [30, 40]
    color: [255, 0, 0]
    entry_side: right
    enabled: true
""")
    counter = make_counter(tmp_path, gate_config=str(config))
    assert "West_Gate" in counter.gates
    gate = counter.gates["West_Gate"]
    assert gate.entry_side == "right"
    assert gate.color == (0, 0, 255)  # BGR from RGB


def test_load_gates_missing_file(tmp_path):
    gates = TrafficVolumeCounter.load_gates(str(tmp_path / "missing.yaml"))
    assert gates == {}


def test_load_gates_invalid_yaml(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text(": invalid yaml")
    gates = TrafficVolumeCounter.load_gates(str(config))
    assert gates == {}


def test_load_gates_disabled_and_no_name(tmp_path):
    config = tmp_path / "gates.yaml"
    config.write_text("""
gates:
  - name: G1
    enabled: false
  - start: [0,0]
    end: [10,10]
  - name: G3
    start: [1,1]
    end: [2,2]
    entry_side: invalid
    color: [0,255,0]
""")
    gates = TrafficVolumeCounter.load_gates(str(config))
    assert "G1" not in gates
    assert "G3" in gates
    assert gates["G3"].entry_side == "left"


def test_format_time():
    assert TrafficVolumeCounter._format_time(0, 30) == "0:00:00"
    from datetime import timedelta

    assert TrafficVolumeCounter._format_time(60, 30) == str(timedelta(seconds=2))
    assert TrafficVolumeCounter._format_time(60, 0) == "00:00:00"


def test_compute_totals(tmp_path):
    counter = make_counter(tmp_path)
    counter.gates["G1"] = VirtualGate(name="G1", p1=(0, 0), p2=(1, 0))
    counter.gates["G1"].entry_count = 3
    counter.gates["G1"].exit_count = 2
    assert counter._compute_totals() == (3, 2)


def test_normalize_class_name(tmp_path):
    counter = make_counter(tmp_path)
    assert counter._normalize_class_name(" Car ") == "car"
    assert counter._normalize_class_name(None) == "object"


def test_allowed_detection(tmp_path):
    counter = make_counter(tmp_path, classes=["car"])
    det = {"centroid": (10, 10), "cls": "car", "conf": 0.9}
    assert counter._allowed_detection(det) == True
    det2 = {"centroid": (10, 10), "cls": "person", "conf": 0.9}
    assert counter._allowed_detection(det2) == False
    det3 = {"centroid": (10, 10), "cls": "car", "conf": 0.1}
    assert counter._allowed_detection(det3) == False
    det4 = {"cls": "car", "conf": 0.9}  # missing centroid
    assert counter._allowed_detection(det4) == False


def test_draw_gate_labels(tmp_path):
    counter = make_counter(tmp_path)
    counter.gates["G1"] = VirtualGate(name="G1", p1=(10, 10), p2=(20, 20))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    counter._draw_gate_labels(frame)
    assert frame.sum() > 0


def test_draw_tracks(tmp_path):
    counter = make_counter(tmp_path)
    tracks = {1: {"centroid": (30, 30), "cls": "car", "conf": 0.9}}
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    counter._draw_tracks(frame, tracks)
    assert frame.sum() > 0


def test_draw_stats_panel(tmp_path):
    counter = make_counter(tmp_path)
    counter.gates["G1"] = VirtualGate(name="G1", p1=(0, 0), p2=(1, 0))
    counter.gates["G1"].entry_count = 2
    counter.gates["G1"].exit_count = 1
    counter.last_event = "G1 ENTRY | ID 1 | car"
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    out = counter._draw_stats_panel(frame, frame_idx=10, fps=30)
    assert out.shape == frame.shape


def test_save_results(tmp_path):
    result = {
        "total_entries": 5,
        "total_exits": 3,
        "gates": {
            "G1": {"entries": 4, "exits": 2},
            "G2": {"entries": 1, "exits": 1},
        },
    }
    out_path = tmp_path / "results.csv"
    TrafficVolumeCounter.save_results(result, out_path)
    assert out_path.exists()
    df = pd.read_csv(out_path)
    assert len(df) == 2


# ---------- process_video with mocks ----------


def test_process_video_full(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    counter = TrafficVolumeCounter(videopath=str(video), gate_config=None)

    # Mock cv2.VideoCapture, VideoWriter, tqdm
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: 30.0,
        cv2.CAP_PROP_FRAME_WIDTH: 640,
        cv2.CAP_PROP_FRAME_HEIGHT: 480,
        cv2.CAP_PROP_FRAME_COUNT: 3,
    }.get(prop, 0)
    # Simulate 3 frames
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cap.read.side_effect = [
        (True, frame.copy()),
        (True, frame.copy()),
        (True, frame.copy()),
        (False, None),
    ]

    with (
        patch("cv2.VideoCapture", return_value=cap),
        patch("cv2.VideoWriter", MagicMock()) as writer_cls,
        patch("cv2.VideoWriter_fourcc", return_value=0),
        patch("src.analysis.gate_counter.logger") as logger_mock,
    ):
        writer = MagicMock()
        writer_cls.return_value = writer

        def detector(frame):
            return [{"centroid": (100, 100), "cls": "car", "conf": 0.9}]

        result = counter.process_video(
            detector, output_video=str(tmp_path / "out.mp4"), max_frames=3, show_progress=False
        )
    assert result["total_entries"] == 0  # no gate crossings because no gates defined
    assert result["total_exits"] == 0


def test_process_video_no_output(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    counter = TrafficVolumeCounter(videopath=str(video))

    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: 0,  # invalid -> default 25
        cv2.CAP_PROP_FRAME_WIDTH: 320,
        cv2.CAP_PROP_FRAME_HEIGHT: 240,
        cv2.CAP_PROP_FRAME_COUNT: 2,
    }.get(prop, 0)
    cap.read.side_effect = [
        (True, np.zeros((240, 320, 3), dtype=np.uint8)),
        (True, np.zeros((240, 320, 3), dtype=np.uint8)),
        (False, None),
    ]

    with patch("cv2.VideoCapture", return_value=cap):
        result = counter.process_video(detector=lambda f: [], max_frames=2, show_progress=False)
    assert result["total_entries"] == 0
