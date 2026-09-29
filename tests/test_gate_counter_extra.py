from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.analysis.gate_counter import RobustTracker, TrafficVolumeCounter, VirtualGate


def test_robust_tracker_existing_track_prev_centroid():
    tracker = RobustTracker()
    tracker.tracks = {1: {"prev_centroid": (5, 5)}}
    det = {"track_id": 1, "centroid": (10, 10)}
    out = tracker.update([det], frame_idx=1)
    assert out[1]["prev_centroid"] == (5, 5)


def test_robust_tracker_cleanup_old_tracks_at_100():
    tracker = RobustTracker(max_track_age_frames=50)
    tracker.tracks = {1: {"last_seen_frame": 0, "centroid": (0, 0), "missed": 0}}
    tracker._frame_counter = 100
    tracker.update([], frame_idx=100)
    assert 1 not in tracker.tracks


def test_load_gates_invalid_color(tmp_path):
    config = tmp_path / "gates.yaml"
    config.write_text(
        "gates:\n  - name: G1\n    start: [0,0]\n    end: [10,0]\n    color: bad-color\n    entry_side: left\n"
    )
    gates = TrafficVolumeCounter.load_gates(str(config))
    assert gates["G1"].color == (0, 255, 255)


def test_draw_tracks_skip_none_centroid(tmp_path):
    video = tmp_path / "dummy.mp4"
    video.write_bytes(b"fake")
    counter = TrafficVolumeCounter(videopath=str(video), gate_config=None)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    tracks = {1: {"centroid": None}}
    counter._draw_tracks(frame, tracks)


def test_process_video_could_not_open(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    counter = TrafficVolumeCounter(videopath=str(video))
    cap = MagicMock()
    cap.isOpened.return_value = False
    with patch("cv2.VideoCapture", return_value=cap), pytest.raises(RuntimeError):
        counter.process_video(detector=lambda f: [])


def test_process_video_tqdm_import_error(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    counter = TrafficVolumeCounter(videopath=str(video))
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: 30.0 if prop == cv2.CAP_PROP_FPS else 1
    calls = {"n": 0}

    def read_side_effect():
        if calls["n"] == 0:
            calls["n"] += 1
            return True, np.zeros((10, 10, 3), dtype=np.uint8)
        return False, None

    cap.read.side_effect = read_side_effect
    with (
        patch("cv2.VideoCapture", return_value=cap),
        patch("builtins.__import__", side_effect=ImportError("no tqdm")),
    ):
        result = counter.process_video(detector=lambda f: [], show_progress=True, max_frames=1)
    assert result["total_entries"] == 0


def test_process_video_preview_interval(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    counter = TrafficVolumeCounter(videopath=str(video))
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: 30.0 if prop == cv2.CAP_PROP_FPS else 1
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    cap.read.side_effect = [(True, frame), (False, None)]
    with (
        patch("cv2.VideoCapture", return_value=cap),
        patch("matplotlib.pyplot.imshow"),
        patch("matplotlib.pyplot.title"),
        patch("matplotlib.pyplot.axis"),
        patch("matplotlib.pyplot.pause"),
        patch("matplotlib.pyplot.clf"),
    ):
        counter.process_video(detector=lambda f: [], preview_interval=1, max_frames=0)


def test_process_video_gate_crossing_and_draw_tracks(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    counter = TrafficVolumeCounter(videopath=str(video), gate_config=None, draw_tracks=True)
    counter.gates["G1"] = VirtualGate(name="G1", p1=(0, 0), p2=(10, 0), entry_side="left")
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: 30.0,
        cv2.CAP_PROP_FRAME_WIDTH: 100,
        cv2.CAP_PROP_FRAME_HEIGHT: 100,
        cv2.CAP_PROP_FRAME_COUNT: 2,
    }.get(prop, 0)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    cap.read.side_effect = [(True, frame), (True, frame), (False, None)]
    detections = [
        [{"centroid": (5, 5), "cls": "car", "conf": 0.9, "track_id": 1}],
        [{"centroid": (5, -5), "cls": "car", "conf": 0.9, "track_id": 1}],
    ]
    call_count = {"n": 0}

    def detector(frame):
        idx = call_count["n"]
        call_count["n"] += 1
        return detections[idx] if idx < len(detections) else []

    with patch("cv2.VideoCapture", return_value=cap):
        result = counter.process_video(detector=detector, max_frames=2)
    assert result["total_entries"] + result["total_exits"] > 0


def test_process_video_tqdm_success_path(tmp_path):
    """Cover tqdm import success, pbar.update, pbar.close."""
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    counter = TrafficVolumeCounter(videopath=str(video))
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: 30.0 if prop == cv2.CAP_PROP_FPS else 1
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    cap.read.side_effect = [(True, frame), (False, None)]
    mock_pbar = MagicMock()
    with (
        patch("cv2.VideoCapture", return_value=cap),
        patch("tqdm.auto.tqdm", return_value=mock_pbar),
    ):
        result = counter.process_video(detector=lambda f: [], show_progress=True, max_frames=1)
    mock_pbar.update.assert_called_once_with(1)
    mock_pbar.close.assert_called_once()
    assert result["total_entries"] == 0


def test_process_video_draw_tracks_branch(tmp_path):
    """Cover _draw_tracks call when draw_tracks=True."""
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    counter = TrafficVolumeCounter(videopath=str(video), gate_config=None, draw_tracks=True)
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: 30.0,
        cv2.CAP_PROP_FRAME_WIDTH: 100,
        cv2.CAP_PROP_FRAME_HEIGHT: 100,
        cv2.CAP_PROP_FRAME_COUNT: 2,
    }.get(prop, 0)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    cap.read.side_effect = [(True, frame), (True, frame), (False, None)]
    det = {"centroid": (5, 5), "cls": "car", "conf": 0.9, "track_id": 1}
    with (
        patch("cv2.VideoCapture", return_value=cap),
        patch("cv2.VideoWriter", MagicMock()),
        patch("cv2.VideoWriter_fourcc", return_value=0),
        patch.object(counter, "_draw_tracks") as draw_mock,
    ):
        counter.process_video(
            detector=lambda f: [det], max_frames=2, output_video=str(tmp_path / "out.mp4")
        )
    draw_mock.assert_called()


def test_process_video_invalid_detection_continue(tmp_path):
    """Cover continue branch when a detection fails _allowed_detection."""
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    counter = TrafficVolumeCounter(videopath=str(video), gate_config=None)
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: 30.0,
        cv2.CAP_PROP_FRAME_WIDTH: 100,
        cv2.CAP_PROP_FRAME_HEIGHT: 100,
        cv2.CAP_PROP_FRAME_COUNT: 2,
    }.get(prop, 0)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    cap.read.side_effect = [(True, frame), (False, None)]

    # Invalid detection: missing centroid -> _allowed_detection False
    invalid_det = {"cls": "car", "conf": 0.9}
    with patch("cv2.VideoCapture", return_value=cap):
        result = counter.process_video(detector=lambda f: [invalid_det], max_frames=1)
    assert result["total_entries"] == 0
