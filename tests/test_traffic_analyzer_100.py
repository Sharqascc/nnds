import json
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

import src.pipeline.traffic_analyzer as ta
from src.pipeline.traffic_analyzer import (
    CompleteTrafficAnalyzer,
    main,
    run_pipeline,
    run_video_to_pet,
    run_video_to_pet_fixed,
)


def _dummy_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    return path


# ---------------- CompleteTrafficAnalyzer missing branches ----------------


def test_calibrate_homography_none():
    analyzer = CompleteTrafficAnalyzer()
    with mock.patch("cv2.findHomography", return_value=(None, None)):
        with pytest.raises(RuntimeError, match="Homography estimation failed"):
            analyzer.calibrate([[0, 0], [1, 1]], [[0, 0, 0], [1, 1, 0]])


def test_calculate_bev_scale_early_return():
    analyzer = CompleteTrafficAnalyzer()
    # world_points_approx and inlier_mask are None -> should return
    analyzer._calculate_bev_scale()
    assert analyzer.bev_x_min is None


def test_pixel_to_world_no_homography():
    analyzer = CompleteTrafficAnalyzer()
    with pytest.raises(RuntimeError, match="Homography not initialized"):
        analyzer.pixel_to_world([0, 0])


def test_validate_bev_no_calibration():
    analyzer = CompleteTrafficAnalyzer()
    with pytest.raises(RuntimeError, match="Calibration must be run before BEV validation"):
        analyzer.validate_bev()


def test_estimate_speed_no_homography():
    analyzer = CompleteTrafficAnalyzer()
    with pytest.raises(RuntimeError, match="Homography not initialized"):
        analyzer.estimate_speed(np.array([[0, 0]]), np.array([0]))


def test_estimate_speed_length_mismatch():
    analyzer = CompleteTrafficAnalyzer()
    analyzer.homography = np.eye(3, dtype=np.float32)
    with pytest.raises(ValueError, match="same length"):
        analyzer.estimate_speed(np.array([[0, 0]]), np.array([0, 1]))


def test_estimate_speed_not_enough_world_positions():
    analyzer = CompleteTrafficAnalyzer()
    analyzer.homography = np.eye(3, dtype=np.float32)
    # Only 3 finite positions -> fallback
    pos = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32)
    times = np.array([0.0, 0.1, 0.2], dtype=np.float32)
    res = analyzer.estimate_speed(pos, times)
    assert res["final_speed"] == 15.0
    assert res["speed_std"] == 2.0


def test_estimate_speed_time_mask_not_enough():
    analyzer = CompleteTrafficAnalyzer()
    analyzer.homography = np.eye(3, dtype=np.float32)
    pos = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.float32)
    times = np.array([0.0, 0.1, 0.2, np.nan, np.nan], dtype=np.float32)
    res = analyzer.estimate_speed(pos, times)
    assert res["final_speed"] == 15.0


def test_estimate_speed_not_enough_speeds():
    analyzer = CompleteTrafficAnalyzer()
    analyzer.homography = np.eye(3, dtype=np.float32)
    # All speeds will be filtered out (e.g., huge speed >50)
    pos = np.array(
        [[0, 0], [1000, 1000], [2000, 2000], [3000, 3000], [4000, 4000]], dtype=np.float32
    )
    times = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    res = analyzer.estimate_speed(pos, times)
    assert res["final_speed"] == 15.0


def test_save_calibration_with_homography(tmp_path):
    analyzer = CompleteTrafficAnalyzer()
    analyzer.homography = np.eye(3, dtype=np.float32)
    analyzer.calibration_metrics = {"final_mae": 0.1}
    analyzer.bev_x_min = 0
    analyzer.bev_x_max = 10
    analyzer.bev_y_min = 0
    analyzer.bev_y_max = 20
    out = tmp_path / "calib.json"
    analyzer.save_calibration(out)
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["homography"] is not None


def test_run_demo_full():
    # Should run without error
    analyzer, speed_results, metrics = ta.run_demo()
    assert isinstance(analyzer, CompleteTrafficAnalyzer)
    assert "final_speed" in speed_results
    assert "mae" in metrics


# ---------------- run_video_to_pet missing branches ----------------


















# ---------------- main missing branches ----------------


def test_main_demo_mode(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["traffic_analyzer.py", "--demo"])
    # Should not raise
    main()


def test_main_no_video(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["traffic_analyzer.py"])
    with pytest.raises(SystemExit, match="--video is required"):
        main()


def test_main_video_not_found(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["traffic_analyzer.py", "--video", "/nonexistent/video.mp4"])
    with pytest.raises(SystemExit, match="Video file not found"):
        main()


def test_main_runs_pipeline(monkeypatch, tmp_path):
    video = _dummy_file(tmp_path / "video.mp4")
    monkeypatch.setattr(
        sys, "argv", ["traffic_analyzer.py", "--video", str(video), "--detector", "sam3"]
    )
    with mock.patch("src.pipeline.traffic_analyzer.run_pipeline") as mock_run:
        main()
        mock_run.assert_called_once()


# ---------------- run_video_to_pet_fixed success ----------------




# ---------------- interactive_detector success ----------------


def test_run_pipeline_unsupported():
    with pytest.raises(ValueError, match="Unsupported detector policy"):
        run_pipeline(SimpleNamespace(detector="invalid", video="dummy", out_csv=None))
