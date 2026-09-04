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


def test_run_video_to_pet_sam3_missing_weights_success(tmp_path, monkeypatch):
    video = _dummy_file(tmp_path / "video.mp4")
    bev = _dummy_file(tmp_path / "bev.json")
    grid = _dummy_file(tmp_path / "grid.json")
    out = tmp_path / "out.csv"
    # sam3_weights_path does not exist -> will be set to None
    missing_sam3 = tmp_path / "missing.pt"

    class FakeResult:
        pet_events = [
            {"pet": 1.0, "frame_idx": 5, "track_a": 1, "track_b": 2, "conflict_type": "crossing"}
        ]

    with mock.patch(
        "src.analysis.grid_trajectory.sam3_grid_pet.run_sam3_grid_pet", return_value=FakeResult()
    ):
        df = run_video_to_pet(
            video,
            bev_config_path=bev,
            grid_config_path=grid,
            detector="sam3",
            sam3_weights_path=missing_sam3,
            out_csv_path=out,
        )
    assert len(df) == 1
    assert df.iloc[0]["pet"] == 1.0


def test_run_video_to_pet_yolo_cpu_success(tmp_path):
    video = _dummy_file(tmp_path / "video.mp4")
    bev = _dummy_file(tmp_path / "bev.json")
    grid = _dummy_file(tmp_path / "grid.json")
    yolo = _dummy_file(tmp_path / "yolo.pt")
    out = tmp_path / "out.csv"

    with mock.patch(
        "src.analysis.grid_trajectory.yolo_cpu_grid_pet.run_yolo_cpu_grid_pet",
        return_value={
            "pet_events": [
                {
                    "pet": 2.0,
                    "frame": 10,
                    "track_a": "track_3",
                    "track_b": "track_4",
                    "conflict_type": "rear-end",
                }
            ]
        },
    ):
        df = run_video_to_pet(
            video,
            bev_config_path=bev,
            grid_config_path=grid,
            detector="yolo-cpu",
            yolo_weights_path=yolo,
            out_csv_path=out,
        )
    assert len(df) == 1
    assert df.iloc[0]["track_a"] == 3  # parsed from string


def test_run_video_to_pet_uvh_coco_success(tmp_path):
    video = _dummy_file(tmp_path / "video.mp4")
    bev = _dummy_file(tmp_path / "bev.json")
    grid = _dummy_file(tmp_path / "grid.json")
    uvh = _dummy_file(tmp_path / "uvh.pt")
    coco = _dummy_file(tmp_path / "coco.pt")
    out = tmp_path / "out.csv"

    pet_event = {
        "pet": 1.5,
        "frame": 12,
        "track_a": 5,
        "track_b": 6,
        "conflict_type": "lane_change",
        "grid_cell": "A1",
        "track_a_entry_frame": 1,
        "track_a_exit_frame": 2,
        "track_a_exit_time_sec": 0.1,
        "track_b_entry_frame": 3,
        "track_b_entry_time_sec": 0.2,
        "track_b_exit_frame": 4,
        "world_traj_i": [0, 1],
        "world_traj_j": [2, 3],
        "traj_a_json": "[]",
        "traj_b_json": "[]",
        "video_source": "test",
        "time_of_day_label": "morning",
        "gate_a_entry": "left",
        "gate_b_entry": "right",
    }
    with mock.patch(
        "src.analysis.grid_trajectory.uvh_coco_fused_grid_pet.run_uvh_coco_fused_grid_pet",
        return_value={"pet_events": [pet_event]},
    ):
        df = run_video_to_pet(
            video,
            bev_config_path=bev,
            grid_config_path=grid,
            detector="uvh-coco-fused",
            uvh_model_path=uvh,
            coco_person_model_path=coco,
            out_csv_path=out,
        )
    assert len(df) == 1
    assert df.iloc[0]["gate_a_entry"] == "left"


def test_run_video_to_pet_rtdetr_not_implemented(tmp_path):
    video = _dummy_file(tmp_path / "video.mp4")
    bev = _dummy_file(tmp_path / "bev.json")
    grid = _dummy_file(tmp_path / "grid.json")
    rtdetr = _dummy_file(tmp_path / "rtdetr.pt")
    with pytest.raises(NotImplementedError, match="RT-DETR video pipeline is not implemented"):
        run_video_to_pet(
            video,
            bev_config_path=bev,
            grid_config_path=grid,
            detector="rtdetr",
            rtdetr_weights_path=rtdetr,
        )


def test_run_video_to_pet_empty_events_writes_csv(tmp_path):
    video = _dummy_file(tmp_path / "video.mp4")
    bev = _dummy_file(tmp_path / "bev.json")
    grid = _dummy_file(tmp_path / "grid.json")
    sam3 = _dummy_file(tmp_path / "sam3.pt")
    out = tmp_path / "out.csv"

    class FakeResult:
        pet_events = []

    with mock.patch(
        "src.analysis.grid_trajectory.sam3_grid_pet.run_sam3_grid_pet", return_value=FakeResult()
    ):
        with pytest.warns(RuntimeWarning, match="No PET events detected"):
            df = run_video_to_pet(
                video,
                bev_config_path=bev,
                grid_config_path=grid,
                detector="sam3",
                sam3_weights_path=sam3,
                out_csv_path=out,
            )
    assert df.empty
    assert out.exists()


def test_run_video_to_pet_dict_events_branch(tmp_path):
    video = _dummy_file(tmp_path / "video.mp4")
    bev = _dummy_file(tmp_path / "bev.json")
    grid = _dummy_file(tmp_path / "grid.json")
    sam3 = _dummy_file(tmp_path / "sam3.pt")
    out = tmp_path / "out.csv"

    # Dict events to cover _get dict branch
    event = {
        "pet": 1.0,
        "frame": 20,
        "track_a": 7,  # direct int
        "track_b": 8,
        "conflict_type": "head_on",
        "grid_cell": "B2",
        "track_a_entry_frame": 10,
        "track_a_exit_frame": 11,
        "track_b_entry_frame": 12,
        "track_b_exit_frame": 13,
        "world_traj_i": [10, 11],
        "world_traj_j": [12, 13],
        "traj_a_json": "[]",
        "traj_b_json": "[]",
        "gate_a_entry": "left",
        "gate_b_entry": "right",
    }
    with mock.patch(
        "src.analysis.grid_trajectory.sam3_grid_pet.run_sam3_grid_pet",
        return_value=type("Result", (), {"pet_events": [event]}),
    ):
        df = run_video_to_pet(
            video,
            bev_config_path=bev,
            grid_config_path=grid,
            detector="sam3",
            sam3_weights_path=sam3,
            out_csv_path=out,
        )
    assert len(df) == 1
    assert df.iloc[0]["track_a"] == 7


def test_run_video_to_pet_parse_track_id_float_nan(tmp_path):
    video = _dummy_file(tmp_path / "video.mp4")
    bev = _dummy_file(tmp_path / "bev.json")
    grid = _dummy_file(tmp_path / "grid.json")
    sam3 = _dummy_file(tmp_path / "sam3.pt")
    out = tmp_path / "out.csv"

    event = {
        "pet": 0.5,
        "frame": 21,
        "track_a": np.nan,  # float NaN -> should return -1
        "track_b": 9,
        "conflict_type": "rear-end",
        "grid_cell": "C3",
    }
    with mock.patch(
        "src.analysis.grid_trajectory.sam3_grid_pet.run_sam3_grid_pet",
        return_value=type("Result", (), {"pet_events": [event]}),
    ):
        df = run_video_to_pet(
            video,
            bev_config_path=bev,
            grid_config_path=grid,
            detector="sam3",
            sam3_weights_path=sam3,
            out_csv_path=out,
        )
    assert df.iloc[0]["track_a"] == -1


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


def test_run_video_to_pet_fixed_success(tmp_path):
    video = _dummy_file(tmp_path / "video.mp4")
    bev = _dummy_file(tmp_path / "bev.json")
    grid = _dummy_file(tmp_path / "grid.json")
    sam3 = _dummy_file(tmp_path / "sam3.pt")
    out = tmp_path / "out.csv"

    class FakeResult:
        pet_events = [
            {"pet": 1.2, "frame_idx": 30, "track_a": 10, "track_b": 11, "conflict_type": "crossing"}
        ]

    with mock.patch(
        "src.analysis.grid_trajectory.sam3_grid_pet.run_sam3_grid_pet", return_value=FakeResult()
    ):
        df = run_video_to_pet_fixed(
            str(video),
            bev_config_path=str(bev),
            grid_config_path=str(grid),
            sam3_weights_path=str(sam3),
            out_csv_path=str(out),
        )
    assert len(df) == 1
    assert out.exists()


# ---------------- interactive_detector success ----------------


def test_run_pipeline_unsupported():
    with pytest.raises(ValueError, match="Unsupported detector policy"):
        run_pipeline(SimpleNamespace(detector="invalid", video="dummy", out_csv=None))
