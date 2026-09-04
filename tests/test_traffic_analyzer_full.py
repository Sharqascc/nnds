import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.pipeline.traffic_analyzer import (
    CompleteTrafficAnalyzer,
    interactive_detector,
    run_pipeline,
    run_video_to_pet,
    run_video_to_pet_fixed,
)

# ---------------- CompleteTrafficAnalyzer.calibrate ----------------


def test_calibrate_homography_failure():
    analyzer = CompleteTrafficAnalyzer()
    pixel = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    world = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    with patch("cv2.findHomography", return_value=(None, None)), pytest.raises(RuntimeError):
        analyzer.calibrate(pixel, world)


def test_calibrate_success_with_mask(tmp_path):
    analyzer = CompleteTrafficAnalyzer()
    pixel = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    world = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    H = np.eye(3, dtype=np.float32)
    mask = np.ones((4, 1), dtype=np.uint8)
    with (
        patch("cv2.findHomography", return_value=(H, mask)),
        patch("cv2.perspectiveTransform", side_effect=lambda pts, h: pts.reshape(-1, 2)),
    ):
        H_out, mask_out = analyzer.calibrate(pixel, world)
    assert H_out is not None
    assert mask_out is not None
    assert "final_mae" in analyzer.calibration_metrics


def test_calibrate_success_no_mask():
    analyzer = CompleteTrafficAnalyzer()
    pixel = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    world = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    H = np.eye(3, dtype=np.float32)
    with patch("cv2.findHomography", return_value=(H, None)):
        H_out, mask_out = analyzer.calibrate(pixel, world)
    assert H_out is not None
    assert mask_out is None


# ---------------- _calculate_bev_scale ----------------


def test_calculate_bev_scale_missing_data():
    analyzer = CompleteTrafficAnalyzer()
    # world_points_approx is None, so method returns early
    analyzer._calculate_bev_scale()
    assert analyzer.bev_x_min is None


def test_calculate_bev_scale_valid():
    analyzer = CompleteTrafficAnalyzer()
    analyzer.world_points_approx = np.array([[0, 0], [10, 10]], dtype=np.float32)
    analyzer.inlier_mask = np.array([True, True])
    analyzer.bev_width = 100
    analyzer.bev_height = 100
    analyzer._calculate_bev_scale(safety_margin=0.1)
    assert analyzer.bev_x_min < 0
    assert analyzer.bev_x_max > 10
    assert analyzer.meters_per_pixel_x > 0


# ---------------- validate_bev ----------------


def test_validate_bev_success():
    analyzer = CompleteTrafficAnalyzer()
    analyzer.homography = np.eye(3, dtype=np.float32)
    analyzer.pixel_points = np.array([[0, 0], [100, 0], [0, 100]], dtype=np.float32)
    analyzer.world_points_approx = np.array([[0, 0], [100, 0], [0, 100]], dtype=np.float32)
    analyzer.inlier_mask = np.array([True, True, True])
    result = analyzer.validate_bev()
    assert "mean_error_all" in result
    assert "rmse" in result


def test_validate_bev_requires_calibration():
    analyzer = CompleteTrafficAnalyzer()
    with pytest.raises(RuntimeError):
        analyzer.validate_bev()


# ---------------- estimate_speed ----------------


def test_estimate_speed_too_few_valid_world_positions():
    analyzer = CompleteTrafficAnalyzer()
    analyzer.homography = np.eye(3, dtype=np.float32)
    pixel_positions = np.array([[0, 0], [1, 1]], dtype=np.float32)  # only 2
    frame_times = np.array([0, 1 / 30.0])
    result = analyzer.estimate_speed(pixel_positions, frame_times)
    assert result["final_speed"] == 15.0


def test_estimate_speed_non_finite_positions_skipped():
    analyzer = CompleteTrafficAnalyzer()
    analyzer.homography = np.eye(3, dtype=np.float32)
    pixel_positions = np.array(
        [[0, 0], [np.nan, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]], dtype=np.float32
    )
    frame_times = np.arange(8) / 30.0
    result = analyzer.estimate_speed(pixel_positions, frame_times)
    assert result["final_speed"] == 15.0  # after removing NaN, <5 valid


def test_estimate_speed_insufficient_speeds():
    analyzer = CompleteTrafficAnalyzer()
    analyzer.homography = np.eye(3, dtype=np.float32)
    # 5 valid points but distances/time give speeds outside range -> no speeds collected
    pixel_positions = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.float32)
    frame_times = np.array([0, 1, 2, 3, 4])
    result = analyzer.estimate_speed(pixel_positions, frame_times)
    assert result["final_speed"] == 15.0


# ---------------- save_calibration ----------------


def test_save_calibration(tmp_path):
    analyzer = CompleteTrafficAnalyzer()
    analyzer.homography = np.eye(3, dtype=np.float32)
    analyzer.bev_x_min = 0
    analyzer.bev_x_max = 10
    analyzer.bev_y_min = 0
    analyzer.bev_y_max = 10
    path = tmp_path / "calib.json"
    analyzer.save_calibration(path)
    data = json.loads(path.read_text())
    assert data["homography"] is not None
    assert data["bev_bounds"]["x_max"] == 10


# ---------------- run_video_to_pet ----------------


def make_temp_file(tmp_path, name, content=""):
    p = tmp_path / name
    p.write_text(content)
    return p


def make_dummy_args(tmp_path):
    video = make_temp_file(tmp_path, "video.mp4")
    bev = make_temp_file(tmp_path, "bev.json", "{}")
    grid = make_temp_file(tmp_path, "grid.json", "{}")
    return video, bev, grid


def test_run_video_to_pet_missing_video(tmp_path):
    with pytest.raises(SystemExit):
        run_video_to_pet(
            video_path=tmp_path / "missing.mp4",
            bev_config_path=tmp_path / "bev.json",
            grid_config_path=tmp_path / "grid.json",
        )


def test_run_video_to_pet_sam3_missing_weights(tmp_path):
    video, bev, grid = make_dummy_args(tmp_path)
    mock_module = MagicMock()
    mock_module.run_sam3_grid_pet = MagicMock(return_value=MagicMock(pet_events=[]))
    with patch.dict(sys.modules, {"src.analysis.grid_trajectory.sam3_grid_pet": mock_module}):
        df = run_video_to_pet(
            video_path=video,
            bev_config_path=bev,
            grid_config_path=grid,
            sam3_weights_path=tmp_path / "sam3.pt",
            detector="sam3",
            max_frames=1,
        )
    assert isinstance(df, pd.DataFrame)


def test_run_video_to_pet_yolo_cpu_missing_weights(tmp_path):
    video, bev, grid = make_dummy_args(tmp_path)
    with pytest.raises(FileNotFoundError):
        run_video_to_pet(
            video_path=video,
            bev_config_path=bev,
            grid_config_path=grid,
            yolo_weights_path=tmp_path / "yolo.pt",
            detector="yolo-cpu",
        )


def test_run_video_to_pet_uvh_coco_missing_models(tmp_path):
    video, bev, grid = make_dummy_args(tmp_path)
    with pytest.raises(FileNotFoundError):
        run_video_to_pet(
            video_path=video,
            bev_config_path=bev,
            grid_config_path=grid,
            uvh_model_path=tmp_path / "uvh.pt",
            coco_person_model_path=tmp_path / "coco.pt",
            detector="uvh-coco-fused",
        )


def test_run_video_to_pet_rtdetr_not_implemented(tmp_path):
    video, bev, grid = make_dummy_args(tmp_path)
    rtdetr = make_temp_file(tmp_path, "rtdetr.pt", "dummy")
    with pytest.raises(NotImplementedError):
        run_video_to_pet(
            video_path=video,
            bev_config_path=bev,
            grid_config_path=grid,
            rtdetr_weights_path=rtdetr,
            detector="rtdetr",
        )


def test_run_video_to_pet_empty_events(tmp_path):
    video, bev, grid = make_dummy_args(tmp_path)
    uvh = make_temp_file(tmp_path, "uvh.pt", "dummy")
    coco = make_temp_file(tmp_path, "coco.pt", "dummy")
    out_csv = tmp_path / "out.csv"
    mock_module = MagicMock()
    mock_module.run_uvh_coco_fused_grid_pet = MagicMock(return_value={"pet_events": []})
    with patch.dict(
        sys.modules, {"src.analysis.grid_trajectory.uvh_coco_fused_grid_pet": mock_module}
    ):
        df = run_video_to_pet(
            video_path=video,
            bev_config_path=bev,
            grid_config_path=grid,
            uvh_model_path=uvh,
            coco_person_model_path=coco,
            out_csv_path=out_csv,
            detector="uvh-coco-fused",
        )
    assert df.empty
    assert out_csv.exists()


def test_run_video_to_pet_with_event_dicts(tmp_path):
    video, bev, grid = make_dummy_args(tmp_path)
    uvh = make_temp_file(tmp_path, "uvh.pt", "dummy")
    coco = make_temp_file(tmp_path, "coco.pt", "dummy")
    out_csv = tmp_path / "out.csv"
    events = [
        {
            "pet": 0.5,
            "frame": 10,
            "track_a": 1,
            "track_b": 2,
            "conflict_type": "crossing",
            "grid_cell": "G_A_1",
            "track_a_entry_frame": 5,
            "track_a_exit_frame": 6,
            "track_b_entry_frame": 7,
            "track_b_exit_frame": 8,
            "traj_a_json": "[]",
            "traj_b_json": "[]",
            "gate_a_entry": "G1",
            "gate_b_entry": "G2",
        }
    ]
    mock_module = MagicMock()
    mock_module.run_uvh_coco_fused_grid_pet = MagicMock(return_value={"pet_events": events})
    with patch.dict(
        sys.modules, {"src.analysis.grid_trajectory.uvh_coco_fused_grid_pet": mock_module}
    ):
        df = run_video_to_pet(
            video_path=video,
            bev_config_path=bev,
            grid_config_path=grid,
            uvh_model_path=uvh,
            coco_person_model_path=coco,
            out_csv_path=out_csv,
            detector="uvh-coco-fused",
        )
    assert len(df) == 1
    assert df.iloc[0]["pet"] == 0.5


# ---------------- run_video_to_pet_fixed ----------------


def test_run_video_to_pet_fixed(tmp_path):
    video, bev, grid = make_dummy_args(tmp_path)
    sam3 = make_temp_file(tmp_path, "sam3.pt", "dummy")
    out_csv = tmp_path / "out_fixed.csv"
    fake_result = MagicMock(
        pet_events=[
            {"pet": 1.0, "frame_idx": 1, "track_a": 1, "track_b": 2, "conflict_type": "crossing"}
        ]
    )
    mock_module = MagicMock()
    mock_module.run_sam3_grid_pet = MagicMock(return_value=fake_result)
    with patch.dict(sys.modules, {"src.analysis.grid_trajectory.sam3_grid_pet": mock_module}):
        df = run_video_to_pet_fixed(
            video_path=str(video),
            bev_config_path=str(bev),
            grid_config_path=str(grid),
            sam3_weights_path=str(sam3),
            out_csv_path=str(out_csv),
        )
    assert len(df) == 1
    assert out_csv.exists()


# ---------------- run_pipeline final version ----------------


def test_run_pipeline_unsupported_detector():
    args = SimpleNamespace(detector="invalid", video="dummy")
    with pytest.raises(ValueError):
        run_pipeline(args)


# ---------------- interactive_detector ----------------


class DummyTensor:
    def __init__(self, data):
        self.data = data

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class DummyBoxes:
    def __init__(self):
        self.xyxy = DummyTensor(np.array([[10, 10, 20, 20], [30, 30, 40, 40]], dtype=np.float32))
        self.conf = DummyTensor(np.array([0.9, 0.8], dtype=np.float32))
        self.cls = DummyTensor(np.array([0, 1], dtype=np.int64))

    def __len__(self):
        return len(self.xyxy.data)


class DummyResult:
    def __init__(self):
        self.boxes = DummyBoxes()


class DummyModel:
    def __init__(self):
        self.names = {0: "car", 1: "person"}

    def __call__(self, frame, **kwargs):
        return [DummyResult()]


def test_interactive_detector_full():
    model = DummyModel()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = interactive_detector(frame, model)
    assert len(detections) == 2
    assert detections[0]["cls"] == "car"


def test_interactive_detector_no_results():
    model = MagicMock(return_value=[])
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = interactive_detector(frame, model)
    assert detections == []


# ---------------- additional coverage ----------------


def test_estimate_speed_length_mismatch():
    analyzer = CompleteTrafficAnalyzer()
    analyzer.homography = np.eye(3, dtype=np.float32)
    pixel = np.array([[0, 0], [1, 1]], dtype=np.float32)
    times = np.array([0.0])  # length mismatch
    with pytest.raises(ValueError):
        analyzer.estimate_speed(pixel, times)


def test_estimate_speed_time_mask_fallback():
    analyzer = CompleteTrafficAnalyzer()
    analyzer.homography = np.eye(3, dtype=np.float32)
    # 8 valid pixel positions, but only 4 finite times
    pixel = np.array([[i, i] for i in range(8)], dtype=np.float32)
    times = np.array([0.0, 1.0, 2.0, 3.0, np.nan, np.nan, np.nan, np.nan])
    result = analyzer.estimate_speed(pixel, times)
    assert result["final_speed"] == 15.0
    assert result["speed_std"] == 2.0


def test_run_video_to_pet_yolo_cpu(tmp_path):
    video, bev, grid = make_dummy_args(tmp_path)
    yolo = make_temp_file(tmp_path, "yolo.pt", "dummy")
    out_csv = tmp_path / "out_yolo.csv"
    mock_module = MagicMock()
    mock_module.run_yolo_cpu_grid_pet = MagicMock(return_value={"pet_events": []})
    with patch.dict(sys.modules, {"src.analysis.grid_trajectory.yolo_cpu_grid_pet": mock_module}):
        df = run_video_to_pet(
            video_path=video,
            bev_config_path=bev,
            grid_config_path=grid,
            yolo_weights_path=yolo,
            out_csv_path=out_csv,
            detector="yolo-cpu",
            max_frames=1,
        )
    assert df.empty
    assert out_csv.exists()


def test_run_video_to_pet_sam3_import_error(tmp_path):
    video, bev, grid = make_dummy_args(tmp_path)
    sam3 = make_temp_file(tmp_path, "sam3.pt", "dummy")
    import builtins

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "src.analysis.grid_trajectory.sam3_grid_pet":
            raise ModuleNotFoundError("mock sam3 missing")
        return original_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", side_effect=fake_import):
        with pytest.raises(ModuleNotFoundError):
            run_video_to_pet(
                video_path=video,
                bev_config_path=bev,
                grid_config_path=grid,
                sam3_weights_path=sam3,
                detector="sam3",
            )


def test_run_video_to_pet_uvh_coco_import_error(tmp_path):
    video, bev, grid = make_dummy_args(tmp_path)
    uvh = make_temp_file(tmp_path, "uvh.pt", "dummy")
    coco = make_temp_file(tmp_path, "coco.pt", "dummy")
    import builtins

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "src.analysis.grid_trajectory.uvh_coco_fused_grid_pet":
            raise ModuleNotFoundError("mock fused missing")
        return original_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", side_effect=fake_import):
        with pytest.raises(ModuleNotFoundError):
            run_video_to_pet(
                video_path=video,
                bev_config_path=bev,
                grid_config_path=grid,
                uvh_model_path=uvh,
                coco_person_model_path=coco,
                detector="uvh-coco-fused",
            )
