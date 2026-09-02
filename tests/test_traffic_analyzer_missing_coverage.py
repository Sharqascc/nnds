import logging
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import src.pipeline.traffic_analyzer as ta
from src.pipeline.traffic_analyzer import run_video_to_pet, run_video_to_pet_fixed, run_pipeline, main

def _create_dummy_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    return path

def test_yolo_cpu_import_error(tmp_path):
    video = _create_dummy_file(tmp_path / "video.mp4")
    bev = _create_dummy_file(tmp_path / "bev.json")
    grid = _create_dummy_file(tmp_path / "grid.json")
    yolo = _create_dummy_file(tmp_path / "yolo.pt")
    # Mock import to raise ModuleNotFoundError for yolo_cpu_grid_pet
    original_import = __import__
    def mock_import(name, *args, **kwargs):
        if name == "src.analysis.grid_trajectory.yolo_cpu_grid_pet":
            raise ModuleNotFoundError("No module named 'src.analysis.grid_trajectory.yolo_cpu_grid_pet'")
        return original_import(name, *args, **kwargs)
    with mock.patch("builtins.__import__", side_effect=mock_import):
        with pytest.raises(ModuleNotFoundError):
            run_video_to_pet(video, bev_config_path=bev, grid_config_path=grid,
                             detector="yolo-cpu", yolo_weights_path=yolo)

def test_uvh_coco_missing_coco_person_model(tmp_path):
    video = _create_dummy_file(tmp_path / "video.mp4")
    bev = _create_dummy_file(tmp_path / "bev.json")
    grid = _create_dummy_file(tmp_path / "grid.json")
    uvh = _create_dummy_file(tmp_path / "uvh.pt")
    # coco_person_model_path is set to nonexistent file
    with pytest.raises(FileNotFoundError, match="COCO person model not found"):
        run_video_to_pet(video, bev_config_path=bev, grid_config_path=grid,
                         detector="uvh-coco-fused", uvh_model_path=uvh,
                         coco_person_model_path=tmp_path / "missing_coco.pt")

def test_rtdetr_missing_weights(tmp_path):
    video = _create_dummy_file(tmp_path / "video.mp4")
    bev = _create_dummy_file(tmp_path / "bev.json")
    grid = _create_dummy_file(tmp_path / "grid.json")
    with pytest.raises(FileNotFoundError, match="RT-DETR weights not found"):
        run_video_to_pet(video, bev_config_path=bev, grid_config_path=grid,
                         detector="rtdetr", rtdetr_weights_path=tmp_path / "missing_rtdetr.pt")

def test_run_video_to_pet_object_event_parsing(tmp_path, monkeypatch):
    video = _create_dummy_file(tmp_path / "video.mp4")
    bev = _create_dummy_file(tmp_path / "bev.json")
    grid = _create_dummy_file(tmp_path / "grid.json")
    sam3 = _create_dummy_file(tmp_path / "sam3.pt")
    out = tmp_path / "out.csv"

    # Create fake event objects with attributes
    class EventObj:
        pass

    events = []
    # Object with attributes that trigger _get else branch and _parse_track_id scenarios
    e = EventObj()
    e.track_a = None
    e.track_b = -1
    e.frame = 10
    e.PET = 1.5
    e.conflict_type = "crossing"
    e.grid_cell = "A1"
    e.track_a_entry_frame = 1
    e.track_a_exit_frame = 2
    e.track_b_entry_frame = 3
    e.track_b_exit_frame = 4
    e.world_traj_i = None
    e.world_traj_j = None
    e.traj_a_json = "[]"
    e.traj_b_json = "[]"
    e.gate_a_entry = "left"
    e.gate_b_entry = "right"
    events.append(e)

    # Another event with string track IDs
    e2 = EventObj()
    e2.track_a = "track_17"
    e2.track_b = "track_23"
    e2.frame = 11
    e2.PET = 2.0
    e2.conflict_type = "rear-end"
    e2.grid_cell = "B2"
    e2.track_a_entry_frame = 5
    e2.track_a_exit_frame = 6
    e2.track_b_entry_frame = 7
    e2.track_b_exit_frame = 8
    e2.world_traj_i = [4, 5]
    e2.world_traj_j = [6, 7]
    e2.traj_a_json = "[]"
    e2.traj_b_json = "[]"
    e2.gate_a_entry = "unknown"
    e2.gate_b_entry = "unknown"
    events.append(e2)

    class FakeResult:
        pet_events = events

    # Patch sam3_grid_pet.run_sam3_grid_pet to return fake result
    with mock.patch("src.analysis.grid_trajectory.sam3_grid_pet.run_sam3_grid_pet", return_value=FakeResult()):
        df = run_video_to_pet(video, bev_config_path=bev, grid_config_path=grid,
                              detector="sam3", sam3_weights_path=sam3,
                              out_csv_path=out)
    assert len(df) == 2
    assert df.iloc[0]["track_a"] == -1
    assert df.iloc[0]["track_b"] == -1
    assert df.iloc[1]["track_a"] == 17
    assert df.iloc[1]["track_b"] == 23

def test_main_logging_basic_config(monkeypatch):
    # Ensure no handlers exist
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    monkeypatch.setattr(ta, "parse_args", lambda: SimpleNamespace(demo=True))
    monkeypatch.setattr(ta, "run_demo", lambda: None)
    # Should not raise SystemExit
    main()

def test_main_if_name_main_block(monkeypatch):
    # Set sys.argv so that main() runs in demo mode and does not SystemExit
    monkeypatch.setattr(sys, 'argv', ['traffic_analyzer.py', '--demo'])
    # Run the module as if it's the main script to execute the if __name__ block
    runpy.run_module("src.pipeline.traffic_analyzer", run_name="__main__")

def test_run_video_to_pet_fixed_import_error(monkeypatch):
    # Make import fail inside run_video_to_pet_fixed
    original_import = __import__
    def mock_import(name, *args, **kwargs):
        if name == "src.analysis.grid_trajectory.sam3_grid_pet":
            raise ModuleNotFoundError("No module named 'src.analysis.grid_trajectory.sam3_grid_pet'")
        return original_import(name, *args, **kwargs)
    with mock.patch("builtins.__import__", side_effect=mock_import):
        with pytest.raises(ModuleNotFoundError):
            run_video_to_pet_fixed("dummy.mp4")

def test_run_pipeline_sam3_returns_dict():
    args = SimpleNamespace(detector="sam3", video="dummy.mp4", out_csv="out.csv")
    result = run_pipeline(args)
    assert result == {"video": "dummy.mp4", "out_csv": "out.csv"}
