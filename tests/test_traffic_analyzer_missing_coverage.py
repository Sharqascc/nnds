import logging
import runpy
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

import src.pipeline.traffic_analyzer as ta
from src.pipeline.traffic_analyzer import (
    main,
    run_pipeline,
    run_video_to_pet,
    run_video_to_pet_fixed,
)


def _create_dummy_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    return path


def test_uvh_coco_missing_coco_person_model(tmp_path):
    video = _create_dummy_file(tmp_path / "video.mp4")
    bev = _create_dummy_file(tmp_path / "bev.json")
    grid = _create_dummy_file(tmp_path / "grid.json")
    uvh = _create_dummy_file(tmp_path / "uvh.pt")
    # coco_person_model_path is set to nonexistent file
    with pytest.raises(FileNotFoundError, match="COCO person model not found"):
        run_video_to_pet(
            video,
            bev_config_path=bev,
            grid_config_path=grid,
            detector="uvh-coco-fused",
            uvh_model_path=uvh,
            coco_person_model_path=tmp_path / "missing_coco.pt",
        )


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
    monkeypatch.setattr(sys, "argv", ["traffic_analyzer.py", "--demo"])
    # Run the module as if it's the main script to execute the if __name__ block
    runpy.run_module("src.pipeline.traffic_analyzer", run_name="__main__")


def test_run_video_to_pet_fixed_import_error(monkeypatch):
    # Make import fail inside run_video_to_pet_fixed
    original_import = __import__

    def mock_import(name, *args, **kwargs):
        if name.endswith("analysis.grid_trajectory.sam3_grid_pet"):
            raise ModuleNotFoundError(
                "No module named 'src.analysis.grid_trajectory.sam3_grid_pet'"
            )
        return original_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=mock_import):
        with pytest.raises(ModuleNotFoundError):
            run_video_to_pet_fixed("dummy.mp4")


def test_run_pipeline_forwards_args_to_run_video_to_pet(monkeypatch):
    """run_pipeline must invoke run_video_to_pet with args from the namespace.

    Historically this test asserted that run_pipeline returned a dict
    without doing anything (matching the a12d32e stub). That stub was a
    regression; run_pipeline is supposed to actually drive the pipeline.
    """
    from src.pipeline import traffic_analyzer

    captured: dict = {}

    def fake_run_video_to_pet(**kwargs):
        captured.update(kwargs)
        return "fake-result"

    monkeypatch.setattr(traffic_analyzer, "run_video_to_pet", fake_run_video_to_pet)

    args = SimpleNamespace(
        detector="uvh-coco-fused",
        video="dummy.mp4",
        out_csv="out.csv",
        bev_config="bev.json",
        grid_config="grid.json",
        sam3_weights="sam3.pt",
        pet_threshold=2.0,
        max_frames=None,
        yolo_weights="data/models/yolo11n.pt",
        uvh_model="data/models/uvh26.pt",
        coco_person_model="data/models/yolo11n.pt",
        rtdetr_weights="rtdetr-l.pt",
        device="auto",
        max_gap=5,
        max_jump=30.0,
        video_source=None,
        gate_config="configs/gate_config.yaml",
        no_progress=False,
    )
    result = run_pipeline(args)
    assert result == "fake-result"
    assert captured["video_path"] == "dummy.mp4"
    assert captured["out_csv_path"] == "out.csv"
    assert captured["detector"] == "uvh-coco-fused"
    assert captured["device"] == "auto"
    assert captured["max_frame_gap"] == 5
    assert captured["show_progress"] is True
