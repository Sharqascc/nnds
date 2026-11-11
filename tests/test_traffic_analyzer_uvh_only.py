"""Focused contract tests for the UVH-COCO-fused MEP."""

from types import SimpleNamespace
from unittest import mock

import pytest

import src.pipeline.traffic_analyzer as ta


def _fake_uvh_module(return_value):
    return SimpleNamespace(
        run_uvh_coco_fused_grid_pet=mock.MagicMock(return_value=return_value)
    )


def _make_paths(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")

    uvh = tmp_path / "uvh.pt"
    uvh.write_bytes(b"placeholder")

    coco = tmp_path / "coco.pt"
    coco.write_bytes(b"placeholder")

    bev = tmp_path / "bev.json"
    bev.write_text("{}")

    grid = tmp_path / "grid.json"
    grid.write_text("{}")

    return video, uvh, coco, bev, grid


def test_unsupported_detector_fails_immediately(tmp_path):
    video, uvh, coco, bev, grid = _make_paths(tmp_path)

    with pytest.raises(ValueError, match="Only 'uvh-coco-fused' is supported"):
        ta.run_video_to_pet(
            video_path=video,
            bev_config_path=bev,
            grid_config_path=grid,
            detector="sam3",
            uvh_model_path=uvh,
            coco_person_model_path=coco,
            out_csv_path=tmp_path / "pet.csv",
        )


def test_missing_uvh_model_fails_before_inference(tmp_path):
    video, _, coco, bev, grid = _make_paths(tmp_path)
    missing = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError, match="UVH model not found"):
        ta.run_video_to_pet(
            video_path=video,
            bev_config_path=bev,
            grid_config_path=grid,
            detector="uvh-coco-fused",
            uvh_model_path=missing,
            coco_person_model_path=coco,
            out_csv_path=tmp_path / "pet.csv",
        )


def test_missing_coco_model_fails_before_inference(tmp_path):
    video, uvh, _, bev, grid = _make_paths(tmp_path)
    missing = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError, match="COCO person model not found"):
        ta.run_video_to_pet(
            video_path=video,
            bev_config_path=bev,
            grid_config_path=grid,
            detector="uvh-coco-fused",
            uvh_model_path=uvh,
            coco_person_model_path=missing,
            out_csv_path=tmp_path / "pet.csv",
        )


def test_uvh_backend_receives_documented_arguments(tmp_path):
    video, uvh, coco, bev, grid = _make_paths(tmp_path)
    out_csv = tmp_path / "pet.csv"

    fake_module = _fake_uvh_module({"pet_events": []})

    with mock.patch.object(ta, "_import_either", return_value=fake_module):
        ta.run_video_to_pet(
            video_path=video,
            bev_config_path=bev,
            grid_config_path=grid,
            detector="uvh-coco-fused",
            uvh_model_path=uvh,
            coco_person_model_path=coco,
            out_csv_path=out_csv,
            video_source="GITI",
            time_of_day_label="unknown",
            max_frames=50,
        )

    fake_module.run_uvh_coco_fused_grid_pet.assert_called_once()

    kwargs = fake_module.run_uvh_coco_fused_grid_pet.call_args.kwargs

    assert kwargs["video_path"] == str(video)
    assert kwargs["uvh_model_path"] == str(uvh)
    assert kwargs["coco_person_model_path"] == str(coco)
    assert kwargs["video_source"] == "GITI"
    assert kwargs["time_of_day_label"] == "unknown"
    assert kwargs["max_frames"] == 50


def test_empty_events_warn_and_write_csv(tmp_path):
    video, uvh, coco, bev, grid = _make_paths(tmp_path)
    out_csv = tmp_path / "pet.csv"

    fake_module = _fake_uvh_module({"pet_events": []})

    with mock.patch.object(ta, "_import_either", return_value=fake_module):
        with pytest.warns(RuntimeWarning, match="No PET events detected"):
            result = ta.run_video_to_pet(
                video_path=video,
                bev_config_path=bev,
                grid_config_path=grid,
                detector="uvh-coco-fused",
                uvh_model_path=uvh,
                coco_person_model_path=coco,
                out_csv_path=out_csv,
            )

    assert out_csv.exists()
    assert result is not None


def test_event_dicts_are_written(tmp_path):
    video, uvh, coco, bev, grid = _make_paths(tmp_path)
    out_csv = tmp_path / "pet.csv"

    event = {
        "pet": 0.5,
        "frame": 10,
        "track_a": 1,
        "track_b": 2,
        "conflict_type": "side_swipe",
        "grid_cell": "CELL_AG_11",
    }

    fake_module = _fake_uvh_module({"pet_events": [event]})

    with mock.patch.object(ta, "_import_either", return_value=fake_module):
        result = ta.run_video_to_pet(
            video_path=video,
            bev_config_path=bev,
            grid_config_path=grid,
            detector="uvh-coco-fused",
            uvh_model_path=uvh,
            coco_person_model_path=coco,
            out_csv_path=out_csv,
            video_source="GITI",
            time_of_day_label="unknown",
        )

    assert out_csv.exists()
    assert result is not None
