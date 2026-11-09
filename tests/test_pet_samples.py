"""Tests for pet_samples.render_pet_samples and the --samples CLI flag."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd
import pytest

from src.analysis.visualization.pet_samples import render_pet_samples


def _traj(frames=10, x0=100, y0=200, dx=10):
    return json.dumps(
        [
            {"frame": f, "x_pixel": x0 + f * dx, "y_pixel": y0, "world_x": 0.0, "world_y": 0.0}
            for f in range(frames)
        ]
    )


@pytest.fixture
def synthetic_video(tmp_path):
    path = tmp_path / "video.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480))
    for i in range(10):
        frame = np.full((480, 640, 3), 60, dtype=np.uint8)
        x = 100 + i * 20
        cv2.rectangle(frame, (x, 200), (x + 80, 280), (0, 128, 255), -1)
        vw.write(frame)
    vw.release()
    assert path.exists() and path.stat().st_size > 500
    return path


@pytest.fixture
def synthetic_detections(tmp_path):
    rows = []
    for f in range(10):
        rows.append(
            {
                "frame": f,
                "track_id": 1,
                "x1": 100 + f * 20,
                "y1": 200,
                "x2": 180 + f * 20,
                "y2": 280,
                "class_name": "car",
                "conf": 0.9,
                "cx": 140 + f * 20,
                "cy": 240,
                "class_id": 0,
                "source": "test",
            }
        )
        rows.append(
            {
                "frame": f,
                "track_id": 2,
                "x1": 300 - f * 20,
                "y1": 220,
                "x2": 380 - f * 20,
                "y2": 300,
                "class_name": "car",
                "conf": 0.9,
                "cx": 340 - f * 20,
                "cy": 260,
                "class_id": 0,
                "source": "test",
            }
        )
    path = tmp_path / "detections.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _event_row(pet, frame, ta=1, tb=2, grid="CELL_A", ctype="side_swipe"):
    return {
        "pet": pet,
        "frame": frame,
        "track_a": ta,
        "track_b": tb,
        "conflict_type": ctype,
        "grid_cell": grid,
        "track_a_entry_frame": 0,
        "track_a_exit_frame": 5,
        "track_b_entry_frame": 3,
        "track_b_exit_frame": 8,
        "track_a_exit_time_sec": 0.1,
        "track_b_entry_time_sec": 0.1,
        "track_b_exit_time_sec": 0.2,
        "world_traj_i": f"track_{ta}",
        "world_traj_j": f"track_{tb}",
        "traj_a_json": _traj(),
        "traj_b_json": _traj(),
        "video_source": "test.mp4",
        "time_of_day_label": "unknown",
        "gate_a_entry": "unknown",
        "gate_b_entry": "unknown",
    }


@pytest.fixture
def synthetic_events(tmp_path):
    path = tmp_path / "events.csv"
    pd.DataFrame(
        [
            _event_row(1.2, 3, grid="CELL_X_0", ctype="side_swipe"),
            _event_row(0.4, 5, grid="CELL_X_1", ctype="rear_end"),
            _event_row(0.9, 7, grid="CELL_X_2", ctype="side_swipe"),
        ]
    ).to_csv(path, index=False)
    return path


def test_writes_n_images(synthetic_video, synthetic_detections, synthetic_events, tmp_path):
    out_dir = tmp_path / "samples"
    paths = render_pet_samples(
        events_csv=synthetic_events,
        detections_csv=synthetic_detections,
        video_path=synthetic_video,
        out_dir=out_dir,
        n=2,
    )
    assert len(paths) == 2
    for p in paths:
        assert p.exists()
        assert p.suffix == ".jpg"
        assert p.stat().st_size > 3000


def test_sorts_by_pet_ascending(
    synthetic_video,
    synthetic_detections,
    synthetic_events,
    tmp_path,
):
    paths = render_pet_samples(
        events_csv=synthetic_events,
        detections_csv=synthetic_detections,
        video_path=synthetic_video,
        out_dir=tmp_path / "samples",
        n=3,
    )
    assert len(paths) == 3
    assert "pet0.400" in paths[0].name
    assert "pet0.900" in paths[1].name
    assert "pet1.200" in paths[2].name


def test_n_larger_than_events(
    synthetic_video,
    synthetic_detections,
    tmp_path,
):
    events = tmp_path / "events.csv"
    pd.DataFrame([_event_row(0.5, 3)]).to_csv(events, index=False)
    paths = render_pet_samples(
        events_csv=events,
        detections_csv=synthetic_detections,
        video_path=synthetic_video,
        out_dir=tmp_path / "samples",
        n=10,
    )
    assert len(paths) == 1


def test_empty_events_returns_empty(
    tmp_path,
    synthetic_video,
    synthetic_detections,
):
    empty = tmp_path / "empty.csv"
    pd.DataFrame(columns=["pet", "frame", "track_a", "track_b"]).to_csv(
        empty,
        index=False,
    )
    paths = render_pet_samples(
        events_csv=empty,
        detections_csv=synthetic_detections,
        video_path=synthetic_video,
        out_dir=tmp_path / "samples",
        n=5,
    )
    assert paths == []


def test_missing_events_csv_raises(
    tmp_path,
    synthetic_video,
    synthetic_detections,
):
    with pytest.raises(FileNotFoundError):
        render_pet_samples(
            events_csv=tmp_path / "nope.csv",
            detections_csv=synthetic_detections,
            video_path=synthetic_video,
            out_dir=tmp_path / "samples",
        )


def test_missing_video_raises(tmp_path, synthetic_events, synthetic_detections):
    with pytest.raises(FileNotFoundError):
        render_pet_samples(
            events_csv=synthetic_events,
            detections_csv=synthetic_detections,
            video_path=tmp_path / "nope.mp4",
            out_dir=tmp_path / "samples",
        )


def test_missing_column_raises(tmp_path, synthetic_video, synthetic_detections):
    bad = tmp_path / "bad.csv"
    pd.DataFrame([{"pet": 0.5, "frame": 3}]).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing required column"):
        render_pet_samples(
            events_csv=bad,
            detections_csv=synthetic_detections,
            video_path=synthetic_video,
            out_dir=tmp_path / "samples",
        )


def test_missing_bbox_for_one_track_still_renders(
    synthetic_video,
    synthetic_detections,
    tmp_path,
):
    events = tmp_path / "events.csv"
    row = _event_row(0.5, 3, ta=99, tb=2)  # 99 not in detections
    pd.DataFrame([row]).to_csv(events, index=False)
    paths = render_pet_samples(
        events_csv=events,
        detections_csv=synthetic_detections,
        video_path=synthetic_video,
        out_dir=tmp_path / "samples",
        n=5,
    )
    assert len(paths) == 1


def test_cli_flag_default_zero():
    from src.pipeline.traffic_analyzer import parse_args

    args = parse_args(["--video", "dummy.mp4"])
    assert getattr(args, "samples", None) == 0


def test_cli_flag_accepts_n():
    from src.pipeline.traffic_analyzer import parse_args

    args = parse_args(["--video", "dummy.mp4", "--samples", "7"])
    assert args.samples == 7


# ── New tests for the 4 enhancements (grid, event ID, entry/exit, PET) ──


def test_renders_with_grid_config_when_available(
    synthetic_video,
    synthetic_detections,
    synthetic_events,
    tmp_path,
):
    """If the repo's grid config exists, rendering with it should succeed."""
    from pathlib import Path as _P

    repo_root = _P(__file__).resolve().parents[1]
    grid_cfg = repo_root / "configs" / "GITI_grid_config.json"
    if not grid_cfg.exists():
        pytest.skip("GITI grid config not present in this checkout")
    paths = render_pet_samples(
        events_csv=synthetic_events,
        detections_csv=synthetic_detections,
        video_path=synthetic_video,
        out_dir=tmp_path / "samples",
        n=2,
        grid_config_path=str(grid_cfg),
    )
    assert len(paths) == 2
    for p in paths:
        assert p.exists() and p.stat().st_size > 3000


def test_renders_with_invalid_grid_config_gracefully(
    synthetic_video,
    synthetic_detections,
    synthetic_events,
    tmp_path,
):
    """Invalid grid config must not crash; images still produced."""
    bad_cfg = tmp_path / "bad_grid.json"
    bad_cfg.write_text("{ not valid json")
    paths = render_pet_samples(
        events_csv=synthetic_events,
        detections_csv=synthetic_detections,
        video_path=synthetic_video,
        out_dir=tmp_path / "samples",
        n=2,
        grid_config_path=str(bad_cfg),
    )
    assert len(paths) == 2


def test_renders_without_grid_config(
    synthetic_video,
    synthetic_detections,
    synthetic_events,
    tmp_path,
):
    """Explicit None grid_config_path: no overlay, still renders."""
    paths = render_pet_samples(
        events_csv=synthetic_events,
        detections_csv=synthetic_detections,
        video_path=synthetic_video,
        out_dir=tmp_path / "samples",
        n=2,
        grid_config_path=None,
    )
    assert len(paths) == 2


def test_renders_when_all_entry_exit_fields_nan(
    synthetic_video,
    synthetic_detections,
    tmp_path,
):
    """Events with NaN in all entry/exit fields still render (panel omits lines)."""
    events = tmp_path / "events.csv"
    row = _event_row(0.5, 3)
    row["track_a_exit_frame"] = float("nan")
    row["track_b_entry_frame"] = float("nan")
    row["track_a_exit_time_sec"] = float("nan")
    row["track_b_entry_time_sec"] = float("nan")
    pd.DataFrame([row]).to_csv(events, index=False)
    paths = render_pet_samples(
        events_csv=events,
        detections_csv=synthetic_detections,
        video_path=synthetic_video,
        out_dir=tmp_path / "samples",
        n=5,
    )
    assert len(paths) == 1
    assert paths[0].stat().st_size > 3000


def test_renders_when_entry_exit_fields_present(
    synthetic_video,
    synthetic_detections,
    synthetic_events,
    tmp_path,
):
    """Events with real entry/exit fields trigger marker + panel lines."""
    paths = render_pet_samples(
        events_csv=synthetic_events,
        detections_csv=synthetic_detections,
        video_path=synthetic_video,
        out_dir=tmp_path / "samples",
        n=2,
    )
    assert len(paths) == 2


def test_renders_with_custom_pet_arithmetic_columns(
    synthetic_video,
    synthetic_detections,
    tmp_path,
):
    """When both times present, panel shows arithmetic; image size grows."""
    events = tmp_path / "events.csv"
    row = _event_row(0.5, 3)
    row["track_a_exit_time_sec"] = 4.5
    row["track_b_entry_time_sec"] = 5.0
    pd.DataFrame([row]).to_csv(events, index=False)
    paths = render_pet_samples(
        events_csv=events,
        detections_csv=synthetic_detections,
        video_path=synthetic_video,
        out_dir=tmp_path / "samples",
        n=1,
    )
    assert len(paths) == 1


def test_cli_passes_grid_config_to_samples(monkeypatch, tmp_path):
    """traffic_analyzer._render_samples passes args.grid_config through."""
    from src.pipeline import traffic_analyzer

    captured = {}

    class FakeModule:
        def render_pet_samples(self, **kwargs):
            captured.update(kwargs)
            return []

    # Fake _import_either
    monkeypatch.setattr(
        traffic_analyzer,
        "_import_either",
        lambda name: FakeModule(),
    )

    # Build minimal args
    class A:
        out_csv = str(tmp_path / "pet.csv")
        video = str(tmp_path / "v.mp4")
        samples = 3
        grid_config = "some/grid.json"

    # Create the "CSVs" so the existence check passes
    (tmp_path / "pet.csv").write_text("pet,frame,track_a,track_b\n")
    (tmp_path / "pet_detections.csv").write_text("frame,track_id,x1,y1,x2,y2\n")
    (tmp_path / "v.mp4").write_bytes(b"x")

    traffic_analyzer._render_samples(A())
    assert captured.get("grid_config_path") == "some/grid.json"
    assert captured.get("n") == 3
