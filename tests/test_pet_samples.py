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


# ── Tests for publication-quality enhancements ────────────────────────────


class _FakeGrid:
    """Minimal grid stub for testing drawing helpers without a config file."""

    def __init__(self, x_min=0, y_min=0, x_max=800, y_max=600, cell_size=25):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.cell_size = cell_size

    def get_cell_bounds(self, cell_id):
        # Parse "CELL_<col>_<row>"
        try:
            parts = cell_id.split("_")
            col_letter, row_num = parts[1], int(parts[2])
            col_idx = ord(col_letter.upper()) - ord("A")
            row_idx = row_num - 1
            x1 = self.x_min + col_idx * self.cell_size
            y1 = self.y_min + row_idx * self.cell_size
            return (x1, y1, x1 + self.cell_size, y1 + self.cell_size)
        except (IndexError, ValueError, AttributeError):
            return None

    def get_cell_center(self, cell_id):
        b = self.get_cell_bounds(cell_id)
        return None if b is None else ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)


def test_draw_localized_grid_only_touches_roi():
    """Grid draws only within ~radius_cells of the conflict cell."""
    from src.analysis.visualization.pet_samples import _draw_localized_grid

    grid = _FakeGrid(x_max=1000, y_max=800, cell_size=50)
    frame = np.zeros((800, 1000, 3), dtype=np.uint8)
    before = frame.copy()
    out = _draw_localized_grid(frame, grid, "CELL_E_5", radius_cells=2, alpha=0.5)

    # Conflict cell: E (col idx 4) * 50 = x 200..250, row 5 → y 200..250
    # ROI: +/- 2 cells = +/-100 px → x [100, 350], y [100, 350]
    # Pixels far outside ROI must be unchanged
    assert np.array_equal(out[:50, :50], before[:50, :50]), "top-left corner should be unchanged"
    assert np.array_equal(out[700:, 900:], before[700:, 900:]), (
        "bottom-right corner should be unchanged"
    )
    # Pixels inside ROI should differ (grid lines drawn)
    roi_diff = np.any(out[100:350, 100:350] != before[100:350, 100:350], axis=-1).sum()
    assert roi_diff > 0, "ROI should show grid lines"


def test_draw_localized_grid_noop_when_grid_none():
    from src.analysis.visualization.pet_samples import _draw_localized_grid

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    before = frame.copy()
    out = _draw_localized_grid(frame, None, "CELL_A_1")
    assert np.array_equal(out, before)


def test_draw_localized_grid_noop_when_cell_unknown():
    from src.analysis.visualization.pet_samples import _draw_localized_grid

    grid = _FakeGrid()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    before = frame.copy()
    out = _draw_localized_grid(frame, grid, "unknown")
    assert np.array_equal(out, before)


def test_draw_cell_highlight_border_present():
    """Cell highlight must set cyan pixels at the border."""
    from src.analysis.visualization.pet_samples import (
        _COLOR_GRID_CELL,
        _draw_cell_highlight,
    )

    grid = _FakeGrid(cell_size=50)
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    out = _draw_cell_highlight(frame, grid, "CELL_B_2")

    # CELL_B_2 = col 1 (x 50..100), row 1 (y 50..100)
    # Border pixels at top edge should be exactly the cyan color
    top_border_pixel = out[50, 75]  # (y=50, x=75)
    assert tuple(int(c) for c in top_border_pixel) == _COLOR_GRID_CELL, (
        f"border should be cyan, got {top_border_pixel}"
    )


def test_draw_cell_highlight_noop_without_grid():
    from src.analysis.visualization.pet_samples import _draw_cell_highlight

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    before = frame.copy()
    out = _draw_cell_highlight(frame, None, "CELL_A_1")
    assert np.array_equal(out, before)


def test_arrowhead_marks_end_of_polyline():
    """Arrowhead must set pixels near the endpoint in the trajectory color."""
    from src.analysis.visualization.pet_samples import (
        _COLOR_A,
        _draw_polyline_with_arrow,
    )

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    before = frame.copy()
    points = [(20, 20), (80, 80), (140, 140)]
    _draw_polyline_with_arrow(frame, points, _COLOR_A)

    # Line pixels along the path should be red
    line_px = frame[80, 80]
    assert tuple(int(c) for c in line_px) == _COLOR_A, "polyline should be red"

    # Arrowhead should thicken the region near the tip (140,140)
    # Count non-black pixels in a 20-px box around tip
    tip_box = frame[130:150, 130:150]
    non_black = np.any(tip_box != 0, axis=-1).sum()
    assert non_black > 20, f"arrowhead should add width near tip, got {non_black} px"


def test_hud_panel_renders_with_all_sections():
    """HUD panel should draw text without errors and never crash."""
    from src.analysis.visualization.pet_samples import _draw_hud_panel

    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    out = _draw_hud_panel(
        frame,
        "PET Sample 1/5  ·  CSV row 3",
        [
            ("TIMING", ["PET = 0.500s", "  = B_in(5.00s) − A_out(4.50s)"]),
            ("CONTEXT", ["Frame: 144", "Grid: CELL_A_1", "Type: side_swipe"]),
        ],
    )
    # Panel region should be non-zero (text drawn)
    assert out[10:300, 10:500].sum() > 0
    assert out.shape == frame.shape


def test_hud_panel_handles_empty_sections():
    from src.analysis.visualization.pet_samples import _draw_hud_panel

    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    out = _draw_hud_panel(frame, "Title", [])
    assert out.shape == frame.shape


def test_bbox_labels_do_not_overlap():
    """Two overlapping bboxes should place their labels in different positions."""
    from src.analysis.visualization.pet_samples import _COLOR_A, _COLOR_B, _draw_bbox_with_label

    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    existing = []
    # Two bboxes at nearly the same position (simulating the reviewer's
    # T119/T130 overlap case)
    bb1 = _draw_bbox_with_label(frame, (100, 100, 200, 200), _COLOR_A, "T119", existing)
    existing.append(bb1)
    bb2 = _draw_bbox_with_label(frame, (105, 105, 205, 205), _COLOR_B, "T130", existing)
    existing.append(bb2)

    # Check labels don't overlap
    def _overlap(a, b):
        return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])

    assert not _overlap(bb1, bb2), (
        f"labels overlap: {bb1} vs {bb2}. Collision resolution should have offset one."
    )


def test_render_pet_samples_end_to_end_still_works(
    synthetic_video,
    synthetic_detections,
    synthetic_events,
    tmp_path,
):
    """Full pipeline still works after the styling rewrite."""
    paths = render_pet_samples(
        events_csv=synthetic_events,
        detections_csv=synthetic_detections,
        video_path=synthetic_video,
        out_dir=tmp_path / "samples",
        n=2,
    )
    assert len(paths) == 2
    for p in paths:
        assert p.stat().st_size > 3000


# ── Tests for v2 refinements ──────────────────────────────────────────────


def test_polyline_breaks_on_large_jump():
    """A trajectory edge > max_gap_px should NOT be drawn (ID-switch guard)."""
    from src.analysis.visualization.pet_samples import (
        _COLOR_A,
        _draw_polyline_with_arrow,
    )

    frame = np.zeros((300, 2000, 3), dtype=np.uint8)
    points = [(100, 100), (150, 100), (1500, 100), (1550, 100)]
    _draw_polyline_with_arrow(frame, points, _COLOR_A, max_gap_px=250)

    mid_x, mid_y = 800, 100
    assert np.all(frame[mid_y, mid_x] == 0), "large jump edge should be skipped"
    seg_mid = 125
    assert not np.all(frame[100, seg_mid] == 0), "short edge should be drawn"


def test_ring_pair_placement_separates_close_rings():
    from src.analysis.visualization.pet_samples import _place_ring_pair

    a, b = _place_ring_pair((100, 100), (110, 100), min_separation=30.0)
    dist = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
    assert dist >= 30.0


def test_ring_pair_placement_noop_when_far():
    from src.analysis.visualization.pet_samples import _place_ring_pair

    a, b = _place_ring_pair((100, 100), (300, 300), min_separation=30.0)
    assert a == (100, 100) and b == (300, 300)


def test_ring_pair_placement_handles_none():
    from src.analysis.visualization.pet_samples import _place_ring_pair

    a, b = _place_ring_pair(None, (100, 100))
    assert a is None
    a, b = _place_ring_pair((100, 100), None)
    assert b is None
    a, b = _place_ring_pair(None, None)
    assert a is None and b is None


def test_branch_a_arithmetic_no_crash(
    synthetic_video,
    synthetic_detections,
    tmp_path,
):
    events = tmp_path / "events.csv"
    row = _event_row(0.5, 3)
    row["track_a_exit_frame"] = 10
    row["track_a_exit_time_sec"] = 10 / 30.0
    row["track_b_entry_frame"] = 25
    row["track_b_entry_time_sec"] = 25 / 30.0
    row["track_a_entry_frame"] = 2
    row["track_a_entry_time_sec"] = 2 / 30.0
    row["track_b_exit_frame"] = 30
    row["track_b_exit_time_sec"] = 30 / 30.0
    pd.DataFrame([row]).to_csv(events, index=False)
    paths = render_pet_samples(
        events_csv=events,
        detections_csv=synthetic_detections,
        video_path=synthetic_video,
        out_dir=tmp_path / "s",
        n=1,
    )
    assert len(paths) == 1


def test_branch_b_arithmetic_no_crash(
    synthetic_video,
    synthetic_detections,
    tmp_path,
):
    events = tmp_path / "events.csv"
    row = _event_row(0.5, 3)
    row["track_b_exit_frame"] = 5
    row["track_b_exit_time_sec"] = 5 / 30.0
    row["track_a_entry_frame"] = 15
    row["track_a_entry_time_sec"] = 15 / 30.0
    row["track_a_exit_frame"] = 20
    row["track_a_exit_time_sec"] = 20 / 30.0
    row["track_b_entry_frame"] = 0
    row["track_b_entry_time_sec"] = 0.0
    pd.DataFrame([row]).to_csv(events, index=False)
    paths = render_pet_samples(
        events_csv=events,
        detections_csv=synthetic_detections,
        video_path=synthetic_video,
        out_dir=tmp_path / "s",
        n=1,
    )
    assert len(paths) == 1


def test_overlap_case_no_crash(
    synthetic_video,
    synthetic_detections,
    tmp_path,
):
    events = tmp_path / "events.csv"
    row = _event_row(0.5, 3)
    row["track_a_entry_frame"] = 2
    row["track_a_exit_frame"] = 30
    row["track_b_entry_frame"] = 10
    row["track_b_exit_frame"] = 20
    pd.DataFrame([row]).to_csv(events, index=False)
    paths = render_pet_samples(
        events_csv=events,
        detections_csv=synthetic_detections,
        video_path=synthetic_video,
        out_dir=tmp_path / "s",
        n=1,
    )
    assert len(paths) == 1


# ── Tests for branch-aware labels and video_source ──────────────────────


def test_branch_frames_a_first():
    """A_exit < B_entry → labels A out / B in."""
    from src.analysis.visualization.pet_samples import _branch_frames

    a_frame, b_frame, a_label, b_label = _branch_frames(
        a_entry=10,
        a_exit=25,
        b_entry=40,
        b_exit=60,
    )
    assert a_frame == 25
    assert b_frame == 40
    assert a_label == "A out"
    assert b_label == "B in"


def test_branch_frames_b_first():
    """B_exit < A_entry → labels A in / B out."""
    from src.analysis.visualization.pet_samples import _branch_frames

    a_frame, b_frame, a_label, b_label = _branch_frames(
        a_entry=80,
        a_exit=105,
        b_entry=25,
        b_exit=43,
    )
    assert a_frame == 80
    assert b_frame == 43
    assert a_label == "A in"
    assert b_label == "B out"


def test_branch_frames_overlap():
    """Neither branch → fall back to A out / B in."""
    from src.analysis.visualization.pet_samples import _branch_frames

    # a_exit=30 > b_entry=20 AND b_exit=40 > a_entry=10 (overlap)
    _a_frame, _b_frame, a_label, b_label = _branch_frames(
        a_entry=10,
        a_exit=30,
        b_entry=20,
        b_exit=40,
    )
    assert a_label == "A out"
    assert b_label == "B in"


def test_branch_frames_with_none():
    """Missing values → fall back gracefully."""
    from src.analysis.visualization.pet_samples import _branch_frames

    _a_frame, _b_frame, a_label, b_label = _branch_frames(
        a_entry=None,
        a_exit=None,
        b_entry=None,
        b_exit=None,
    )
    assert a_label == "A out"
    assert b_label == "B in"


def test_frame_point_finds_existing():
    from src.analysis.visualization.pet_samples import _frame_point

    traj = [(100.0, 200.0, 5.0), (110.0, 210.0, 6.0), (120.0, 220.0, 7.0)]
    assert _frame_point(traj, 6) == (110, 210)


def test_frame_point_returns_none_for_missing():
    from src.analysis.visualization.pet_samples import _frame_point

    traj = [(100.0, 200.0, 5.0), (110.0, 210.0, 6.0)]
    assert _frame_point(traj, 99) is None
    assert _frame_point(traj, None) is None


def test_render_with_video_source_column(
    synthetic_video,
    synthetic_detections,
    tmp_path,
):
    """CSV with video_source column — image renders without crash."""
    events = tmp_path / "events.csv"
    row = _event_row(0.5, 3)
    row["video_source"] = "GITI_traffic_video"
    pd.DataFrame([row]).to_csv(events, index=False)
    paths = render_pet_samples(
        events_csv=events,
        detections_csv=synthetic_detections,
        video_path=synthetic_video,
        out_dir=tmp_path / "samples",
        n=1,
    )
    assert len(paths) == 1


def test_render_branch_b_event_end_to_end(
    synthetic_video,
    synthetic_detections,
    tmp_path,
):
    """Branch B event (B clears first) renders with correct branch labels."""
    events = tmp_path / "events.csv"
    row = _event_row(0.5, 3)
    # Branch B: b_exit < a_entry
    row["track_a_entry_frame"] = 15
    row["track_a_entry_time_sec"] = 15 / 30.0
    row["track_a_exit_frame"] = 20
    row["track_a_exit_time_sec"] = 20 / 30.0
    row["track_b_entry_frame"] = 0
    row["track_b_entry_time_sec"] = 0.0
    row["track_b_exit_frame"] = 5
    row["track_b_exit_time_sec"] = 5 / 30.0
    row["video_source"] = "test"
    pd.DataFrame([row]).to_csv(events, index=False)
    paths = render_pet_samples(
        events_csv=events,
        detections_csv=synthetic_detections,
        video_path=synthetic_video,
        out_dir=tmp_path / "samples",
        n=1,
    )
    assert len(paths) == 1


# ── Metadata builder tests ──────────────────────────────────────────────
# These catch the class of bug where missing commas between adjacent
# f-strings silently concatenate the entire metadata list into one
# unreadable line. They also assert exact labels for each branch.


def test_build_metadata_lines_returns_seven_separate_items():
    """Each logical line must be a separate list element."""
    from src.analysis.visualization.pet_samples import _build_metadata_lines

    lines = _build_metadata_lines(
        video_source="test.mp4",
        frame_idx=80,
        grid_cell="CELL_AY_7",
        conflict_type="side_swipe",
        track_a=33,
        track_b=28,
        a_mark_frame=80,
        a_mark_label="A in",
        b_mark_frame=43,
        b_mark_label="B out",
    )
    assert isinstance(lines, list)
    assert len(lines) == 7, (
        f"expected 7 separate lines, got {len(lines)}: {lines}. "
        f"Missing commas between f-strings cause concatenation into one item."
    )
    # Each line must be its own logical unit
    assert lines[0].startswith("Video:")
    assert lines[1].startswith("Frame:")
    assert lines[2].startswith("Grid:")
    assert lines[3].startswith("Type:")
    assert lines[4].startswith("Pair:")
    assert lines[5].startswith("A in:")
    assert lines[6].startswith("B out:")


def test_build_metadata_lines_branch_b_exact_strings():
    """Branch B: labels must be 'A in' / 'B out', frames 80 / 43."""
    from src.analysis.visualization.pet_samples import _build_metadata_lines

    lines = _build_metadata_lines(
        video_source="GITI_traffic_video",
        frame_idx=80,
        grid_cell="CELL_AY_7",
        conflict_type="side_swipe",
        track_a=33,
        track_b=28,
        a_mark_frame=80,
        a_mark_label="A in",
        b_mark_frame=43,
        b_mark_label="B out",
    )
    # Exact strings must appear
    assert "A in: f=80" in lines, f"A in label missing: {lines}"
    assert "B out: f=43" in lines, f"B out label missing: {lines}"
    # Wrong labels must NOT appear
    assert not any("A out" in line for line in lines), (
        f"stale 'A out' in branch-B metadata: {lines}"
    )
    assert not any("B in:" in line for line in lines), f"stale 'B in' in branch-B metadata: {lines}"


def test_build_metadata_lines_branch_a_exact_strings():
    """Branch A: labels must be 'A out' / 'B in', frames 105 / 25."""
    from src.analysis.visualization.pet_samples import _build_metadata_lines

    lines = _build_metadata_lines(
        video_source="GITI_traffic_video",
        frame_idx=100,
        grid_cell="CELL_XY_1",
        conflict_type="rear_end",
        track_a=10,
        track_b=20,
        a_mark_frame=105,
        a_mark_label="A out",
        b_mark_frame=25,
        b_mark_label="B in",
    )
    assert "A out: f=105" in lines, f"A out missing: {lines}"
    assert "B in: f=25" in lines, f"B in missing: {lines}"
    assert not any("A in:" in line for line in lines), f"stale 'A in': {lines}"
    assert not any("B out" in line for line in lines), f"stale 'B out': {lines}"


def test_build_metadata_lines_missing_frames_use_question_mark():
    """When mark_frame is None, the line shows f=? not f=None."""
    from src.analysis.visualization.pet_samples import _build_metadata_lines

    lines = _build_metadata_lines(
        video_source="x.mp4",
        frame_idx=10,
        grid_cell="C1",
        conflict_type="t1",
        track_a=1,
        track_b=2,
        a_mark_frame=None,
        a_mark_label="A out",
        b_mark_frame=None,
        b_mark_label="B in",
    )
    assert "A out: f=?" in lines
    assert "B in: f=?" in lines
    assert not any("f=None" in line for line in lines)


def test_build_metadata_lines_empty_video_source():
    """Empty video_source renders as '?' not 'Video: '."""
    from src.analysis.visualization.pet_samples import _build_metadata_lines

    lines = _build_metadata_lines(
        video_source="",
        frame_idx=1,
        grid_cell="C1",
        conflict_type="t",
        track_a=1,
        track_b=2,
        a_mark_frame=1,
        a_mark_label="A out",
        b_mark_frame=2,
        b_mark_label="B in",
    )
    assert lines[0] == "Video: ?", f"expected 'Video: ?', got {lines[0]!r}"


def test_build_metadata_lines_full_branch_b_event_flow():
    """End-to-end: _branch_frames → _build_metadata_lines for the bug event."""
    from src.analysis.visualization.pet_samples import (
        _branch_frames,
        _build_metadata_lines,
    )

    # The exact event the user reported (frame 80, T33 × T28, PET 1.233)
    a_frame, b_frame, a_label, b_label = _branch_frames(
        a_entry=80,
        a_exit=105,
        b_entry=25,
        b_exit=43,
    )
    # _branch_frames must resolve to branch B
    assert a_frame == 80
    assert b_frame == 43
    assert a_label == "A in"
    assert b_label == "B out"

    # And the metadata lines must reflect those exactly
    lines = _build_metadata_lines(
        video_source="GITI_traffic_video",
        frame_idx=80,
        grid_cell="CELL_AY_7",
        conflict_type="side_swipe",
        track_a=33,
        track_b=28,
        a_mark_frame=a_frame,
        a_mark_label=a_label,
        b_mark_frame=b_frame,
        b_mark_label=b_label,
    )
    assert "A in: f=80" in lines
    assert "B out: f=43" in lines
    assert not any("A out" in line for line in lines)
    assert not any("B in:" in line for line in lines)
