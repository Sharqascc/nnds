import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pandas as pd
import pytest

from src.analysis.visualization.pet_verification_visualizer import PETVerificationVisualizer


@pytest.fixture
def sample_event_df():
    data = {
        "event_id": [1],
        "pet": [2.0],
        "frame": [100],
        "track_a": [1],
        "track_b": [2],
        "grid_cell": ["G_A_1"],
        "track_a_exit_frame": [90],
        "track_a_exit_time_sec": [3.0],
        "track_b_entry_frame": [110],
        "track_b_entry_time_sec": [3.667],
        "site": ["GITI"],
        "world_traj_i": [
            json.dumps(
                [
                    {"frame": 80, "x_pixel": 100, "y_pixel": 200},
                    {"frame": 90, "x_pixel": 120, "y_pixel": 220},
                ]
            )
        ],
        "world_traj_j": [
            json.dumps(
                [
                    {"frame": 100, "x_pixel": 150, "y_pixel": 250},
                    {"frame": 110, "x_pixel": 170, "y_pixel": 270},
                ]
            )
        ],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_video_path(tmp_path):
    video_path = tmp_path / "dummy.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for _ in range(200):
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    writer.release()
    return str(video_path)


def test_initialization(sample_event_df, sample_video_path, tmp_path):
    csv_path = tmp_path / "events.csv"
    sample_event_df.to_csv(csv_path, index=False)
    viz = PETVerificationVisualizer(str(csv_path), sample_video_path)
    assert viz.df is not None
    assert viz.video_path == Path(sample_video_path)


def test_load_event_success(sample_event_df, sample_video_path, tmp_path):
    csv_path = tmp_path / "events.csv"
    sample_event_df.to_csv(csv_path, index=False)
    viz = PETVerificationVisualizer(str(csv_path), sample_video_path)
    event = viz.load_event(1)
    assert event["event_id"] == 1
    assert event["pet"] == 2.0


def test_load_event_missing(sample_event_df, sample_video_path, tmp_path):
    csv_path = tmp_path / "events.csv"
    sample_event_df.to_csv(csv_path, index=False)
    viz = PETVerificationVisualizer(str(csv_path), sample_video_path)
    with pytest.raises(ValueError):
        viz.load_event(999)


def test_parse_traj_json(sample_event_df, sample_video_path, tmp_path):
    csv_path = tmp_path / "events.csv"
    sample_event_df.to_csv(csv_path, index=False)
    viz = PETVerificationVisualizer(str(csv_path), sample_video_path)
    traj = viz.parse_traj('[{"frame":1, "x_pixel":10, "y_pixel":20}]')
    assert len(traj) == 1


def test_draw_trajectory(sample_event_df, sample_video_path, tmp_path):
    csv_path = tmp_path / "events.csv"
    sample_event_df.to_csv(csv_path, index=False)
    viz = PETVerificationVisualizer(str(csv_path), sample_video_path)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    traj = [{"frame": 0, "x_pixel": 10, "y_pixel": 10}, {"frame": 1, "x_pixel": 20, "y_pixel": 20}]
    out = viz.draw_trajectory(frame, traj, (255, 0, 0), current_frame=1)
    assert out.shape == frame.shape
    assert np.any(out != 0)


def test_draw_grid_cell(sample_event_df, sample_video_path, tmp_path):
    csv_path = tmp_path / "events.csv"
    sample_event_df.to_csv(csv_path, index=False)
    viz = PETVerificationVisualizer(str(csv_path), sample_video_path)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    out = viz.draw_grid_cell(frame, "G_A_1")
    assert out.shape == frame.shape
    assert np.any(out != 0)


def test_draw_timing_info(sample_event_df, sample_video_path, tmp_path):
    csv_path = tmp_path / "events.csv"
    sample_event_df.to_csv(csv_path, index=False)
    viz = PETVerificationVisualizer(str(csv_path), sample_video_path)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    event = viz.load_event(1)
    out = viz.draw_timing_info(frame, event)
    assert out.shape == frame.shape
    assert np.any(out != 0)


def test_process_frame(sample_event_df, sample_video_path, tmp_path):
    csv_path = tmp_path / "events.csv"
    sample_event_df.to_csv(csv_path, index=False)
    viz = PETVerificationVisualizer(str(csv_path), sample_video_path)
    event = viz.load_event(1)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    out = viz.process_frame(frame, event, current_frame=100)
    assert out.shape == frame.shape
    assert np.any(out != 0)


def test_generate_video(sample_event_df, sample_video_path, tmp_path):
    csv_path = tmp_path / "events.csv"
    sample_event_df.to_csv(csv_path, index=False)
    viz = PETVerificationVisualizer(str(csv_path), sample_video_path)
    output = tmp_path / "out.mp4"
    viz.generate_video(1, str(output), fps=10)
    assert output.exists()
    assert output.stat().st_size > 0


def test_smooth_points_short_trajectory():
    viz = PETVerificationVisualizer.__new__(PETVerificationVisualizer)
    traj = [{"x_pixel": 1, "y_pixel": 1}, {"x_pixel": 2, "y_pixel": 2}]
    pts = viz._smooth_points(traj)
    assert len(pts) == 2
    assert pts[0] == (1, 1)


def test_smooth_points_smooths():
    viz = PETVerificationVisualizer.__new__(PETVerificationVisualizer)
    traj = [{"x_pixel": i * 10, "y_pixel": i * 10} for i in range(10)]
    pts = viz._smooth_points(traj)
    assert len(pts) == 10
    # Smoothed middle point should be near original (not too far)
    assert abs(pts[5][0] - 50) < 5


def test_get_position_at_valid_and_none():
    viz = PETVerificationVisualizer.__new__(PETVerificationVisualizer)
    traj = [{"frame": 0, "x_pixel": 10, "y_pixel": 20}, {"frame": 5, "x_pixel": 15, "y_pixel": 25}]
    pos = viz._get_position_at(traj, 3)
    assert pos == (10, 20)
    pos2 = viz._get_position_at(traj, 5)
    assert pos2 == (15, 25)
    assert viz._get_position_at(traj, -1) is None


def test_parse_traj_list_input():
    viz = PETVerificationVisualizer.__new__(PETVerificationVisualizer)
    traj = [{"frame": 1, "x_pixel": 2, "y_pixel": 3}]
    assert viz.parse_traj(traj) is traj


def test_parse_traj_invalid_string():
    viz = PETVerificationVisualizer.__new__(PETVerificationVisualizer)
    assert viz.parse_traj("not-json") == []


def test_smooth_points_small_window():
    viz = PETVerificationVisualizer.__new__(PETVerificationVisualizer)
    traj = [{"x_pixel": i * 10, "y_pixel": i * 10} for i in range(10)]
    pts = viz._smooth_points(traj, window=5, polyorder=2)
    assert len(pts) == 10


def test_generate_video_no_trajectory(tmp_path):
    csv_path = tmp_path / "events.csv"
    pd.DataFrame(
        {
            "event_id": [1],
            "pet": [2.0],
            "frame": [50],
            "track_a": [1],
            "track_b": [2],
            "grid_cell": ["G_A_1"],
            "first_track_id": [1],
            "second_track_id": [2],
            "first_exit_frame": [40],
            "first_exit_time_sec": [1.5],
            "second_entry_frame": [60],
            "second_entry_time_sec": [2.0],
            "site": ["GITI"],
            "traj_a_json": ["[]"],
            "traj_b_json": ["[]"],
        }
    ).to_csv(csv_path, index=False)
    viz = PETVerificationVisualizer(str(csv_path), str(tmp_path / "dummy.mp4"))
    with pytest.raises(ValueError):
        viz.generate_video(1, str(tmp_path / "out.mp4"))


def test_generate_video_cannot_open_source(tmp_path, monkeypatch):
    csv_path = tmp_path / "events.csv"
    pd.DataFrame(
        {
            "event_id": [1],
            "pet": [2.0],
            "frame": [50],
            "track_a": [1],
            "track_b": [2],
            "grid_cell": ["G_A_1"],
            "first_track_id": [1],
            "second_track_id": [2],
            "first_exit_frame": [40],
            "first_exit_time_sec": [1.5],
            "second_entry_frame": [60],
            "second_entry_time_sec": [2.0],
            "site": ["GITI"],
            "traj_a_json": [
                '[{"frame":0,"x_pixel":10,"y_pixel":10},{"frame":1,"x_pixel":20,"y_pixel":20}]'
            ],
            "traj_b_json": [
                '[{"frame":0,"x_pixel":30,"y_pixel":30},{"frame":1,"x_pixel":40,"y_pixel":40}]'
            ],
        }
    ).to_csv(csv_path, index=False)
    viz = PETVerificationVisualizer(str(csv_path), str(tmp_path / "dummy.mp4"))
    monkeypatch.setattr("cv2.VideoCapture", lambda *a, **k: MagicMock(isOpened=lambda: False))
    with pytest.raises(RuntimeError):
        viz.generate_video(1, str(tmp_path / "out.mp4"))


# We'll need MagicMock import; already imported? Add if missing.


def test_parse_traj_list_input():
    viz = PETVerificationVisualizer.__new__(PETVerificationVisualizer)
    traj = [{"frame": 1, "x_pixel": 2, "y_pixel": 3}]
    assert viz.parse_traj(traj) == traj


def test_parse_traj_invalid_string():
    viz = PETVerificationVisualizer.__new__(PETVerificationVisualizer)
    assert viz.parse_traj("not a json") == []


def test_smooth_points_small_window():
    viz = PETVerificationVisualizer.__new__(PETVerificationVisualizer)
    traj = [{"x_pixel": i * 10, "y_pixel": i * 10} for i in range(10)]
    pts = viz._smooth_points(traj, window=5, polyorder=2)
    assert len(pts) == 10


def test_generate_video_no_trajectory(sample_event_df, tmp_path):
    csv_path = tmp_path / "events.csv"
    sample_event_df.to_csv(csv_path, index=False)
    viz = PETVerificationVisualizer(str(csv_path), str(tmp_path / "dummy.mp4"))
    # make trajectory empty
    viz.df.loc[0, "traj_a_json"] = "[]"
    viz.df.loc[0, "traj_b_json"] = "[]"
    with pytest.raises(ValueError):
        viz.generate_video(1, str(tmp_path / "out.mp4"))


def test_generate_video_cannot_open_source(sample_event_df, tmp_path):
    csv_path = tmp_path / "events.csv"
    sample_event_df.to_csv(csv_path, index=False)
    viz = PETVerificationVisualizer(
        str(csv_path), str(tmp_path / "dummy.mp4"), background_mode="video"
    )
    with patch("cv2.VideoCapture", return_value=MagicMock(isOpened=lambda: False)):
        with pytest.raises(RuntimeError):
            viz.generate_video(1, str(tmp_path / "out.mp4"))


def test_generate_video_zero_frames(sample_event_df, tmp_path):
    csv_path = tmp_path / "events.csv"
    sample_event_df.to_csv(csv_path, index=False)
    viz = PETVerificationVisualizer(
        str(csv_path), str(tmp_path / "dummy.mp4"), background_mode="video"
    )
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.return_value = 0  # frame count zero
    with patch("cv2.VideoCapture", return_value=cap), pytest.raises(RuntimeError):
        viz.generate_video(1, str(tmp_path / "out.mp4"))


def test_generate_video_writer_fail(sample_event_df, tmp_path, monkeypatch):
    csv_path = tmp_path / "events.csv"
    sample_event_df.to_csv(csv_path, index=False)
    viz = PETVerificationVisualizer(str(csv_path), str(tmp_path / "dummy.mp4"))
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_COUNT: 2,
        cv2.CAP_PROP_FRAME_WIDTH: 320,
        cv2.CAP_PROP_FRAME_HEIGHT: 240,
    }.get(prop, 0)
    writer = MagicMock()
    writer.isOpened.return_value = False
    with patch("cv2.VideoCapture", return_value=cap), patch("cv2.VideoWriter", return_value=writer):
        with pytest.raises(RuntimeError):
            viz.generate_video(1, str(tmp_path / "out.mp4"))


def test_generate_video_read_fails_break(sample_event_df, tmp_path):
    csv_path = tmp_path / "events.csv"
    sample_event_df.to_csv(csv_path, index=False)
    viz = PETVerificationVisualizer(str(csv_path), str(tmp_path / "dummy.mp4"))
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_COUNT: 2,
        cv2.CAP_PROP_FRAME_WIDTH: 320,
        cv2.CAP_PROP_FRAME_HEIGHT: 240,
    }.get(prop, 0)
    cap.read.side_effect = [(True, np.zeros((240, 320, 3), dtype=np.uint8)), (False, None)]
    writer = MagicMock()
    writer.isOpened.return_value = True
    with patch("cv2.VideoCapture", return_value=cap), patch("cv2.VideoWriter", return_value=writer):
        viz.generate_video(1, str(tmp_path / "out.mp4"), fps=10)


def test_generate_video_single_frame(sample_event_df, tmp_path):
    csv_path = tmp_path / "events.csv"
    sample_event_df.to_csv(csv_path, index=False)
    viz = PETVerificationVisualizer(str(csv_path), str(tmp_path / "dummy.mp4"))
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_COUNT: 1,
        cv2.CAP_PROP_FRAME_WIDTH: 320,
        cv2.CAP_PROP_FRAME_HEIGHT: 240,
    }.get(prop, 0)
    cap.read.return_value = (True, np.zeros((240, 320, 3), dtype=np.uint8))
    writer = MagicMock()
    writer.isOpened.return_value = True
    with patch("cv2.VideoCapture", return_value=cap), patch("cv2.VideoWriter", return_value=writer):
        viz.generate_video(1, str(tmp_path / "out.mp4"), fps=10)


def test_smooth_points_savgol():
    viz = PETVerificationVisualizer.__new__(PETVerificationVisualizer)
    traj = [{"x_pixel": i * 10, "y_pixel": i * 10} for i in range(20)]
    pts = viz._smooth_points(traj, window=11, polyorder=3)
    assert len(pts) == 20
    # Middle point should be close to original straight line
    assert abs(pts[10][0] - 100) < 5


def test_draw_text_background():
    viz = PETVerificationVisualizer.__new__(PETVerificationVisualizer)
    frame = np.full((100, 200, 3), 255, dtype=np.uint8)
    out = viz._draw_text_background(frame, (10, 10), (150, 60), alpha=0.5)
    assert out.shape == frame.shape
    # Background should not be pure white after overlay
    assert np.any(out < 250)


def test_parse_traj_non_str_non_list():
    viz = PETVerificationVisualizer.__new__(PETVerificationVisualizer)
    assert viz.parse_traj(123) == []


def test_smooth_points_fallback_moving_average():
    viz = PETVerificationVisualizer.__new__(PETVerificationVisualizer)
    # n=3, window=3, polyorder=2 -> w=3 < polyorder+2=4, triggers fallback
    traj = [
        {"x_pixel": 0, "y_pixel": 0},
        {"x_pixel": 10, "y_pixel": 10},
        {"x_pixel": 20, "y_pixel": 20},
    ]
    pts = viz._smooth_points(traj, window=3, polyorder=2)
    assert len(pts) == 3


def test_draw_text_background_invalid_roi():
    viz = PETVerificationVisualizer.__new__(PETVerificationVisualizer)
    frame = np.full((100, 200, 3), 255, dtype=np.uint8)
    out = viz._draw_text_background(frame, (150, 150), (140, 140), alpha=0.5)
    assert np.array_equal(out, frame)


def test_enhance_background_changes_frame():
    viz = PETVerificationVisualizer.__new__(PETVerificationVisualizer)
    frame = np.full((100, 100, 3), 100, dtype=np.uint8)
    out = viz._enhance_background(frame)
    assert out.shape == frame.shape
    # CLAHE/unsharp should produce some change
    assert not np.array_equal(out, frame)


def test_generate_video_with_video_background_success(sample_event_df, tmp_path):
    csv_path = tmp_path / "events.csv"
    sample_event_df.to_csv(csv_path, index=False)
    viz = PETVerificationVisualizer(
        str(csv_path), str(tmp_path / "dummy.mp4"), background_mode="video"
    )
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_COUNT: 2,
        cv2.CAP_PROP_FRAME_WIDTH: 320,
        cv2.CAP_PROP_FRAME_HEIGHT: 240,
    }.get(prop, 0)
    frame0 = np.zeros((240, 320, 3), dtype=np.uint8)
    frame1 = np.zeros((240, 320, 3), dtype=np.uint8)
    cap.read.side_effect = [(True, frame0), (True, frame1)]
    cap.release.return_value = None
    writer = MagicMock()
    writer.isOpened.return_value = True
    with patch("cv2.VideoCapture", return_value=cap), patch("cv2.VideoWriter", return_value=writer):
        out = viz.generate_video(1, str(tmp_path / "out.mp4"), fps=10)
    assert out is not None
    assert writer.write.call_count == 2


def test_generate_video_max_frames_cap(sample_event_df, tmp_path):
    csv_path = tmp_path / "events.csv"
    sample_event_df.to_csv(csv_path, index=False)
    viz = PETVerificationVisualizer(
        str(csv_path), str(tmp_path / "dummy.mp4"), background_mode="video"
    )
    cap = MagicMock()
    cap.isOpened.return_value = True
    # Simulate 1000-frame video
    cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_COUNT: 1000,
        cv2.CAP_PROP_FRAME_WIDTH: 320,
        cv2.CAP_PROP_FRAME_HEIGHT: 240,
    }.get(prop, 0)
    cap.read.return_value = (True, np.zeros((240, 320, 3), dtype=np.uint8))
    writer = MagicMock()
    writer.isOpened.return_value = True
    with patch("cv2.VideoCapture", return_value=cap), patch("cv2.VideoWriter", return_value=writer):
        viz.generate_video(1, str(tmp_path / "out.mp4"), fps=30, max_frames=200)
    # Should write the conflict-centered window length (40 frames in this test)
    assert writer.write.call_count == 40


def test_draw_grid_cell_with_center():
    viz = PETVerificationVisualizer.__new__(PETVerificationVisualizer)
    viz.colors = {
        "track_a": (255, 0, 0),
        "track_b": (0, 165, 255),
        "grid": (0, 255, 255),
        "conflict": (0, 0, 255),
        "text": (255, 255, 255),
    }
    frame = np.full((200, 300, 3), 255, dtype=np.uint8)
    out = viz.draw_grid_cell(frame, "G_A_1", center=(150, 100), radius=40)
    assert out.shape == frame.shape


def test_smooth_points_preserves_straight_line():
    viz = PETVerificationVisualizer.__new__(PETVerificationVisualizer)
    # Perfect straight line y = 2*x
    traj = [{"x_pixel": i, "y_pixel": 2 * i} for i in range(30)]
    pts = viz._smooth_points(traj, window=11, polyorder=3)
    # All points should remain on the line (max deviation < 1 pixel)
    for (x, y), orig in zip(pts, traj):
        expected_y = 2 * orig["x_pixel"]
        assert abs(y - expected_y) < 1.0, f"Smoothed point {x, y} deviates from line"


def test_smooth_points_reduces_noise():
    viz = PETVerificationVisualizer.__new__(PETVerificationVisualizer)
    # Trajectory with sinusoidal noise
    n = 50
    x = np.arange(n)
    y_true = 0.1 * x  # gentle slope
    noise = 5 * np.sin(np.linspace(0, 4 * np.pi, n))
    y_raw = y_true + noise
    traj_raw = [{"x_pixel": float(x[i]), "y_pixel": float(y_raw[i])} for i in range(n)]

    # Total variation (sum of absolute first differences)
    raw_tv = np.sum(np.abs(np.diff(y_raw)))

    pts_smooth = viz._smooth_points(traj_raw, window=21, polyorder=3)
    y_smooth = np.array([p[1] for p in pts_smooth])
    smooth_tv = np.sum(np.abs(np.diff(y_smooth)))

    assert smooth_tv < raw_tv, f"Smoothed TV {smooth_tv:.2f} should be less than raw {raw_tv:.2f}"


def test_draw_trajectory_uses_anti_aliased_line():
    viz = PETVerificationVisualizer.__new__(PETVerificationVisualizer)
    # Patch cv2.line to assert lineType=cv2.LINE_AA is passed
    with patch("cv2.line", wraps=cv2.line) as mock_line:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        traj = [{"x_pixel": 10, "y_pixel": 10}, {"x_pixel": 80, "y_pixel": 80}]
        viz.draw_trajectory(frame, traj, (255, 0, 0), current_frame=None)
        # Check that at least one call used LINE_AA
        line_type_used = False
        for call in mock_line.call_args_list:
            args, kwargs = call
            if "lineType" in kwargs and kwargs["lineType"] == cv2.LINE_AA:
                line_type_used = True
                break
        assert line_type_used, "cv2.line was not called with LINE_AA"


def test_draw_grid_cell_red_visible():
    viz = PETVerificationVisualizer.__new__(PETVerificationVisualizer)
    viz.colors = {
        "track_a": (255, 0, 0),
        "track_b": (0, 165, 255),
        "grid": (0, 0, 255),  # red
        "conflict": (0, 0, 255),
        "text": (255, 255, 255),
    }
    frame = np.full((200, 300, 3), 255, dtype=np.uint8)
    out = viz.draw_grid_cell(frame, "G_A_1", center=(150, 100), radius=40)
    # Check for red pixels (BGR red = 0,0,255)
    red_mask = (out[:, :, 2] > 200) & (out[:, :, 0] < 100) & (out[:, :, 1] < 100)
    assert int(np.sum(red_mask)) > 100, "Expected visible red grid cell"


def test_generate_video_centered_window(sample_event_df, tmp_path):
    csv_path = tmp_path / "events.csv"
    sample_event_df.to_csv(csv_path, index=False)
    viz = PETVerificationVisualizer(
        str(csv_path), str(tmp_path / "dummy.mp4"), background_mode="video"
    )
    cap = MagicMock()
    cap.isOpened.return_value = True
    total_frames = 1000
    cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_COUNT: total_frames,
        cv2.CAP_PROP_FRAME_WIDTH: 320,
        cv2.CAP_PROP_FRAME_HEIGHT: 240,
    }.get(prop, 0)
    cap.read.return_value = (True, np.zeros((240, 320, 3), dtype=np.uint8))
    writer = MagicMock()
    writer.isOpened.return_value = True
    with patch("cv2.VideoCapture", return_value=cap), patch("cv2.VideoWriter", return_value=writer):
        viz.generate_video(1, str(tmp_path / "out.mp4"), fps=30, max_frames=200)
    # Should write the conflict-centered window length (40 frames in this test)
    assert writer.write.call_count == 40


def test_generate_video_with_spatial_grid(sample_event_df, tmp_path):
    csv_path = tmp_path / "events.csv"
    sample_event_df.to_csv(csv_path, index=False)
    mock_grid = MagicMock()
    mock_grid.draw_overlay.return_value = np.zeros((240, 320, 3), dtype=np.uint8)
    viz = PETVerificationVisualizer(
        str(csv_path),
        str(tmp_path / "dummy.mp4"),
        background_mode="schematic",
        spatial_grid=mock_grid,
    )
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_COUNT: 2,
        cv2.CAP_PROP_FRAME_WIDTH: 320,
        cv2.CAP_PROP_FRAME_HEIGHT: 240,
    }.get(prop, 0)
    cap.read.return_value = (True, np.zeros((240, 320, 3), dtype=np.uint8))
    writer = MagicMock()
    writer.isOpened.return_value = True
    with patch("cv2.VideoCapture", return_value=cap), patch("cv2.VideoWriter", return_value=writer):
        viz.generate_video(1, str(tmp_path / "out.mp4"), fps=10, max_frames=2)
    assert mock_grid.draw_overlay.call_count > 0


def test_draw_trajectory_animation_grows():
    viz = PETVerificationVisualizer.__new__(PETVerificationVisualizer)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    traj = [
        {"frame": 0, "x_pixel": 10, "y_pixel": 10},
        {"frame": 1, "x_pixel": 20, "y_pixel": 20},
        {"frame": 2, "x_pixel": 80, "y_pixel": 80},
    ]
    out0 = viz.draw_trajectory(frame.copy(), traj, (255, 0, 0), current_frame=0)
    out2 = viz.draw_trajectory(frame.copy(), traj, (255, 0, 0), current_frame=2)
    # At frame 0, the segment between point0 and point1 should not exist
    assert out0[15, 15].sum() == 0, "Early frame should not draw future segment"
    # At frame 2, the segment should be drawn
    assert out2[15, 15].sum() > 0, "Later frame should include the segment"


def test_generate_video_uses_cap_set(sample_event_df, tmp_path):
    csv_path = tmp_path / "events.csv"
    sample_event_df.to_csv(csv_path, index=False)
    viz = PETVerificationVisualizer(
        str(csv_path), str(tmp_path / "dummy.mp4"), background_mode="video"
    )
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_COUNT: 100,
        cv2.CAP_PROP_FRAME_WIDTH: 320,
        cv2.CAP_PROP_FRAME_HEIGHT: 240,
    }.get(prop, 0)
    cap.read.return_value = (True, np.zeros((240, 320, 3), dtype=np.uint8))
    writer = MagicMock()
    writer.isOpened.return_value = True
    with patch("cv2.VideoCapture", return_value=cap), patch("cv2.VideoWriter", return_value=writer):
        viz.generate_video(1, str(tmp_path / "out.mp4"), fps=10, max_frames=10)
    # cap.set should have been called with CAP_PROP_POS_FRAMES
    assert cap.set.called
    # At least one call with property CAP_PROP_POS_FRAMES
    assert any(args[0] == cv2.CAP_PROP_POS_FRAMES for args, kwargs in cap.set.call_args_list)


def test_init_missing_required_columns(tmp_path):
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame({"event_id": [1], "pet": [2.0]}).to_csv(csv_path, index=False)
    with pytest.raises(ValueError):
        PETVerificationVisualizer(str(csv_path), "dummy.mp4")
