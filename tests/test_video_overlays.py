
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import cv2
import pytest

from src.analysis.visualization.video_overlays import (
    VideoOverlayPlotter,
    COLORS_BGR,
    DEFAULT_THRESHOLDS,
    overlay_conflict_frame,
    generate_conflict_video,
    create_before_during_after,
    save_conflict_frame,
    overlay_full_visualization,
)


@pytest.fixture
def plotter():
    return VideoOverlayPlotter(dpi=100, show_grid=True, grid_alpha=0.5)


def make_frame(w=200, h=150):
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------- init and severity ----------------

def test_init_defaults(plotter):
    assert plotter.dpi == 100
    assert plotter.colorblind_safe is True
    assert plotter.thresholds == DEFAULT_THRESHOLDS
    assert plotter.font_scale == 0.6
    assert plotter.line_thickness == 2
    assert plotter.show_grid is True
    assert plotter.grid_alpha == 0.5


def test_get_severity_color(plotter):
    assert plotter._get_severity_color(0.1) == COLORS_BGR["red"]
    assert plotter._get_severity_color(0.7) == COLORS_BGR["orange"]
    assert plotter._get_severity_color(1.2) == COLORS_BGR["yellow"]
    assert plotter._get_severity_color(2.0) == COLORS_BGR["green"]
    assert plotter._get_severity_color(6.0) == COLORS_BGR["blue"]


def test_get_severity_label(plotter):
    assert plotter._get_severity_label(0.1) == "CRITICAL"
    assert plotter._get_severity_label(0.7) == "SERIOUS"
    assert plotter._get_severity_label(1.2) == "MODERATE"
    assert plotter._get_severity_label(2.0) == "SLIGHT"
    assert plotter._get_severity_label(6.0) == "SAFE"


# ---------------- overlay_trajectories ----------------

def test_overlay_trajectories_default_colors(plotter):
    frame = make_frame()
    traj = [[(0, 10, 10), (1, 20, 20), (2, 30, 30)]]
    out = plotter.overlay_trajectories(frame, [traj])
    assert out.shape == frame.shape


def test_overlay_trajectories_short_traj_skipped(plotter):
    frame = make_frame()
    traj = [[(0, 10, 10)]]  # len < 2
    out = plotter.overlay_trajectories(frame, [traj])
    assert out.shape == frame.shape


def test_overlay_trajectories_with_ids_and_no_arrows(plotter):
    frame = make_frame()
    traj = [[(0, 10, 10), (1, 20, 20)]]
    out = plotter.overlay_trajectories(frame, [traj], track_ids=[1], show_arrows=False)
    assert out.shape == frame.shape


def test_overlay_trajectories_custom_colors(plotter):
    frame = make_frame()
    traj = [[(0, 10, 10), (1, 20, 20)]]
    out = plotter.overlay_trajectories(frame, [traj], colors=[(0,0,255)])
    assert out.shape == frame.shape


# ---------------- overlay_bounding_boxes ----------------

def test_overlay_bounding_boxes_default(plotter):
    frame = make_frame()
    boxes = [(10, 10, 50, 50)]
    out = plotter.overlay_bounding_boxes(frame, boxes)
    assert out.shape == frame.shape


def test_overlay_bounding_boxes_with_ids(plotter):
    frame = make_frame()
    boxes = [(10, 10, 50, 50)]
    out = plotter.overlay_bounding_boxes(frame, boxes, track_ids=[1])
    assert out.shape == frame.shape


# ---------------- overlay_conflict_info ----------------

def test_overlay_conflict_info_all_positions(plotter):
    frame = make_frame()
    for pos in ["top-left", "top-right", "bottom-left", "bottom-right"]:
        out = plotter.overlay_conflict_info(frame, pet_value=0.5, ttc_value=2.0, frame_number=10, timestamp=0.5, position=pos)
        assert out.shape == frame.shape


def test_overlay_conflict_info_minimal(plotter):
    frame = make_frame()
    out = plotter.overlay_conflict_info(frame, pet_value=0.5)
    assert out.shape == frame.shape


# ---------------- overlay_conflict_zone ----------------

def test_overlay_conflict_zone_default(plotter):
    frame = make_frame()
    out = plotter.overlay_conflict_zone(frame, center=(100, 100))
    assert out.shape == frame.shape


def test_overlay_conflict_zone_custom_color(plotter):
    frame = make_frame()
    out = plotter.overlay_conflict_zone(frame, center=(100, 100), color=(0,255,0), radius=30, alpha=0.5)
    assert out.shape == frame.shape


# ---------------- overlay_conflict_frame ----------------

def make_mock_cap(ret=True, frame=None):
    if frame is None:
        frame = make_frame()
    cap = MagicMock()
    cap.set.return_value = None
    cap.read.return_value = (ret, frame)
    cap.release.return_value = None
    return cap


def test_overlay_conflict_frame_success(plotter):
    cap = make_mock_cap()
    with patch('cv2.VideoCapture', return_value=cap),          patch.object(plotter, 'overlay_trajectories', return_value=make_frame()),          patch.object(plotter, 'overlay_conflict_zone', return_value=make_frame()),          patch.object(plotter, 'overlay_conflict_info', return_value=make_frame()):
        out = plotter.overlay_conflict_frame(
            video_path='dummy.mp4', frame_idx=0,
            trajectories=[[(0, 10, 10), (1, 20, 20)]],
            pet_value=1.0,
            conflict_center=(100,100)
        )
    assert out.shape == (150, 200, 3)


def test_overlay_conflict_frame_failure(plotter):
    cap = make_mock_cap(ret=False)
    with patch('cv2.VideoCapture', return_value=cap):
        with pytest.raises(RuntimeError):
            plotter.overlay_conflict_frame(
                video_path='dummy.mp4', frame_idx=0,
                trajectories=[]
            )


def test_overlay_conflict_frame_with_grid(plotter):
    grid = MagicMock()
    grid.draw_overlay.return_value = make_frame()
    grid.get_cell_center.return_value = (100,100)
    grid.cell_size = 20
    cap = make_mock_cap()
    with patch('cv2.VideoCapture', return_value=cap),          patch.object(plotter, 'overlay_trajectories', return_value=make_frame()):
        out = plotter.overlay_conflict_frame(
            video_path='dummy.mp4', frame_idx=0,
            trajectories=[[(0, 10, 10), (1, 20, 20)]],
            pet_value=1.0,
            grid=grid,
            cell_id='G1'
        )
    assert out.shape == (150, 200, 3)


# ---------------- save_frame ----------------

def test_save_frame_png(plotter, tmp_path):
    save_path = tmp_path / "sub" / "frame.png"
    with patch('cv2.imwrite', return_value=True) as imwrite_mock:
        plotter.save_frame(make_frame(), str(save_path))
    assert save_path.parent.exists()
    imwrite_mock.assert_called_once()


def test_save_frame_jpg(plotter, tmp_path):
    save_path = tmp_path / "frame.jpg"
    with patch('cv2.imwrite', return_value=True) as imwrite_mock:
        plotter.save_frame(make_frame(), str(save_path))
    imwrite_mock.assert_called_once()


# ---------------- generate_conflict_video ----------------

def make_mock_video_capture(ret=True, frame=None, frame_count=1):
    if frame is None:
        frame = make_frame()
    cap = MagicMock()
    cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_WIDTH: 200,
        cv2.CAP_PROP_FRAME_HEIGHT: 150,
    }.get(prop, 0)
    cap.set.return_value = None
    cap.read.return_value = (ret, frame)
    cap.release.return_value = None
    return cap


def test_generate_conflict_video_success(plotter):
    cap = make_mock_video_capture()
    writer = MagicMock()
    with patch('cv2.VideoCapture', return_value=cap),          patch('cv2.VideoWriter', return_value=writer),          patch('cv2.VideoWriter_fourcc', return_value=0),          patch.object(plotter, 'overlay_trajectories', return_value=make_frame()),          patch.object(plotter, 'overlay_conflict_info', return_value=make_frame()):
        plotter.generate_conflict_video(
            video_path='dummy.mp4',
            frame_range=(0, 0),
            trajectories=[[(0,10,10),(1,20,20)]],
            output_path='output/test.mp4',
            fps=10
        )
    writer.write.assert_called_once()


# ---------------- Convenience functions ----------------

def test_convenience_overlay_conflict_frame():
    with patch.object(VideoOverlayPlotter, 'overlay_conflict_frame', return_value=make_frame()) as mock_overlay:
        out = overlay_conflict_frame('dummy.mp4', 0, [[(0,0,0),(1,1,1)]])
    assert out.shape == (150, 200, 3)


def test_convenience_generate_conflict_video():
    with patch.object(VideoOverlayPlotter, 'generate_conflict_video') as mock_gen:
        generate_conflict_video('dummy.mp4', (0,1), [[(0,0,0),(1,1,1)]])
    mock_gen.assert_called_once()


def test_create_before_during_after():
    with patch.object(VideoOverlayPlotter, 'overlay_conflict_frame', return_value=make_frame()),          patch.object(VideoOverlayPlotter, 'save_frame') as save_mock:
        create_before_during_after('dummy.mp4', 0,1,2, [[(0,0,0),(1,1,1)]], save_path='output/seq.png')
    save_mock.assert_called_once()


def test_save_conflict_frame_success(tmp_path):
    grid = MagicMock()
    grid.draw_overlay.return_value = make_frame()
    grid.get_cell_center.return_value = (100,100)
    grid.cell_size = 20
    cap = make_mock_cap()
    with patch('src.analysis.visualization.video_overlays.SpatialGrid', return_value=grid),          patch('cv2.VideoCapture', return_value=cap),          patch.object(VideoOverlayPlotter, 'save_frame') as save_mock:
        out = save_conflict_frame('dummy.mp4', 'grid.json', 'G1', 0, str(tmp_path/'out.png'))
    save_mock.assert_called_once()
    assert out.endswith('.png')


def test_save_conflict_frame_no_spatial_grid():
    with patch('src.analysis.visualization.video_overlays.SpatialGrid', None):
        with pytest.raises(ImportError):
            save_conflict_frame('dummy.mp4', 'grid.json', 'G1', 0, 'out.png')


def test_overlay_full_visualization():
    frame = make_frame()
    grid = MagicMock()
    grid.draw_overlay.return_value = frame.copy()
    grid.get_cell_center.return_value = (100,100)
    detections = [{'bbox': [10,10,50,50], 'confidence': 0.8, 'track_id': 1}]
    out = overlay_full_visualization(
        frame, 5, grid, detections=detections, conflict_cell='G1', pet_value=0.5, vehicle_ids=[1,2]
    )
    assert out.shape == frame.shape
