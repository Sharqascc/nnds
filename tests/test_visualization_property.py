
import cv2
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.analysis.visualization.video_overlays import (
    COLORS_BGR,
    DEFAULT_THRESHOLDS,
    overlay_conflict_frame,
)


@given(st.sampled_from(list(COLORS_BGR.keys())))
def test_color_palette_valid_bgr(color_name):
    bgr = COLORS_BGR[color_name]
    assert isinstance(bgr, tuple)
    assert len(bgr) == 3
    assert all(0 <= channel <= 255 for channel in bgr)

@given(st.lists(st.floats(min_value=0.01, max_value=10.0), min_size=5, max_size=5))
def test_default_thresholds_positive_sorted(values):
    thresholds = DEFAULT_THRESHOLDS
    assert all(v > 0 for v in thresholds.values())
    assert thresholds["critical"] < thresholds["serious"] < thresholds["moderate"] < thresholds["safe"]

def test_overlay_conflict_frame_shape_preserved(tmp_path):
    # Create a short synthetic video
    video_path = tmp_path / "test.mp4"
    height, width = 60, 80
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 5.0, (width, height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        video_path = tmp_path / "test.avi"
        out = cv2.VideoWriter(str(video_path), fourcc, 5.0, (width, height))
    assert out.isOpened(), "Could not create test video"
    for _ in range(3):
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    # Correct format: list of trajectories; each trajectory is list of (t, x, y)
    trajectory = [[(0.2, 0.3, 0.1), (0.4, 0.5, 0.2)]]
    result = overlay_conflict_frame(str(video_path), 0, trajectory)
    assert result.shape == (height, width, 3)
    assert result.dtype == np.uint8
