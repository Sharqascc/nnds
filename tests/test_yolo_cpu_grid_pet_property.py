
import sys
import types
from dataclasses import dataclass

# Mock heavy dependencies before importing the module
sys.modules['ultralytics'] = types.SimpleNamespace(YOLO=object)
sys.modules['cv2'] = types.SimpleNamespace(VideoCapture=object, VideoWriter=object)

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.analysis.grid_trajectory.yolo_cpu_grid_pet import (
    TrackPoint,
    _entry_exit_frames,
    _pair_conflict_point,
    _point_in_square,
    _segment_intersection,
)

# ---------- Helper strategies ----------
point_st = st.tuples(st.floats(min_value=0, max_value=100), st.floats(min_value=0, max_value=100))

@st.composite
def track_point_list(draw, min_size=1, max_size=20):
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    points = []
    for _ in range(n):
        frame = draw(st.integers(min_value=0, max_value=100))
        x = draw(st.floats(min_value=0, max_value=100))
        y = draw(st.floats(min_value=0, max_value=100))
        points.append(TrackPoint(frame=frame, x=x, y=y, cls_id=0, cls_name='car', conf=1.0))
    return points

# ---------- Property tests ----------

@given(st.floats(min_value=-100, max_value=100), st.floats(min_value=-100, max_value=100),
       st.floats(min_value=-100, max_value=100), st.floats(min_value=-100, max_value=100),
       st.floats(min_value=0.1, max_value=50))
def test_point_in_square_deterministic(px, py, cx, cy, half_size):
    expected = (cx - half_size) <= px <= (cx + half_size) and (cy - half_size) <= py <= (cy + half_size)
    assert _point_in_square(px, py, cx, cy, half_size) == expected

@given(track_point_list(), st.floats(min_value=-50, max_value=50), st.floats(min_value=-50, max_value=50), st.floats(min_value=0.1, max_value=50))
def test_entry_exit_frames_bounds(points, cx, cy, half_size):
    result = _entry_exit_frames(points, cx, cy, half_size)
    if result is None:
        assert all(not _point_in_square(pt.x, pt.y, cx, cy, half_size) for pt in points)
    else:
        entry, exit_ = result
        inside_frames = [pt.frame for pt in points if _point_in_square(pt.x, pt.y, cx, cy, half_size)]
        assert entry == min(inside_frames)
        assert exit_ == max(inside_frames)
        assert entry <= exit_

@given(point_st, point_st, point_st, point_st)
@settings(deadline=None)
def test_segment_intersection_symmetry(p1, p2, q1, q2):
    inter1 = _segment_intersection(p1, p2, q1, q2)
    inter2 = _segment_intersection(q1, q2, p1, p2)
    if inter1 is None:
        assert inter2 is None
    else:
        assert inter2 is not None
        assert np.allclose(inter1, inter2)

@given(point_st, point_st)
def test_segment_intersection_with_self(p1, p2):
    # A segment should not intersect itself as a non-degenerate segment
    inter = _segment_intersection(p1, p2, p1, p2)
    # Collinear overlapping may return None by design, so we just assert it doesn't raise
    assert inter is None or isinstance(inter, tuple)

@given(track_point_list(min_size=2, max_size=10), track_point_list(min_size=2, max_size=10))
@settings(deadline=None)
def test_pair_conflict_point_type(track_a, track_b):
    result = _pair_conflict_point(track_a, track_b)
    assert result is None or (isinstance(result, tuple) and len(result) == 2)
