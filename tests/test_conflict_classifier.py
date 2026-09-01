
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.conflict_classifier import classify_conflict_geometry


def _make_traj(points):
    return json.dumps([{"frame": p[0], "x_pixel": p[1], "y_pixel": p[2],
                        "world_x": p[1] * 0.1, "world_y": p[2] * 0.1} for p in points])

def test_head_on():
    # Two tracks moving toward each other along x-axis
    traj_a = _make_traj([(0, 0, 0), (1, 10, 0), (2, 20, 0), (3, 30, 0)])
    traj_b = _make_traj([(0, 100, 0), (1, 90, 0), (2, 80, 0), (3, 70, 0)])
    result = classify_conflict_geometry(traj_a, traj_b, conflict_frame=4, fps=30.0)
    assert result == 'head_on', f"Expected head_on, got {result}"

def test_crossing():
    # One moving along x, other along y
    traj_a = _make_traj([(0, 0, 0), (1, 10, 0), (2, 20, 0), (3, 30, 0)])
    traj_b = _make_traj([(0, 50, 0), (1, 50, 10), (2, 50, 20), (3, 50, 30)])
    result = classify_conflict_geometry(traj_a, traj_b, conflict_frame=4, fps=30.0)
    assert result == 'crossing', f"Expected crossing, got {result}"

def test_rear_end():
    # Same direction, one faster
    traj_a = _make_traj([(0, 0, 0), (1, 5, 0), (2, 10, 0), (3, 15, 0)])
    traj_b = _make_traj([(0, 20, 0), (1, 25, 0), (2, 30, 0), (3, 35, 0)])
    result = classify_conflict_geometry(traj_a, traj_b, conflict_frame=4, fps=30.0)
    assert result in ['rear_end', 'side_swipe'], f"Expected rear_end/side_swipe, got {result}"
