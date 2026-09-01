"""
Tests for the core PET computation functions in uvh_coco_fused_grid_pet.
"""
import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.grid_trajectory.uvh_coco_fused_grid_pet import (
    _compute_pet_from_windows,
    _entry_exit_frames,
    _pair_conflict_point,
    _line_side,
    _point_in_square,
    _segment_intersection,
    _segment_bbox,
    _bbox_overlap,
    TrackPoint
)

# ============================================
# 1. PET Formula Tests
# ============================================

def test_pet_sequential_a_to_b():
    """Test PET when A exits before B enters."""
    pet, first, second, frame_ref = _compute_pet_from_windows(
        a_entry=80, a_exit=90,
        b_entry=105, b_exit=115,
        fps=30.0
    )
    assert pet == pytest.approx(0.5, abs=0.001)
    assert first == "a"
    assert second == "b"
    assert frame_ref == 105

def test_pet_sequential_b_to_a():
    """Test PET when B exits before A enters."""
    pet, first, second, frame_ref = _compute_pet_from_windows(
        a_entry=80, a_exit=95,
        b_entry=60, b_exit=70,
        fps=30.0
    )
    assert pet == pytest.approx((80 - 70) / 30, abs=0.001)
    assert first == "b"
    assert second == "a"

def test_pet_overlap_returns_none():
    """Test overlapping occupancy returns None."""
    result = _compute_pet_from_windows(
        a_entry=80, a_exit=95,
        b_entry=90, b_exit=105,
        fps=30.0
    )
    assert result is None

def test_pet_zero_on_boundary():
    """Test PET when exit and entry occur on same frame."""
    pet, _, _, _ = _compute_pet_from_windows(
        a_entry=80, a_exit=90,
        b_entry=90, b_exit=100,
        fps=30.0
    )
    assert pet == pytest.approx(0.0, abs=0.001)

def test_pet_negative_rejected():
    """Test that negative PET is not computed (overlap check)."""
    result = _compute_pet_from_windows(
        a_entry=80, a_exit=95,
        b_entry=85, b_exit=100,
        fps=30.0
    )
    assert result is None

# ============================================
# 2. Entry/Exit Frames Tests
# ============================================

def test_entry_exit_frames_basic():
    """Test basic entry/exit frame detection."""
    points = [
        TrackPoint(frame=50, x=50, y=50, cls_id=0, cls_name='car', conf=0.9),
        TrackPoint(frame=60, x=100, y=100, cls_id=0, cls_name='car', conf=0.9),
        TrackPoint(frame=65, x=110, y=95, cls_id=0, cls_name='car', conf=0.9),
        TrackPoint(frame=70, x=150, y=100, cls_id=0, cls_name='car', conf=0.9)
    ]
    window = _entry_exit_frames(points, cx=100, cy=100, half_size=20)
    assert window == (60, 65)

def test_entry_exit_frames_no_inside():
    """Test when no points are inside the zone."""
    points = [
        TrackPoint(frame=50, x=0, y=0, cls_id=0, cls_name='car', conf=0.9),
        TrackPoint(frame=60, x=200, y=200, cls_id=0, cls_name='car', conf=0.9)
    ]
    window = _entry_exit_frames(points, cx=100, cy=100, half_size=20)
    assert window is None

def test_entry_exit_frames_boundary():
    """Test points exactly on boundary are inside."""
    points = [
        TrackPoint(frame=50, x=80, y=100, cls_id=0, cls_name='car', conf=0.9),
        TrackPoint(frame=60, x=120, y=100, cls_id=0, cls_name='car', conf=0.9)
    ]
    window = _entry_exit_frames(points, cx=100, cy=100, half_size=20)
    assert window == (50, 60)

# ============================================
# 3. Conflict Point Detection Tests
# ============================================

def test_pair_conflict_point_crossing():
    """Test conflict point for crossing trajectories."""
    track_a = [
        TrackPoint(frame=0, x=0, y=100, cls_id=0, cls_name='car', conf=0.9),
        TrackPoint(frame=10, x=200, y=100, cls_id=0, cls_name='car', conf=0.9)
    ]
    track_b = [
        TrackPoint(frame=0, x=100, y=0, cls_id=0, cls_name='car', conf=0.9),
        TrackPoint(frame=10, x=100, y=200, cls_id=0, cls_name='car', conf=0.9)
    ]
    conflict = _pair_conflict_point(track_a, track_b)
    assert conflict is not None
    assert abs(conflict[0] - 100) < 1.0
    assert abs(conflict[1] - 100) < 1.0

def test_pair_conflict_point_no_crossing():
    """Test no conflict point for non-intersecting trajectories."""
    track_a = [
        TrackPoint(frame=0, x=0, y=0, cls_id=0, cls_name='car', conf=0.9),
        TrackPoint(frame=10, x=200, y=0, cls_id=0, cls_name='car', conf=0.9)
    ]
    track_b = [
        TrackPoint(frame=0, x=100, y=100, cls_id=0, cls_name='car', conf=0.9),
        TrackPoint(frame=10, x=100, y=200, cls_id=0, cls_name='car', conf=0.9)
    ]
    conflict = _pair_conflict_point(track_a, track_b)
    assert conflict is None

# ============================================
# 4. Geometric Helper Tests
# ============================================

def test_line_side():
    """Test which side of a line a point is on."""
    p1 = (0, 0)
    p2 = (10, 0)
    # Point above line
    side = _line_side((5, 5), p1, p2)
    assert side > 0
    # Point below line
    side = _line_side((5, -5), p1, p2)
    assert side < 0

def test_point_in_square():
    """Test whether a point is inside a square."""
    assert _point_in_square(100, 100, cx=100, cy=100, half_size=20)
    assert _point_in_square(80, 100, cx=100, cy=100, half_size=20)
    assert not _point_in_square(130, 100, cx=100, cy=100, half_size=20)

def test_segment_intersection():
    """Test segment intersection."""
    # Crossing segments
    inter = _segment_intersection((0, 0), (10, 10), (0, 10), (10, 0))
    assert inter is not None
    assert abs(inter[0] - 5) < 1e-6
    assert abs(inter[1] - 5) < 1e-6
    # Parallel segments
    inter = _segment_intersection((0, 0), (10, 0), (0, 5), (10, 5))
    assert inter is None

def test_segment_bbox():
    """Test segment bounding box."""
    bbox = _segment_bbox((0, 0), (10, 20))
    assert bbox == (0, 0, 10, 20)

def test_bbox_overlap():
    """Test bounding box overlap."""
    box1 = (0, 0, 10, 10)
    box2 = (5, 5, 15, 15)
    assert _bbox_overlap(box1, box2)
    box3 = (20, 20, 30, 30)
    assert not _bbox_overlap(box1, box3)

def test_trackpoint_dataclass():
    """Test TrackPoint dataclass."""
    pt = TrackPoint(frame=10, x=100, y=200, cls_id=2, cls_name='auto', conf=0.85)
    assert pt.frame == 10
    assert pt.x == 100
    assert pt.y == 200
    assert pt.cls_id == 2
    assert pt.cls_name == 'auto'
    assert pt.conf == 0.85
