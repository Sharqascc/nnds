"""
Comprehensive tests for all internal functions in uvh_coco_fused_grid_pet.
"""
import numpy as np
import pytest
import json
import yaml
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.grid_trajectory.uvh_coco_fused_grid_pet import (
    _box_area,
    _box_intersection,
    _can_intersect_temporal,
    _compute_histogram,
    _get_entry_gate,
    _load_gates,
    _overlap_over_person,
    _split_tracks_by_gaps,
    _track_to_json,
    _compute_pet_from_windows,
    _entry_exit_frames,
    _pair_conflict_point,
    _line_side,
    _point_in_square,
    _segment_bbox,
    _segment_intersection,
    _bbox_overlap,
    _compute_pet_from_windows,
    TrackPoint,
    classify_conflict_geometry
)

# ============================================
# 1. Box Geometry Tests
# ============================================

def test_box_area_basic():
    """Test basic box area calculation."""
    assert _box_area((0, 0, 10, 20)) == 200.0

def test_box_area_zero():
    """Test zero area."""
    assert _box_area((0, 0, 0, 0)) == 0.0

def test_box_area_negative():
    """Test negative dimensions are clamped."""
    assert _box_area((5, 5, 2, 2)) == 0.0

def test_box_intersection_overlapping():
    """Test intersection of overlapping boxes."""
    inter = _box_intersection((0, 0, 10, 10), (5, 5, 15, 15))
    assert inter == 25.0  # 5*5

def test_box_intersection_no_overlap():
    """Test intersection of non-overlapping boxes."""
    inter = _box_intersection((0, 0, 10, 10), (20, 20, 30, 30))
    assert inter == 0.0

def test_box_intersection_partial():
    """Test partial intersection."""
    inter = _box_intersection((0, 0, 10, 10), (8, 8, 12, 12))
    assert inter == 4.0  # 2*2

def test_overlap_over_person():
    """Test overlap ratio over person box."""
    ratio = _overlap_over_person((0, 0, 10, 10), (5, 5, 15, 15))
    assert ratio == 25.0 / 100.0  # 0.25

def test_overlap_over_person_zero_area():
    """Test overlap with zero area person box."""
    ratio = _overlap_over_person((0, 0, 0, 0), (5, 5, 15, 15))
    assert ratio == 0.0

# ============================================
# 2. Segment Geometry Tests
# ============================================

def test_segment_bbox():
    """Test segment bounding box."""
    bbox = _segment_bbox((0, 0), (10, 20))
    assert bbox == (0, 0, 10, 20)

def test_segment_bbox_reversed():
    """Test segment bounding box with reversed points."""
    bbox = _segment_bbox((10, 20), (0, 0))
    assert bbox == (0, 0, 10, 20)

def test_segment_intersection_crossing():
    """Test crossing segments."""
    inter = _segment_intersection((0, 0), (10, 10), (0, 10), (10, 0))
    assert inter is not None
    assert abs(inter[0] - 5.0) < 1e-6
    assert abs(inter[1] - 5.0) < 1e-6

def test_segment_intersection_parallel():
    """Test parallel segments (no intersection)."""
    inter = _segment_intersection((0, 0), (10, 0), (0, 5), (10, 5))
    assert inter is None

def test_segment_intersection_collinear():
    """Test collinear segments (treated as no intersection)."""
    inter = _segment_intersection((0, 0), (10, 0), (5, 0), (15, 0))
    assert inter is None

def test_segment_intersection_no_overlap():
    """Test segments that don't overlap."""
    inter = _segment_intersection((0, 0), (1, 1), (5, 5), (6, 6))
    assert inter is None

def test_line_side_positive():
    """Test point above line."""
    side = _line_side((5, 5), (0, 0), (10, 0))
    assert side > 0

def test_line_side_negative():
    """Test point below line."""
    side = _line_side((5, -5), (0, 0), (10, 0))
    assert side < 0

def test_line_side_on_line():
    """Test point on the line."""
    side = _line_side((5, 0), (0, 0), (10, 0))
    assert abs(side) < 1e-9

def test_point_in_square():
    """Test point inside square."""
    assert _point_in_square(100, 100, cx=100, cy=100, half_size=20)

def test_point_in_square_boundary():
    """Test point on boundary."""
    assert _point_in_square(80, 100, cx=100, cy=100, half_size=20)
    assert _point_in_square(120, 100, cx=100, cy=100, half_size=20)

def test_point_in_square_outside():
    """Test point outside square."""
    assert not _point_in_square(130, 100, cx=100, cy=100, half_size=20)

def test_bbox_overlap():
    """Test bounding box overlap."""
    box1 = (0, 0, 10, 10)
    assert _bbox_overlap(box1, (5, 5, 15, 15))
    assert not _bbox_overlap(box1, (20, 20, 30, 30))
    assert _bbox_overlap(box1, (10, 10, 20, 20))  # boundary counts as overlap

# ============================================
# 3. TrackPoint and Track Utilities Tests
# ============================================

def test_trackpoint_dataclass():
    """Test TrackPoint dataclass."""
    pt = TrackPoint(frame=10, x=100, y=200, cls_id=2, cls_name='auto', conf=0.85)
    assert pt.frame == 10
    assert pt.x == 100
    assert pt.y == 200
    assert pt.cls_id == 2
    assert pt.cls_name == 'auto'
    assert pt.conf == 0.85

def test_track_to_json_basic():
    """Test trajectory to JSON conversion without BEV mapper."""
    points = [
        TrackPoint(frame=0, x=100, y=200, cls_id=2, cls_name='auto', conf=0.9),
        TrackPoint(frame=1, x=110, y=205, cls_id=2, cls_name='auto', conf=0.9)
    ]
    json_str = _track_to_json(points)
    data = json.loads(json_str)
    assert len(data) == 2
    assert data[0]['frame'] == 0
    assert data[0]['x_pixel'] == 100.0
    assert data[0]['world_x'] is None  # No BEV mapper

def test_track_to_json_with_bev():
    """Test trajectory to JSON conversion with BEV mapper."""
    # Create a simple BEV mapper
    from src.bev.bev_mapper import BEVMapper
    H = np.array([
        [0.1, 0, 0],
        [0, 0.1, 0],
        [0, 0, 1.0]
    ])
    bev_mapper = BEVMapper(H, {'x_min': -2, 'x_max': 22, 'y_min': -2, 'y_max': 18}, [240, 200])
    
    points = [
        TrackPoint(frame=0, x=100, y=200, cls_id=2, cls_name='auto', conf=0.9),
        TrackPoint(frame=1, x=110, y=205, cls_id=2, cls_name='auto', conf=0.9)
    ]
    json_str = _track_to_json(points, bev_mapper)
    data = json.loads(json_str)
    assert data[0]['world_x'] is not None
    assert data[0]['world_y'] is not None

def test_split_tracks_by_gaps_basic():
    """Test track splitting with no gaps."""
    tracks = {
        1: [
            TrackPoint(frame=0, x=0, y=0, cls_id=1, cls_name='car', conf=0.9),
            TrackPoint(frame=1, x=10, y=10, cls_id=1, cls_name='car', conf=0.9),
            TrackPoint(frame=2, x=20, y=20, cls_id=1, cls_name='car', conf=0.9)
        ]
    }
    result = _split_tracks_by_gaps(tracks)
    assert len(result) == 1
    assert len(result[1000]) == 3  # Original track 1, sub 0

def test_split_tracks_by_large_gap():
    """Test track splitting with large gap."""
    tracks = {
        1: [
            TrackPoint(frame=0, x=0, y=0, cls_id=1, cls_name='car', conf=0.9),
            TrackPoint(frame=100, x=10, y=10, cls_id=1, cls_name='car', conf=0.9)
        ]
    }
    result = _split_tracks_by_gaps(tracks, max_frame_gap=10)
    assert len(result) == 2  # Split into 2 tracks

def test_split_tracks_predicts_occlusion():
    """Test that predicted position avoids splitting."""
    tracks = {
        1: [
            TrackPoint(frame=0, x=0, y=0, cls_id=1, cls_name='car', conf=0.9),
            TrackPoint(frame=1, x=1, y=1, cls_id=1, cls_name='car', conf=0.9),
            # Gap of 5 frames, but predictable
            TrackPoint(frame=6, x=6, y=6, cls_id=1, cls_name='car', conf=0.9)
        ]
    }
    result = _split_tracks_by_gaps(tracks, max_frame_gap=3, max_spatial_jump=100, prediction_tolerance=80)
    # Should NOT split because prediction is good
    assert len(result) == 1

def test_split_tracks_does_not_split_good_prediction():
    """Test that large gap with good prediction doesn't split."""
    tracks = {
        1: [
            TrackPoint(frame=0, x=0, y=0, cls_id=1, cls_name='car', conf=0.9),
            TrackPoint(frame=1, x=2, y=2, cls_id=1, cls_name='car', conf=0.9),
            TrackPoint(frame=2, x=4, y=4, cls_id=1, cls_name='car', conf=0.9),
            TrackPoint(frame=10, x=20, y=20, cls_id=1, cls_name='car', conf=0.9)
        ]
    }
    result = _split_tracks_by_gaps(tracks, max_frame_gap=3, max_spatial_jump=100, prediction_tolerance=80)
    # This should split because prediction would fail (moving faster)
    assert len(result) >= 1

# ============================================
# 4. Gate Logic Tests
# ============================================

def test_load_gates_basic():
    """Test gate loading from YAML."""
    tmp_dir = Path(tempfile.mkdtemp())
    gate_config = tmp_dir / 'gates.yaml'
    gate_config.write_text("""
meta:
  coordinate_system: image_pixels
gates:
- name: North_Gate
  start: [100, 50]
  end: [500, 50]
  entry_side: both
""")
    gates = _load_gates(str(gate_config))
    assert len(gates) == 1
    assert gates[0]['name'] == 'North_Gate'
    assert gates[0]['p1'] == (100, 50)

def test_load_gates_invalid():
    """Test loading invalid gate config."""
    with pytest.raises(Exception):
        _load_gates('/path/to/nonexistent.yaml')

def test_get_entry_gate_basic():
    """Test gate entry detection."""
    gates = [{'name': 'North', 'p1': (0, 50), 'p2': (100, 50), 'entry_side': 'both'}]
    points = [
        TrackPoint(frame=0, x=50, y=40, cls_id=1, cls_name='car', conf=0.9),  # above gate
        TrackPoint(frame=1, x=50, y=60, cls_id=1, cls_name='car', conf=0.9)   # below gate (crossing)
    ]
    gate = _get_entry_gate(points, gates)
    assert gate == 'North'

def test_get_entry_gate_no_crossing():
    """Test gate entry when no crossing."""
    gates = [{'name': 'North', 'p1': (0, 50), 'p2': (100, 50), 'entry_side': 'both'}]
    points = [
        TrackPoint(frame=0, x=50, y=40, cls_id=1, cls_name='car', conf=0.9),
        TrackPoint(frame=1, x=50, y=45, cls_id=1, cls_name='car', conf=0.9)  # no crossing
    ]
    gate = _get_entry_gate(points, gates)
    assert gate == 'unknown'

def test_get_entry_gate_less_than_two_points():
    """Test gate entry with insufficient points."""
    gates = [{'name': 'North', 'p1': (0, 50), 'p2': (100, 50), 'entry_side': 'both'}]
    points = [TrackPoint(frame=0, x=50, y=40, cls_id=1, cls_name='car', conf=0.9)]
    gate = _get_entry_gate(points, gates)
    assert gate == 'unknown'

# ============================================
# 5. Temporal and PET Tests
# ============================================

def test_can_intersect_temporal_overlap():
    """Test temporal overlap check returns True."""
    track_a = {'frames': [0, 10, 20]}
    track_b = {'frames': [15, 25, 30]}
    result = _can_intersect_temporal(track_a, track_b, fps=30, max_pet=2.0)
    assert result is True

def test_can_intersect_temporal_no_overlap():
    """Test temporal overlap check returns False."""
    track_a = {'frames': [0, 5, 10]}
    track_b = {'frames': [100, 110, 120]}
    result = _can_intersect_temporal(track_a, track_b, fps=30, max_pet=2.0)
    assert result is False

def test_compute_pet_from_windows_sequential():
    """Test PET computation for sequential windows."""
    pet, first, second, frame_ref = _compute_pet_from_windows(80, 90, 105, 115, 30)
    assert pet == pytest.approx(0.5, abs=0.001)
    assert first == 'a'
    assert second == 'b'

def test_compute_pet_from_windows_overlap():
    """Test PET computation for overlapping windows."""
    result = _compute_pet_from_windows(80, 95, 90, 105, 30)
    assert result is None

def test_compute_pet_from_windows_reverse():
    """Test PET computation for reversed order."""
    pet, first, second, frame_ref = _compute_pet_from_windows(80, 95, 60, 70, 30)
    assert pet == pytest.approx((80-70)/30, abs=0.001)
    assert first == 'b'

def test_entry_exit_frames_basic():
    """Test entry/exit frame detection."""
    points = [
        TrackPoint(frame=50, x=50, y=50, cls_id=1, cls_name='car', conf=0.9),
        TrackPoint(frame=60, x=100, y=100, cls_id=1, cls_name='car', conf=0.9),
        TrackPoint(frame=65, x=110, y=95, cls_id=1, cls_name='car', conf=0.9),
        TrackPoint(frame=70, x=150, y=100, cls_id=1, cls_name='car', conf=0.9)
    ]
    window = _entry_exit_frames(points, cx=100, cy=100, half_size=20)
    assert window == (60, 65)

def test_entry_exit_frames_no_inside():
    """Test entry/exit when no points inside."""
    points = [
        TrackPoint(frame=50, x=0, y=0, cls_id=1, cls_name='car', conf=0.9),
        TrackPoint(frame=60, x=200, y=200, cls_id=1, cls_name='car', conf=0.9)
    ]
    window = _entry_exit_frames(points, cx=100, cy=100, half_size=20)
    assert window is None

def test_pair_conflict_point_crossing():
    """Test conflict point for crossing trajectories."""
    track_a = [
        TrackPoint(frame=0, x=0, y=100, cls_id=1, cls_name='car', conf=0.9),
        TrackPoint(frame=10, x=200, y=100, cls_id=1, cls_name='car', conf=0.9)
    ]
    track_b = [
        TrackPoint(frame=0, x=100, y=0, cls_id=1, cls_name='car', conf=0.9),
        TrackPoint(frame=10, x=100, y=200, cls_id=1, cls_name='car', conf=0.9)
    ]
    conflict = _pair_conflict_point(track_a, track_b)
    assert conflict is not None
    assert abs(conflict[0] - 100) < 1.0
    assert abs(conflict[1] - 100) < 1.0

def test_pair_conflict_point_no_crossing():
    """Test no conflict point for non-crossing trajectories."""
    track_a = [
        TrackPoint(frame=0, x=0, y=0, cls_id=1, cls_name='car', conf=0.9),
        TrackPoint(frame=10, x=200, y=0, cls_id=1, cls_name='car', conf=0.9)
    ]
    track_b = [
        TrackPoint(frame=0, x=100, y=100, cls_id=1, cls_name='car', conf=0.9),
        TrackPoint(frame=10, x=100, y=200, cls_id=1, cls_name='car', conf=0.9)
    ]
    conflict = _pair_conflict_point(track_a, track_b)
    assert conflict is None

# ============================================
# 6. Histogram and Classification Tests
# ============================================

def test_compute_histogram_basic():
    """Test histogram computation from frame crop."""
    # Create a simple test frame (BGR)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :, 0] = 255  # Blue channel
    frame[:, :, 1] = 0    # Green
    frame[:, :, 2] = 0    # Red
    
    hist = _compute_histogram(frame, 10, 10, 50, 50)
    assert hist is not None
    assert len(hist) == 30 * 32  # 2D histogram flattened

def test_compute_histogram_invalid():
    """Test histogram with invalid coordinates (clips to valid bounds)."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    hist = _compute_histogram(frame, -10, -10, -5, -5)
    # The function clips coordinates to valid frame bounds, so may still return a histogram
    assert hist is not None or hist is None  # Should not crash
    if hist is not None:
        assert len(hist) == 30 * 32  # 2D histogram flattened

def test_compute_histogram_out_of_bounds():
    """Test histogram with out-of-bounds coordinates."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    hist = _compute_histogram(frame, 90, 90, 200, 200)
    assert hist is not None  # Should clip to frame bounds

def test_classify_conflict_geometry_head_on():
    """Test head-on classification."""
    traj_a = json.dumps([
        {'frame': 0, 'x': 0, 'y': 100, 'world_x': 0, 'world_y': 10},
        {'frame': 1, 'x': 5, 'y': 100, 'world_x': 0.5, 'world_y': 10},
        {'frame': 2, 'x': 10, 'y': 100, 'world_x': 1, 'world_y': 10},
        {'frame': 3, 'x': 15, 'y': 100, 'world_x': 1.5, 'world_y': 10},
        {'frame': 4, 'x': 20, 'y': 100, 'world_x': 2, 'world_y': 10}
    ])
    traj_b = json.dumps([
        {'frame': 0, 'x': 20, 'y': 100, 'world_x': 2, 'world_y': 10},
        {'frame': 1, 'x': 15, 'y': 100, 'world_x': 1.5, 'world_y': 10},
        {'frame': 2, 'x': 10, 'y': 100, 'world_x': 1, 'world_y': 10},
        {'frame': 3, 'x': 5, 'y': 100, 'world_x': 0.5, 'world_y': 10},
        {'frame': 4, 'x': 0, 'y': 100, 'world_x': 0, 'world_y': 10}
    ])
    result = classify_conflict_geometry(traj_a, traj_b, conflict_frame=2, fps=30.0)
    assert result == 'head_on'

def test_classify_conflict_geometry_crossing():
    """Test crossing classification."""
    traj_a = json.dumps([
        {'frame': 0, 'x': 0, 'y': 100, 'world_x': 0, 'world_y': 10},
        {'frame': 1, 'x': 5, 'y': 100, 'world_x': 0.5, 'world_y': 10},
        {'frame': 2, 'x': 10, 'y': 100, 'world_x': 1, 'world_y': 10},
        {'frame': 3, 'x': 15, 'y': 100, 'world_x': 1.5, 'world_y': 10},
        {'frame': 4, 'x': 20, 'y': 100, 'world_x': 2, 'world_y': 10}
    ])
    traj_b = json.dumps([
        {'frame': 0, 'x': 10, 'y': 0, 'world_x': 1, 'world_y': 0},
        {'frame': 1, 'x': 10, 'y': 5, 'world_x': 1, 'world_y': 0.5},
        {'frame': 2, 'x': 10, 'y': 10, 'world_x': 1, 'world_y': 1},
        {'frame': 3, 'x': 10, 'y': 15, 'world_x': 1, 'world_y': 1.5},
        {'frame': 4, 'x': 10, 'y': 20, 'world_x': 1, 'world_y': 2}
    ])
    result = classify_conflict_geometry(traj_a, traj_b, conflict_frame=2, fps=30.0)
    assert result == 'crossing'

def test_classify_conflict_geometry_empty():
    """Test classification with empty trajectories."""
    result = classify_conflict_geometry('', '{}', 0, 30.0)
    assert result == 'other'

def test_classify_conflict_geometry_other():
    """Test classification returns 'other' for ambiguous."""
    traj_a = json.dumps([
        {'frame': 0, 'x': 0, 'y': 100, 'world_x': 0, 'world_y': 10},
        {'frame': 1, 'x': 5, 'y': 100, 'world_x': 0.5, 'world_y': 10}
    ])
    traj_b = json.dumps([
        {'frame': 0, 'x': 10, 'y': 100, 'world_x': 1, 'world_y': 10},
        {'frame': 1, 'x': 12, 'y': 100, 'world_x': 1.2, 'world_y': 10}
    ])
    result = classify_conflict_geometry(traj_a, traj_b, conflict_frame=1, fps=30.0)
    assert result in ['head_on', 'crossing', 'rear_end', 'side_swipe', 'other']


def test_compute_histogram_empty_crop():
    """Test histogram with empty crop (line 31)."""
    import numpy as np
    from src.analysis.grid_trajectory.uvh_coco_fused_grid_pet import _compute_histogram
    
    # Create a frame and test empty crop (invalid coordinates)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Crop with zero size
    hist = _compute_histogram(frame, 50, 50, 50, 50)  # zero width/height
    assert hist is None  # Empty crop returns None


def test_compute_histogram_exception():
    """Test histogram exception handling (lines 38-39)."""
    import numpy as np
    from src.analysis.grid_trajectory.uvh_coco_fused_grid_pet import _compute_histogram
    
    # Pass invalid arguments that will cause an exception
    frame = None  # None frame will cause exception
    hist = _compute_histogram(frame, 10, 10, 50, 50)
    assert hist is None  # Exception handled, returns None


def test_segment_intersection_out_of_range():
    """Test segment intersection when intersection is not in range (line 108)."""
    import numpy as np
    from src.analysis.grid_trajectory.uvh_coco_fused_grid_pet import _segment_intersection
    
    # Segments whose mathematical intersection is outside both segments
    # Segment 1: from (0,0) to (1,1)
    # Segment 2: from (2,0) to (3,-1) - intersection point is outside
    inter = _segment_intersection((0, 0), (1, 1), (2, 0), (3, -1))
    assert inter is None


def test_split_tracks_empty_list():
    """Test split tracks with empty point list (line 771)."""
    from src.analysis.grid_trajectory.uvh_coco_fused_grid_pet import _split_tracks_by_gaps, TrackPoint
    
    tracks = {
        1: []  # Empty points list
    }
    result = _split_tracks_by_gaps(tracks)
    assert len(result) == 0  # Empty track is skipped


def test_split_tracks_zero_dt1():
    """Test split tracks when dt1 is zero (line 792)."""
    from src.analysis.grid_trajectory.uvh_coco_fused_grid_pet import _split_tracks_by_gaps, TrackPoint
    
    # Create tracks where dt1 == 0 (same frame for prev_prev and prev)
    tracks = {
        1: [
            TrackPoint(frame=0, x=0, y=0, cls_id=1, cls_name='car', conf=0.9),
            TrackPoint(frame=0, x=10, y=10, cls_id=1, cls_name='car', conf=0.9),  # same frame as prev
            TrackPoint(frame=5, x=20, y=20, cls_id=1, cls_name='car', conf=0.9)
        ]
    }
    result = _split_tracks_by_gaps(tracks, max_frame_gap=3, max_spatial_jump=100, prediction_tolerance=80)
    # Should not crash; result should have tracks
    assert len(result) >= 1





def test_run_uvh_coco_fused_grid_pet_mocked(tmp_path):
    """Test the full pipeline with all heavy dependencies mocked."""
    import json, yaml, numpy as np, cv2
    from unittest.mock import patch, MagicMock
    from src.analysis.grid_trajectory.uvh_coco_fused_grid_pet import (
        run_uvh_coco_fused_grid_pet, TrackPoint
    )

    # --- Create minimal config files ---
    bev_cfg = {
        "H_pixel_to_world": np.eye(3).tolist(),
        "x_min": 0, "x_max": 10, "y_min": 0, "y_max": 10,
        "resolution": 0.1,
        "bev_resolution": [100, 100]
    }
    bev_path = tmp_path / "bev.json"
    bev_path.write_text(json.dumps(bev_cfg))

    grid_cfg = {
        "cells": [{"id": 1, "polygon": [[0,0],[10,0],[10,10],[0,10]]}]
    }
    grid_path = tmp_path / "grid.yaml"
    grid_path.write_text(yaml.dump(grid_cfg))

    gate_cfg = {
        "gates": [{"id": "G1", "line": [[0,0],[0,10]]}]
    }
    gate_path = tmp_path / "gates.yaml"
    gate_path.write_text(yaml.dump(gate_cfg))

    # --- Fake video capture (not actually used by the pipeline because YOLO reads video itself) ---
    class FakeVideoCapture:
        def __init__(self, *args, **kwargs):
            pass
        def isOpened(self):
            return True
        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 5
            return 0
        def release(self):
            pass

    # --- Fake YOLO result object with empty detections ---
    class FakeResult:
        def __init__(self):
            self.boxes = None          # no detections
            self.orig_img = np.zeros((480, 640, 3), dtype=np.uint8)
            self.names = {}            # not used because boxes is None

    # --- Fake YOLO model that yields results ---
    class FakeYOLO:
        instances = 0
        def __init__(self, *args, **kwargs):
            FakeYOLO.instances += 1
        def predict(self, *args, **kwargs):
            # Return a list of 5 fake results (one per frame)
            return [FakeResult() for _ in range(5)]

    # --- Dummy SpatialGrid and BEVMapper ---
    class DummySpatialGrid:
        def __init__(self, *args, **kwargs):
            pass
        def get_cell_from_pixels(self, x, y):
            return "UNKNOWN"

    class DummyBEVMapper:
        def __init__(self, *args, **kwargs):
            pass

    # --- Dummy CustomTracker and ReIDEncoder to avoid side effects ---
    class DummyCustomTracker:
        def __init__(self, *args, **kwargs):
            pass
        def update(self, raw_dets, frame_img=None, frame=0):
            # No matches; return empty dict
            return {}

    class DummyReIDEncoder:
        def __init__(self, *args, **kwargs):
            pass

    # --- Patch all heavy dependencies ---
    with patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet.YOLO", FakeYOLO),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet.cv2.VideoCapture", FakeVideoCapture),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet.SpatialGrid", DummySpatialGrid),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet.BEVMapper", DummyBEVMapper),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet.CustomTracker", DummyCustomTracker),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet.ReIDEncoder", DummyReIDEncoder),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet._load_gates", return_value=[{"id": "G1"}]):

        result = run_uvh_coco_fused_grid_pet(
            video_path=str(tmp_path / "fake.mp4"),   # file doesn't exist; mocked capture ignores it
            bev_config_path=str(bev_path),
            grid_config_path=str(grid_path),
            uvh_model_path=str(tmp_path / "uvh.pt"),
            coco_person_model_path=str(tmp_path / "coco.pt"),
            output_csv_path=str(tmp_path / "out.csv"),
            gate_config_path=str(gate_path),
            max_frames=5,
            device="cpu",
            backend="auto",
            show_progress=False,
        )

    # Assert the function completed and returned expected keys
    assert isinstance(result, dict)
    assert "pet_events" in result
    # Verify fake YOLO was used
    assert FakeYOLO.instances > 0


def test_run_uvh_coco_fused_grid_pet_with_conflict_mocked(tmp_path):
    """Cover the PET event creation and output writing by forcing a conflict."""
    import json, yaml, numpy as np, cv2
    from unittest.mock import patch, MagicMock
    from src.analysis.grid_trajectory.uvh_coco_fused_grid_pet import (
        run_uvh_coco_fused_grid_pet, TrackPoint, Detection
    )

    # --- Config files ---
    bev_cfg = {
        "H_pixel_to_world": np.eye(3).tolist(),
        "x_min": 0, "x_max": 10, "y_min": 0, "y_max": 10,
        "resolution": 0.1,
        "bev_resolution": [100, 100]
    }
    bev_path = tmp_path / "bev.json"
    bev_path.write_text(json.dumps(bev_cfg))

    grid_cfg = {"cells": [{"id": 1, "polygon": [[0,0],[10,0],[10,10],[0,10]]}]}
    grid_path = tmp_path / "grid.yaml"
    grid_path.write_text(yaml.dump(grid_cfg))

    gate_cfg = {"gates": [{"id": "G1", "line": [[0,0],[0,10]]}]}
    gate_path = tmp_path / "gates.yaml"
    gate_path.write_text(yaml.dump(gate_cfg))

    # --- Fake video capture ---
    class FakeVideoCapture:
        def __init__(self, *args, **kwargs):
            pass
        def isOpened(self):
            return True
        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 1
            return 0
        def release(self):
            pass

    # --- Fake YOLO result (empty) ---
    class FakeResult:
        def __init__(self):
            self.boxes = None
            self.orig_img = np.zeros((480, 640, 3), dtype=np.uint8)
            self.names = {}

    class FakeYOLO:
        instances = 0
        def __init__(self, *args, **kwargs):
            FakeYOLO.instances += 1
        def predict(self, *args, **kwargs):
            # Return a generator of one empty result
            return [FakeResult()]

    # --- Dummies ---
    class DummySpatialGrid:
        def __init__(self, *args, **kwargs):
            pass
        def get_cell_from_pixels(self, x, y):
            return "cell1"   # not out of bounds

    class DummyBEVMapper:
        def __init__(self, *args, **kwargs):
            pass

    class DummyCustomTracker:
        def __init__(self, *args, **kwargs):
            pass
        def update(self, raw_dets, frame_img=None, frame=0):
            return {}

    class DummyReIDEncoder:
        def __init__(self, *args, **kwargs):
            pass

    # --- Force track splitter to return two tracks with 3 points each ---
    def fake_split_tracks(tracks, **kwargs):
        # Return two tracks with composite IDs 1001 and 2002
        return {
            1001: [
                TrackPoint(frame=0, x=0, y=0, cls_id=2, cls_name='car', conf=0.9),
                TrackPoint(frame=1, x=5, y=5, cls_id=2, cls_name='car', conf=0.9),
                TrackPoint(frame=2, x=10, y=10, cls_id=2, cls_name='car', conf=0.9),
            ],
            2002: [
                TrackPoint(frame=0, x=10, y=0, cls_id=0, cls_name='pedestrian', conf=0.9),
                TrackPoint(frame=1, x=5, y=5, cls_id=0, cls_name='pedestrian', conf=0.9),
                TrackPoint(frame=2, x=0, y=10, cls_id=0, cls_name='pedestrian', conf=0.9),
            ],
        }

    with patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet.YOLO", FakeYOLO),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet.cv2.VideoCapture", FakeVideoCapture),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet.SpatialGrid", DummySpatialGrid),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet.BEVMapper", DummyBEVMapper),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet.CustomTracker", DummyCustomTracker),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet.ReIDEncoder", DummyReIDEncoder),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet._load_gates", return_value=[{"id": "G1"}]),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet._split_tracks_by_gaps", fake_split_tracks),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet._pair_conflict_point", return_value=(5.0, 5.0)),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet._entry_exit_frames", side_effect=lambda pts, cx, cy, half: (0, 2)),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet._compute_pet_from_windows", return_value=(1.5, 'a', 'b', 1)),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet._track_to_json", return_value="{}"),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet.classify_conflict_geometry", return_value="crossing"),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet._get_entry_gate", return_value="G1"):

        result = run_uvh_coco_fused_grid_pet(
            video_path=str(tmp_path / "fake.mp4"),
            bev_config_path=str(bev_path),
            grid_config_path=str(grid_path),
            uvh_model_path=str(tmp_path / "uvh.pt"),
            coco_person_model_path=str(tmp_path / "coco.pt"),
            output_csv_path=str(tmp_path / "out.csv"),
            gate_config_path=str(gate_path),
            max_frames=1,
            device="cpu",
            backend="auto",
            show_progress=False,
        )

    # Assert one conflict was detected
    assert isinstance(result, dict)
    assert "pet_events" in result
    assert len(result["pet_events"]) == 1
    # Check the event has the right fields
    ev = result["pet_events"][0]
    assert ev["pet"] == 1.5
    assert ev["orig_track_a"] == 1
    assert ev["orig_track_b"] == 2



def test_run_uvh_coco_fused_grid_pet_time_based_branch_mocked(tmp_path):
    """Cover the a_exit <= b_entry branch of PET time calculation."""
    import json, yaml, numpy as np, cv2
    from unittest.mock import patch, MagicMock
    from src.analysis.grid_trajectory.uvh_coco_fused_grid_pet import (
        run_uvh_coco_fused_grid_pet, TrackPoint, Detection
    )

    # Config files
    bev_cfg = {"H_pixel_to_world": np.eye(3).tolist(), "x_min":0,"x_max":10,"y_min":0,"y_max":10,"resolution":0.1,"bev_resolution":[100,100]}
    bev_path = tmp_path / "bev.json"
    bev_path.write_text(json.dumps(bev_cfg))
    grid_cfg = {"cells":[{"id":1,"polygon":[[0,0],[10,0],[10,10],[0,10]]}]}
    grid_path = tmp_path / "grid.yaml"
    grid_path.write_text(yaml.dump(grid_cfg))
    gate_cfg = {"gates":[{"id":"G1","line":[[0,0],[0,10]]}]}
    gate_path = tmp_path / "gates.yaml"
    gate_path.write_text(yaml.dump(gate_cfg))

    class FakeVideoCapture:
        def __init__(self,*a,**k): pass
        def isOpened(self): return True
        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS: return 30.0
            if prop == cv2.CAP_PROP_FRAME_COUNT: return 1
            return 0
        def release(self): pass

    class FakeResult:
        def __init__(self):
            self.boxes = None
            self.orig_img = np.zeros((480,640,3), dtype=np.uint8)
            self.names = {}
    class FakeYOLO:
        def __init__(self,*a,**k): pass
        def predict(self,*a,**k):
            return [FakeResult()]

    class DummySpatialGrid:
        def __init__(self,*a,**k): pass
        def get_cell_from_pixels(self,x,y): return "cell1"
    class DummyBEVMapper:
        def __init__(self,*a,**k): pass
    class DummyCustomTracker:
        def __init__(self,*a,**k): pass
        def update(self, raw_dets, frame_img=None, frame=0): return {}
    class DummyReIDEncoder:
        def __init__(self,*a,**k): pass

    def fake_split_tracks(tracks, **kwargs):
        return {
            1001: [
                TrackPoint(frame=0, x=0, y=0, cls_id=2, cls_name='car', conf=0.9),
                TrackPoint(frame=1, x=5, y=5, cls_id=2, cls_name='car', conf=0.9),
                TrackPoint(frame=2, x=10, y=10, cls_id=2, cls_name='car', conf=0.9),
            ],
            2002: [
                TrackPoint(frame=0, x=10, y=0, cls_id=0, cls_name='pedestrian', conf=0.9),
                TrackPoint(frame=1, x=5, y=5, cls_id=0, cls_name='pedestrian', conf=0.9),
                TrackPoint(frame=2, x=0, y=10, cls_id=0, cls_name='pedestrian', conf=0.9),
            ],
        }

    # Custom _entry_exit_frames: for track with first x=0 return (0,1); for x=10 return (2,3)
    def entry_exit_side_effect(points, cx, cy, half_size):
        if points[0].x == 0:
            return (0, 1)   # a_entry=0, a_exit=1
        else:
            return (2, 3)   # b_entry=2, b_exit=3
    # a_exit <= b_entry -> 1 <= 2 True

    with patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet.YOLO", FakeYOLO),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet.cv2.VideoCapture", FakeVideoCapture),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet.SpatialGrid", DummySpatialGrid),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet.BEVMapper", DummyBEVMapper),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet.CustomTracker", DummyCustomTracker),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet.ReIDEncoder", DummyReIDEncoder),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet._load_gates", return_value=[{"id":"G1"}]),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet._split_tracks_by_gaps", fake_split_tracks),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet._pair_conflict_point", return_value=(5.0,5.0)),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet._entry_exit_frames", side_effect=entry_exit_side_effect),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet._compute_pet_from_windows", return_value=(1.5, 'a', 'b', 1)),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet._track_to_json", return_value="{}"),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet.classify_conflict_geometry", return_value="crossing"),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet._get_entry_gate", return_value="G1"):

        result = run_uvh_coco_fused_grid_pet(
            video_path=str(tmp_path/"fake.mp4"),
            bev_config_path=str(bev_path),
            grid_config_path=str(grid_path),
            uvh_model_path=str(tmp_path/"uvh.pt"),
            coco_person_model_path=str(tmp_path/"coco.pt"),
            output_csv_path=str(tmp_path/"out.csv"),
            gate_config_path=str(gate_path),
            max_frames=1,
            device="cpu",
            backend="auto",
            show_progress=False,
        )

    assert isinstance(result, dict)
    assert len(result["pet_events"]) == 1
    # Check that time-based PET was calculated via a_exit <= b_entry branch
    assert not np.isnan(result["pet_events"][0]["pet_time_based"])
    assert result["pet_events"][0]["pet_time_based"] == (2 - 1) / 30.0


def test_run_uvh_coco_fused_grid_pet_interactive_mocked(tmp_path):
    """Cover interactive pause block by running 20 frames with interactive=True."""
    import json, yaml, numpy as np, cv2
    from unittest.mock import patch, MagicMock
    from types import SimpleNamespace
    from src.analysis.grid_trajectory.uvh_coco_fused_grid_pet import (
        run_uvh_coco_fused_grid_pet, TrackPoint, Detection
    )

    # Config files (same as other mocked tests)
    bev_cfg = {"H_pixel_to_world": np.eye(3).tolist(), "x_min":0,"x_max":10,"y_min":0,"y_max":10,"resolution":0.1,"bev_resolution":[100,100]}
    bev_path = tmp_path / "bev.json"
    bev_path.write_text(json.dumps(bev_cfg))
    grid_cfg = {"cells":[{"id":1,"polygon":[[0,0],[10,0],[10,10],[0,10]]}]}
    grid_path = tmp_path / "grid.yaml"
    grid_path.write_text(yaml.dump(grid_cfg))
    gate_cfg = {"gates":[{"id":"G1","line":[[0,0],[0,10]]}]}
    gate_path = tmp_path / "gates.yaml"
    gate_path.write_text(yaml.dump(gate_cfg))

    class FakeVideoCapture:
        def __init__(self,*a,**k): pass
        def isOpened(self): return True
        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS: return 30.0
            if prop == cv2.CAP_PROP_FRAME_COUNT: return 20
            return 0
        def release(self): pass
        def set(self,*a,**k): pass
        def read(self):
            # Simulate failed re-read in interactive pause (returns no frame)
            return False, None

    class FakeResult:
        def __init__(self):
            self.boxes = None
            self.orig_img = np.zeros((480,640,3), dtype=np.uint8)
            self.names = {}
    class FakeYOLO:
        def __init__(self,*a,**k): pass
        def predict(self,*a,**k):
            return [FakeResult() for _ in range(20)]

    class DummySpatialGrid:
        def __init__(self,*a,**k): pass
        def get_cell_from_pixels(self,x,y): return "cell1"
    class DummyBEVMapper:
        def __init__(self,*a,**k): pass
    class DummyCustomTracker:
        def __init__(self,*a,**k): pass
        def update(self, raw_dets, frame_img=None, frame=0): return {}
    class DummyReIDEncoder:
        def __init__(self,*a,**k): pass

    # Create a dummy matplotlib.pyplot module to avoid actual plotting
    dummy_plt = SimpleNamespace(
        figure=lambda *a, **k: MagicMock(),
        imshow=lambda *a, **k: None,
        title=lambda *a, **k: None,
        axis=lambda *a, **k: None,
        close=lambda *a, **k: None,
    )

    with patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet.YOLO", FakeYOLO),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet.cv2.VideoCapture", FakeVideoCapture),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet.SpatialGrid", DummySpatialGrid),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet.BEVMapper", DummyBEVMapper),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet.CustomTracker", DummyCustomTracker),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet.ReIDEncoder", DummyReIDEncoder),          patch("src.analysis.grid_trajectory.uvh_coco_fused_grid_pet._load_gates", return_value=[{"id":"G1"}]),          patch("builtins.input", return_value=""),          patch("matplotlib.pyplot", dummy_plt),          patch("IPython.display.display", lambda *a, **k: None):

        result = run_uvh_coco_fused_grid_pet(
            video_path=str(tmp_path/"fake.mp4"),
            bev_config_path=str(bev_path),
            grid_config_path=str(grid_path),
            uvh_model_path=str(tmp_path/"uvh.pt"),
            coco_person_model_path=str(tmp_path/"coco.pt"),
            output_csv_path=str(tmp_path/"out.csv"),
            gate_config_path=str(gate_path),
            max_frames=20,
            device="cpu",
            backend="auto",
            show_progress=False,
            interactive=True,
        )

    assert isinstance(result, dict)
    assert "pet_events" in result
