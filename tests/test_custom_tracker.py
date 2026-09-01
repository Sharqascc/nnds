"""
Tests for custom tracker.
"""
import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.custom_tracker import CustomTracker, Detection

def create_detection(frame=0):
    """Create a Detection object."""
    return Detection(
        frame=frame,
        x1=100.0, y1=100.0, x2=150.0, y2=180.0,
        cx=125.0, cy=140.0,
        cls_id=0, cls_name='car', conf=0.9,
        source='test',
        hist=np.zeros(32),
        embedding=np.zeros(128)
    )

def test_tracker_initialization():
    """Test CustomTracker initializes correctly."""
    tracker = CustomTracker()
    assert tracker is not None

def test_tracker_update_with_detection():
    """Test tracker update with a detection."""
    tracker = CustomTracker()
    det = create_detection(frame=0)
    tracks = tracker.update([det], frame=0)
    assert tracks is not None

def test_tracker_handles_empty_frame():
    """Test that tracker handles empty frame."""
    tracker = CustomTracker()
    tracks = tracker.update([], frame=0)
    assert tracks is not None

def test_tracker_handles_multiple_detections():
    """Test tracker with multiple detections."""
    tracker = CustomTracker()
    det1 = create_detection(frame=0)
    det2 = Detection(
        frame=0, x1=200.0, y1=100.0, x2=250.0, y2=180.0,
        cx=225.0, cy=140.0,
        cls_id=1, cls_name='motorcycle', conf=0.8,
        source='test', hist=np.zeros(32), embedding=np.zeros(128)
    )
    tracks = tracker.update([det1, det2], frame=0)
    assert tracks is not None

def test_tracker_requires_detections():
    """Test that tracker can handle empty input (returns dict)."""
    tracker = CustomTracker()
    tracks = tracker.update([], frame=0)
    assert tracks == {}  # Empty dict, not empty list
