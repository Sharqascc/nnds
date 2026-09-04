
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.pipeline.custom_tracker import CustomTracker, Detection


def detection_strategy():
    return st.builds(
        Detection,
        frame=st.integers(min_value=0, max_value=100),
        x1=st.floats(0, 10),
        y1=st.floats(0, 10),
        x2=st.floats(20, 50),
        y2=st.floats(20, 50),
        cx=st.floats(0, 50),
        cy=st.floats(0, 50),
        cls_id=st.integers(0, 10),
        cls_name=st.text(max_size=5),
        conf=st.floats(0, 1),
        source=st.text(max_size=5),
    )

@given(st.lists(detection_strategy(), max_size=20))
def test_tracker_update_returns_valid_mapping(detections):
    tracker = CustomTracker()
    matched = tracker.update(detections)
    for det_idx, track_id in matched.items():
        assert det_idx < len(detections)
        assert track_id > 0

@given(st.floats(0, 100), st.floats(0, 100), st.floats(0, 100), st.floats(0, 100))
def test_iou_range(box1_x1, box1_y1, box1_x2, box1_y2):
    tracker = CustomTracker()
    iou = tracker._iou((box1_x1, box1_y1, max(box1_x1+1, box1_x2), max(box1_y1+1, box1_y2)),
                       (box1_x1, box1_y1, max(box1_x1+10, box1_x2+10), max(box1_y1+10, box1_y2+10)))
    assert 0.0 <= iou <= 1.0
