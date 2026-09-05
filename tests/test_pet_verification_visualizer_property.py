
import json

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.analysis.visualization.pet_verification_visualizer import PETVerificationVisualizer


# We only need an instance for pure methods; no CSV/video required
def make_visualizer():
    return PETVerificationVisualizer.__new__(PETVerificationVisualizer)

@given(st.lists(st.integers(min_value=0, max_value=100), max_size=10))
def test_parse_traj_list_input(traj_list):
    viz = make_visualizer()
    assert viz.parse_traj(traj_list) == traj_list

@given(st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=10))
def test_parse_traj_json_string(traj_list):
    viz = make_visualizer()
    json_str = json.dumps(traj_list)
    result = viz.parse_traj(json_str)
    assert isinstance(result, list)
    assert result == traj_list

@given(st.text(max_size=50))
def test_parse_traj_invalid_string_returns_list(s):
    viz = make_visualizer()
    result = viz.parse_traj(s)
    assert isinstance(result, list)

@given(st.lists(st.fixed_dictionaries({
        'frame': st.integers(min_value=0, max_value=100),
        'x_pixel': st.floats(min_value=0.0, max_value=1000.0),
        'y_pixel': st.floats(min_value=0.0, max_value=1000.0),
    }), min_size=3, max_size=30))
def test_smooth_points_preserves_length(points):
    viz = make_visualizer()
    smoothed = viz._smooth_points(points)
    assert len(smoothed) == len(points)
    assert all(isinstance(pt, tuple) and len(pt) == 2 and all(isinstance(v, int) for v in pt) for pt in smoothed)

@given(st.lists(st.fixed_dictionaries({
        'frame': st.integers(min_value=0, max_value=100),
        'x_pixel': st.floats(min_value=0.0, max_value=1000.0),
        'y_pixel': st.floats(min_value=0.0, max_value=1000.0),
    }), min_size=1, max_size=2))
def test_smooth_points_short_traj(points):
    viz = make_visualizer()
    smoothed = viz._smooth_points(points)
    # With <3 points, it just converts to int tuples (if keys exist)
    assert len(smoothed) == len(points)
    assert all(isinstance(pt, tuple) and len(pt) == 2 for pt in smoothed)

@given(st.lists(st.fixed_dictionaries({
        'frame': st.integers(min_value=0, max_value=100),
        'x_pixel': st.floats(min_value=0.0, max_value=1000.0),
        'y_pixel': st.floats(min_value=0.0, max_value=1000.0),
    }), min_size=1, max_size=10),
    st.integers(min_value=0, max_value=100))
def test_get_position_at(points, frame_idx):
    viz = make_visualizer()
    pos = viz._get_position_at(points, frame_idx)
    assert pos is None or (isinstance(pos, tuple) and len(pos) == 2 and all(isinstance(v, int) for v in pos))

def test_get_event_value_priority():
    viz = make_visualizer()
    event = pd.Series({'first_track_id': 1, 'track_a': 2, 'other': 5})
    assert viz._get_event_value(event, ['first_track_id', 'track_a'], -1) == 1
    assert viz._get_event_value(event, ['missing', 'track_a'], -1) == 2
    assert viz._get_event_value(event, ['missing1', 'missing2'], -1) == -1
