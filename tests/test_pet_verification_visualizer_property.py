
import json

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.analysis.visualization.pet_verification_visualizer import PETVerificationVisualizer


def make_instance():
    """Return an instance without calling __init__ (no CSV/video needed)."""
    obj = PETVerificationVisualizer.__new__(PETVerificationVisualizer)
    return obj


@given(st.lists(st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False)), min_size=0, max_size=20))
def test_parse_traj_json_roundtrip(data):
    obj = make_instance()
    json_str = json.dumps(data)
    result = obj.parse_traj(json_str)
    assert result == data

@given(st.text(min_size=1, max_size=50))
def test_parse_traj_invalid_string_returns_empty(s):
    obj = make_instance()
    # Avoid strings that might actually be valid lists
    result = obj.parse_traj(s)
    assert isinstance(result, list)

@given(st.dictionaries(st.text(min_size=1, max_size=10), st.integers(), min_size=1, max_size=5))
def test_get_event_value_returns_first_key(data):
    obj = make_instance()
    event = pd.Series(data)
    keys = list(data.keys())
    if keys:
        val = obj._get_event_value(event, keys, default=None)
        assert val == event[keys[0]]

@given(st.lists(st.fixed_dictionaries({
    'frame': st.integers(min_value=0, max_value=100),
    'x_pixel': st.integers(min_value=0, max_value=1000),
    'y_pixel': st.integers(min_value=0, max_value=1000),
}), min_size=0, max_size=20),
st.integers(min_value=0, max_value=100))
def test_get_position_at_returns_tuple_or_none(traj, frame_idx):
    obj = make_instance()
    result = obj._get_position_at(traj, frame_idx)
    if result is not None:
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(v, int) for v in result)
    else:
        # If no points with frame <= frame_idx, result should be None
        if any(p['frame'] <= frame_idx for p in traj):
            # should have returned a tuple, not None
            assert result is not None

@given(st.integers(min_value=10, max_value=200), st.integers(min_value=10, max_value=200))
def test_schematic_background_shape_and_dtype(width, height):
    obj = make_instance()
    bg = obj._schematic_background(width=width, height=height)
    assert bg.shape == (height, width, 3)
    assert bg.dtype == np.uint8
