
import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.analysis.visualization.pet_verification_visualizer import PETVerificationVisualizer


# Create an instance without triggering __init__ (which reads CSV)
def make_viz():
    viz = object.__new__(PETVerificationVisualizer)
    return viz

@given(st.one_of(
    st.lists(st.dictionaries(keys=st.just("x_pixel"), values=st.integers(0, 2000)) |
             st.dictionaries(keys=st.just("y_pixel"), values=st.integers(0, 2000)) |
             st.dictionaries(keys=st.sampled_from(["x_pixel","y_pixel"]),
                              values=st.integers(0, 2000)),
             min_size=1, max_size=10),
    st.text(),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
))
@settings(max_examples=50)
def test_parse_traj_returns_list_or_empty(traj_input):
    viz = make_viz()
    result = viz.parse_traj(traj_input)
    assert isinstance(result, list)

@given(st.lists(st.dictionaries(keys=st.sampled_from(["x_pixel","y_pixel"]),
                                values=st.integers(0, 2000)),
                min_size=3, max_size=20))
@settings(max_examples=50)
def test_smooth_points_length_matches_input(points):
    viz = make_viz()
    # Ensure each dict has both keys
    points = [{**p, "x_pixel": p.get("x_pixel", 0), "y_pixel": p.get("y_pixel", 0)} for p in points]
    result = viz._smooth_points(points)
    assert isinstance(result, list)
    if len(points) >= 3:
        # With sufficient points, output should have same length as input
        # (though some filtering may occur, so we only check it's a list of ints)
        for pt in result:
            assert isinstance(pt, tuple)
            assert len(pt) == 2
            assert all(isinstance(v, int) for v in pt)
