
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.analysis.conflict_classifier import (
    _angle_between,
    _get_velocity_vector,
    classify_conflict_geometry,
)


@given(st.floats(min_value=-1000, max_value=1000),
       st.floats(min_value=-1000, max_value=1000),
       st.floats(min_value=-1000, max_value=1000),
       st.floats(min_value=-1000, max_value=1000))
def test_angle_between_range_and_symmetry(v1x, v1y, v2x, v2y):
    v1 = (v1x, v1y)
    v2 = (v2x, v2y)
    ang1 = _angle_between(v1, v2)
    ang2 = _angle_between(v2, v1)
    assert 0.0 <= ang1 <= 180.0
    assert math.isclose(ang1, ang2, rel_tol=1e-9, abs_tol=1e-9)

@given(st.lists(st.fixed_dictionaries({
        'frame': st.integers(min_value=0, max_value=100),
        'x_pixel': st.floats(min_value=-1000, max_value=1000),
        'y_pixel': st.floats(min_value=-1000, max_value=1000),
    }), max_size=10))
def test_get_velocity_vector_tuple_type(points):
    v = _get_velocity_vector(points, before_frame=100)
    assert isinstance(v, tuple)
    assert len(v) == 2
    assert all(isinstance(comp, float) for comp in v)

@given(st.lists(st.fixed_dictionaries({
        'frame': st.integers(min_value=0, max_value=100),
        'x_pixel': st.floats(min_value=-1000, max_value=1000),
        'y_pixel': st.floats(min_value=-1000, max_value=1000),
    }), max_size=1))
def test_get_velocity_vector_zero_when_insufficient(points):
    v = _get_velocity_vector(points, before_frame=100)
    assert v == (0.0, 0.0)

@given(st.text(max_size=50), st.text(max_size=50), st.integers(min_value=0, max_value=200))
def test_classify_conflict_geometry_valid_output(traj_a, traj_b, conflict_frame):
    result = classify_conflict_geometry(traj_a, traj_b, conflict_frame)
    assert result in {"rear_end", "head_on", "crossing", "side_swipe", "other"}

@pytest.mark.parametrize("bad_json", ["null", "true", "123", '"abc"'])
def test_classify_conflict_geometry_non_list_json(bad_json):
    # Should not raise, even if JSON scalar is passed
    result = classify_conflict_geometry(bad_json, "[]", 1)
    assert result in {"rear_end", "head_on", "crossing", "side_swipe", "other"}
