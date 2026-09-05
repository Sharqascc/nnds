
import ast

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.analysis.pet_diffusion_analysis import (
    compute_distance_to_threshold,
    compute_error_metrics,
    parse_trajectory,
)


# ---------- parse_trajectory ----------
@given(st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                min_size=1, max_size=20))
def test_parse_trajectory_list_input(values):
    result = parse_trajectory(values)
    assert isinstance(result, np.ndarray)
    assert result.shape == (len(values),)
    assert np.all(np.isfinite(result))

@given(st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                min_size=1, max_size=20))
def test_parse_trajectory_string_input(values):
    s = str(values)
    result = parse_trajectory(s)
    assert isinstance(result, np.ndarray)
    assert result.shape == (len(values),)

@given(st.text(max_size=50))
def test_parse_trajectory_invalid_string(s):
    result = parse_trajectory(s)
    # It should not raise; either None or ndarray
    assert result is None or isinstance(result, np.ndarray)

# ---------- compute_distance_to_threshold ----------
@st.composite
def trajectory_4col(draw, min_rows=1, max_rows=20):
    rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    return draw(st.lists(st.tuples(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    ), min_size=rows, max_size=rows).map(lambda lst: np.array(lst, dtype=float)))

@given(trajectory_4col(), st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
def test_compute_distance_to_threshold_shape_and_ranges(traj, d_thresh):
    distances, first_hit = compute_distance_to_threshold(traj, d_thresh)
    assert isinstance(distances, np.ndarray)
    assert distances.shape == (len(traj),)
    assert np.all(distances >= 0.0)
    assert first_hit is None or (isinstance(first_hit, int) and 0 <= first_hit < len(traj))

@given(st.lists(st.tuples(
        st.floats(min_value=-10, max_value=10),
        st.floats(min_value=-10, max_value=10),
    ), min_size=1, max_size=10).map(lambda lst: np.array(lst, dtype=float)))
def test_compute_distance_to_threshold_invalid_shape_raises(traj):
    with pytest.raises(ValueError):
        compute_distance_to_threshold(traj)

# ---------- compute_error_metrics ----------
@given(st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                min_size=1, max_size=20))
def test_compute_error_metrics_invariants(values):
    real = np.array(values, dtype=float)
    pred = real + 1.0  # constant offset
    metrics = compute_error_metrics(real, pred)
    assert isinstance(metrics, dict)
    for key in ['mae', 'rmse', 'mse', 'mean_error', 'std_error', 'median_error', 'max_error', 'r_squared']:
        assert key in metrics
        assert isinstance(metrics[key], float)
        assert np.isfinite(metrics[key])
    assert metrics['mae'] == pytest.approx(1.0)
    assert metrics['rmse'] == pytest.approx(1.0)
    assert metrics['mse'] == pytest.approx(1.0)
    assert metrics['max_error'] == pytest.approx(1.0)

@given(st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                min_size=1, max_size=20))
def test_compute_error_metrics_length_mismatch_raises(values):
    real = np.array(values)
    pred = real[:-1] if len(real) > 1 else np.array([])
    with pytest.raises(ValueError):
        compute_error_metrics(real, pred)
