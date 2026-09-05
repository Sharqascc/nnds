
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.core.validation import (
    ValidationMetrics,
    compute_error_metrics,
    validate_numeric_array,
)


@given(st.lists(st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False, allow_subnormal=False),
                min_size=1, max_size=100))
def test_compute_error_metrics_invariants(errors):
    metrics = compute_error_metrics(errors)
    arr = np.asarray(errors, dtype=float)
    assert isinstance(metrics, ValidationMetrics)
    assert metrics.num_samples == len(arr)
    assert metrics.max_error == pytest.approx(np.max(arr))
    assert metrics.mean_error == pytest.approx(np.mean(arr))
    assert metrics.rmse == pytest.approx(np.sqrt(np.mean(arr**2)))
    assert metrics.rmse >= 0.0
    assert metrics.max_error >= metrics.mean_error - 1e-9


@given(st.lists(st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False, allow_subnormal=False),
                min_size=1, max_size=100))
def test_compute_error_metrics_rmse_bounds(errors):
    metrics = compute_error_metrics(errors)
    arr = np.asarray(errors, dtype=float)
    max_abs = np.max(np.abs(arr))
    # RMS is always between |mean| and max_abs (inclusive)
    assert metrics.rmse >= abs(metrics.mean_error) - 1e-9
    assert metrics.rmse <= max_abs + 1e-9


@given(st.lists(st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False, allow_subnormal=False),
                min_size=1, max_size=50))
def test_validate_numeric_array_valid(data):
    arr = validate_numeric_array("test", data, ndim=1)
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 1
    assert arr.size == len(data)
    assert np.all(np.isfinite(arr))


@given(st.lists(st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False, allow_subnormal=False),
                min_size=0, max_size=0))
def test_validate_numeric_array_empty_raises(data):
    with pytest.raises(ValueError):
        validate_numeric_array("test", data, ndim=1)


@given(st.lists(st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False, allow_subnormal=False),
                min_size=1, max_size=50))
def test_validate_numeric_array_nonfinite_raises(data):
    # Force at least one non-finite value
    data[0] = float('nan')
    with pytest.raises(ValueError):
        validate_numeric_array("test", data, ndim=1)
