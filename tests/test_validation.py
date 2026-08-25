import numpy as np

from src.core.validation import (
    compute_error_metrics,
    validate_bev_result,
    validate_numeric_array,
)


def test_compute_error_metrics():
    m = compute_error_metrics([1.0, 2.0, 3.0])
    assert m.num_samples == 3
    assert np.isclose(m.mean_error, 2.0)
    assert np.isclose(m.max_error, 3.0)


def test_validate_numeric_array():
    arr = validate_numeric_array("x", [[1, 2], [3, 4]], ndim=2)
    assert arr.shape == (2, 2)


def test_validate_bev_result():
    validate_bev_result(
        {
            "pointerrors": [{"error": 0.1}, {"error": 0.2}],
            "meanerrorall": 0.15,
            "meanerrorinliers": 0.12,
            "stderrorall": 0.05,
            "maxerror": 0.2,
            "rmse": 0.158,
        }
    )
