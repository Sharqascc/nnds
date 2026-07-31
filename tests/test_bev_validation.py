import numpy as np

from core.validation import compute_error_metrics


def test_bev_error_metrics():
    m = compute_error_metrics([0.1, 0.2, 0.3])
    assert m.num_samples == 3
    assert np.isclose(m.mean_error, 0.2)
    assert np.isclose(m.max_error, 0.3)
