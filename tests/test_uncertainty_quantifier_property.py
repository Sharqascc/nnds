
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.analysis.ssm.uncertainty_quantifier import (
    compute_confidence_interval,
    compute_effect_size,
    compute_sample_size,
)


@given(st.floats(min_value=0.01, max_value=2.0),
       st.floats(min_value=0.5, max_value=0.99),
       st.floats(min_value=0.01, max_value=0.10))
def test_compute_sample_size_valid(effect_size, power, alpha):
    n = compute_sample_size(effect_size, power, alpha)
    assert isinstance(n, int)
    assert n >= 1

@settings(deadline=None)
@given(st.lists(st.floats(min_value=0.1, max_value=9.9), min_size=2, max_size=10, unique=True))
def test_compute_effect_size_self_zero(data):
    arr = np.array(data)
    d = compute_effect_size(arr, arr, estimator="cohens_d")
    assert abs(d) < 1e-9

@given(st.lists(st.floats(min_value=0.1, max_value=9.9), min_size=2, max_size=10, unique=True))
def test_compute_confidence_interval_ordered(data):
    arr = np.array(data)
    lower, upper = compute_confidence_interval(arr, method="parametric")
    assert lower <= upper

@given(st.lists(st.floats(min_value=0.1, max_value=9.9), min_size=2, max_size=10, unique=True))
def test_compute_confidence_interval_contains_mean(data):
    arr = np.array(data)
    lower, upper = compute_confidence_interval(arr, method="parametric")
    assert lower <= np.mean(arr) <= upper
