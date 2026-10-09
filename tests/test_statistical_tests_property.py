"""Property tests for paper.analysis.statistical_tests."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from paper.analysis import statistical_tests as stt


@given(
    a=st.lists(
        st.floats(0.0, 5.0, allow_nan=False, allow_infinity=False),
        min_size=5, max_size=30,
    ),
    b=st.lists(
        st.floats(0.0, 5.0, allow_nan=False, allow_infinity=False),
        min_size=5, max_size=30,
    ),
)
def test_mwu_and_ks_shapes(a, b):
    arr_a = np.array(a)
    arr_b = np.array(b)
    # Skip degenerate inputs — scipy returns NaN for zero-variance arrays
    if arr_a.std() == 0.0 or arr_b.std() == 0.0:
        return
    out = stt.mwu_and_ks(arr_a, arr_b)
    assert "pet_mwu" in out
    assert "pet_ks" in out
    assert 0.0 <= out["pet_mwu"]["p"] <= 1.0
    assert 0.0 <= out["pet_ks"]["p"] <= 1.0
    assert 0.0 <= out["pet_ks"]["statistic"] <= 1.0


def test_severity_test_returns_chi2():
    a = np.array([0.5, 1.2, 1.4, 2.5, 0.8])
    b = np.array([0.6, 1.1, 1.9, 2.8, 0.9])
    out = stt.severity_test(a, b)
    assert "severity_chi2" in out
    assert out["severity_chi2"]["chi2"] >= 0.0
    assert 0.0 <= out["severity_chi2"]["p"] <= 1.0


def test_post_hoc_power_is_in_range():
    a = np.array([1.0, 1.2, 1.4, 1.5, 1.6, 1.8])
    b = np.array([1.5, 1.7, 1.9, 2.0, 2.1, 2.3])
    out = stt.post_hoc_power(a, b)
    assert "power" in out
    assert 0.0 <= out["power"]["power_approx"] <= 1.0
    assert np.isfinite(out["power"]["cohens_d"])
