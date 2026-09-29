import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.analysis.verification.statistical_testing import StatisticalTester


def non_negative_floats(min_size=0, max_size=50):
    return st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=min_size,
        max_size=max_size,
    )


@given(pvals=non_negative_floats())
def test_bonferroni_adjusted_pvalues_within_bounds(pvals):
    tester = StatisticalTester()
    adjusted = tester.adjust_p_values(np.array(pvals), method="bonferroni")
    if len(pvals) == 0:
        assert adjusted.size == 0
        return
    assert np.all(np.isfinite(adjusted))
    assert np.all(adjusted >= 0.0)
    assert np.all(adjusted <= 1.0)


@given(pvals=non_negative_floats(min_size=1))
def test_holm_monotonicity_and_bounds(pvals):
    tester = StatisticalTester()
    sorted_pvals = np.sort(pvals)
    adjusted = tester.adjust_p_values(sorted_pvals, method="holm")
    assert np.all(np.isfinite(adjusted))
    assert np.all(adjusted >= 0.0)
    assert np.all(adjusted <= 1.0)
    assert np.all(np.diff(adjusted) >= -1e-12)


@given(pvals=non_negative_floats(min_size=1))
def test_fdr_bh_monotonicity_and_bounds(pvals):
    tester = StatisticalTester()
    sorted_pvals = np.sort(pvals)
    adjusted = tester.adjust_p_values(sorted_pvals, method="fdr_bh")
    assert np.all(np.isfinite(adjusted))
    assert np.all(adjusted >= 0.0)
    assert np.all(adjusted <= 1.0)
    assert np.all(np.diff(adjusted) >= -1e-12)


@pytest.mark.parametrize("method", ["bonferroni", "holm", "fdr_bh"])
def test_empty_input_returns_empty_array(method):
    tester = StatisticalTester()
    result = tester.adjust_p_values(np.array([]), method=method)
    assert isinstance(result, np.ndarray)
    assert result.size == 0


@pytest.mark.parametrize("method", ["bonferroni", "holm", "fdr_bh"])
def test_identical_pvalues_return_valid_adjusted(method):
    tester = StatisticalTester()
    identical = np.full(10, 0.42)
    adjusted = tester.adjust_p_values(identical, method=method)
    assert adjusted.shape == identical.shape
    assert np.all(np.isfinite(adjusted))
    assert np.all(adjusted >= 0.0)
    assert np.all(adjusted <= 1.0)
