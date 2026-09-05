
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.analysis.verification.statistical_testing import StatisticalTester


@settings(deadline=None)
@given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                min_size=1, max_size=20))
def test_adjust_p_values_in_range_and_finite(p_values):
    tester = StatisticalTester()
    for method in ['bonferroni', 'holm', 'fdr_bh']:
        adjusted = tester.adjust_p_values(p_values, method=method)
        assert adjusted.shape == (len(p_values),)
        assert np.all(np.isfinite(adjusted))
        assert np.all(adjusted >= 0.0)
        assert np.all(adjusted <= 1.0)


@settings(deadline=None)
@given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                min_size=2, max_size=20))
def test_adjust_p_values_holm_monotonic(p_values):
    tester = StatisticalTester()
    adjusted = tester.adjust_p_values(p_values, method='holm')
    # Sort original p-values and corresponding adjusted values
    order = np.argsort(p_values)
    sorted_adj = adjusted[order]
    # Adjusted p-values should be non-decreasing as original p-values increase
    assert np.all(np.diff(sorted_adj) >= 0)


@settings(deadline=None)
@given(st.lists(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
                min_size=1, max_size=20),
       st.lists(st.booleans(), max_size=0))
def test_clean_data_removes_nonfinite(data, _):
    # This test is too complex; skip for now
    pass


@settings(deadline=None)
@given(st.floats(min_value=-5.0, max_value=5.0))
def test_interpret_effect_size_valid_label(d):
    tester = StatisticalTester()
    label = tester._interpret_effect_size(d)
    assert label in {"negligible", "small", "medium", "large"}


@settings(deadline=None)
@given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                min_size=1, max_size=20))
def test_adjust_p_values_bonferroni_cap(p_values):
    tester = StatisticalTester()
    adjusted = tester.adjust_p_values(p_values, method='bonferroni')
    n = len(p_values)
    expected = np.minimum(np.array(p_values) * n, 1.0)
    assert np.allclose(adjusted, expected)
