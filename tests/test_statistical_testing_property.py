import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.analysis.verification import statistical_testing as st_mod


@settings(deadline=None)
@given(
    st.lists(
        st.floats(min_value=0.1, max_value=9.9, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=20,
        unique=True,
    ),
    st.lists(
        st.floats(min_value=0.1, max_value=9.9, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=20,
        unique=True,
    ),
)
def test_pet_difference_result_valid(a, b):
    result = st_mod.test_pet_difference(np.array(a), np.array(b), parametric=True)
    assert isinstance(result, dict)
    assert "test_statistics" in result
    assert "p_value" in result["test_statistics"]


@settings(deadline=None)
@given(
    st.lists(
        st.floats(min_value=0.1, max_value=9.9, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=20,
        unique=True,
    ),
    st.lists(
        st.floats(min_value=0.1, max_value=9.9, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=20,
        unique=True,
    ),
)
def test_ttc_difference_result_valid(a, b):
    result = st_mod.test_ttc_difference(np.array(a), np.array(b), parametric=True)
    assert isinstance(result, dict)
    assert "test_statistics" in result
    assert "p_value" in result["test_statistics"]


@settings(deadline=None)
@given(
    st.lists(
        st.floats(min_value=0.1, max_value=9.9, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=20,
        unique=True,
    ),
    st.lists(
        st.floats(min_value=0.1, max_value=9.9, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=20,
        unique=True,
    ),
)
def test_paired_test_result_valid(a, b):
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    result = st_mod.paired_test(np.array(a), np.array(b), parametric=True)
    assert isinstance(result, dict)
    assert "test_statistics" in result
    assert "p_value" in result["test_statistics"]


@settings(deadline=None)
@given(st.lists(st.integers(min_value=1, max_value=50), min_size=2, max_size=10))
def test_chi_square_test_result_valid(observed):
    result = st_mod.chi_square_test(np.array(observed, dtype=float))
    assert isinstance(result, dict)
    assert "test_statistics" in result or "p_value" in result
