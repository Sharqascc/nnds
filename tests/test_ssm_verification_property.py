"""Property tests for SSMVerifier.check_data_quality."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.analysis.ssm import ssm_verification as sv


REQUIRED_KEYS = {
    "metric_name", "checks", "warnings", "errors",
    "passed", "clean_data", "summary", "statistics",
}


def test_result_dict_has_expected_keys():
    v = sv.SSMVerifier()
    r = v.check_data_quality(np.arange(20).astype(float))
    assert REQUIRED_KEYS.issubset(r.keys())


def test_empty_array_fails():
    v = sv.SSMVerifier()
    r = v.check_data_quality(np.array([]))
    assert r["passed"] is False
    assert r["clean_data"] is None


def test_all_nan_fails():
    v = sv.SSMVerifier()
    r = v.check_data_quality(np.array([np.nan, np.nan, np.nan]))
    assert r["passed"] is False


@given(
    vals=st.lists(
        st.floats(0.0, 10.0, allow_nan=False, allow_infinity=False),
        min_size=10, max_size=30,
    )
)
def test_clean_input_passes(vals):
    v = sv.SSMVerifier()
    r = v.check_data_quality(np.array(vals), name="test")
    assert r["passed"] is True
    assert r["clean_data"] is not None
    assert len(r["clean_data"]) == len(vals)


@given(
    vals=st.lists(
        st.floats(0.0, 10.0, allow_nan=False, allow_infinity=False),
        min_size=10, max_size=30,
    ),
    n_nan=st.integers(0, 5),
)
def test_nan_values_are_removed(vals, n_nan):
    arr = np.array(vals + [np.nan] * n_nan)
    v = sv.SSMVerifier()
    r = v.check_data_quality(arr)
    if r["clean_data"] is not None:
        assert not np.any(np.isnan(r["clean_data"]))


@given(
    vals=st.lists(
        st.floats(0.0, 10.0, allow_nan=False, allow_infinity=False),
        min_size=10, max_size=30,
    ),
    n_inf=st.integers(0, 3),
)
def test_inf_values_are_removed(vals, n_inf):
    arr = np.array(vals + [np.inf] * n_inf)
    v = sv.SSMVerifier()
    r = v.check_data_quality(arr)
    if r["clean_data"] is not None:
        assert not np.any(np.isinf(r["clean_data"]))


def test_statistics_present_on_valid_data():
    v = sv.SSMVerifier()
    r = v.check_data_quality(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]))
    assert "n" in r["statistics"]
    assert r["statistics"]["n"] == 10
    assert "mean" in r["statistics"]
    assert r["statistics"]["min"] <= r["statistics"]["max"]
