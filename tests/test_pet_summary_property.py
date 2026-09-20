import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.analysis.pet_summary import PETEventAnalyzer


def make_analyzer(pet_values):
    tmpdir = tempfile.TemporaryDirectory()
    csv_path = Path(tmpdir.name) / "test.csv"
    pd.DataFrame({"pet": pet_values}).to_csv(csv_path, index=False)
    return PETEventAnalyzer(csv_path), tmpdir


@given(st.lists(st.floats(min_value=0.1, max_value=9.9), min_size=2, max_size=30))
@pytest.mark.property
def test_basic_stats_count_matches_rows(pet_values):
    analyzer, tmpdir = make_analyzer(pet_values)
    try:
        stats = analyzer.basic_stats(ci=0.95)
        assert stats["count"] == len(pet_values)
    finally:
        tmpdir.cleanup()


@given(st.lists(st.floats(min_value=0.1, max_value=9.9), min_size=2, max_size=30))
@pytest.mark.property
def test_basic_stats_ci_bounds(pet_values):
    analyzer, tmpdir = make_analyzer(pet_values)
    try:
        stats = analyzer.basic_stats(ci=0.95)
        assert stats["ci_mean_lower"] <= stats["mean"] <= stats["ci_mean_upper"]
        assert stats["ci_mean_lower"] <= stats["ci_mean_upper"]
    finally:
        tmpdir.cleanup()


@given(
    st.integers(min_value=2, max_value=20).flatmap(
        lambda n: st.tuples(
            st.lists(st.floats(min_value=0.1, max_value=9.9), min_size=n, max_size=n),
            st.lists(st.floats(min_value=0.1, max_value=9.9), min_size=n, max_size=n),
        )
    )
)
@pytest.mark.property
def test_cohens_d_non_negative(samples):
    a, b = samples
    d = PETEventAnalyzer._cohens_d(np.array(a), np.array(b))
    assert d >= 0


@given(
    st.lists(st.floats(min_value=0.1, max_value=9.9), min_size=1, max_size=20),
    st.lists(st.floats(min_value=0.1, max_value=9.9), min_size=1, max_size=20),
)
@pytest.mark.property
def test_cliffs_delta_in_range(a, b):
    delta = PETEventAnalyzer._cliffs_delta(np.array(a), np.array(b))
    assert -1.0 <= delta <= 1.0


@given(st.floats(min_value=-10, max_value=10))
@pytest.mark.property
def test_interpret_effect_size_valid(d):
    label = PETEventAnalyzer._interpret_effect_size(d)
    assert label in {"negligible", "small", "medium", "large"}


@given(
    st.lists(
        st.floats(-1000.0, 1000.0, allow_nan=False, allow_infinity=False), min_size=2, max_size=20
    ),
    st.lists(
        st.floats(-1000.0, 1000.0, allow_nan=False, allow_infinity=False), min_size=2, max_size=20
    ),
)
@settings(max_examples=50)
def test_cohens_d_properties(sample1, sample2):
    """Cohen's d should be non-negative, finite, and symmetric wrt input order."""
    # Ensure same length
    n = min(len(sample1), len(sample2))
    sample1 = sample1[:n]
    sample2 = sample2[:n]

    d1 = PETEventAnalyzer._cohens_d(np.array(sample1), np.array(sample2))
    d2 = PETEventAnalyzer._cohens_d(np.array(sample2), np.array(sample1))

    assert d1 >= 0
    assert d2 >= 0
    assert np.isclose(d1, d2, atol=1e-9)  # should be symmetric


@given(
    st.lists(
        st.floats(-1000.0, 1000.0, allow_nan=False, allow_infinity=False), min_size=2, max_size=20
    ),
    st.lists(
        st.floats(-1000.0, 1000.0, allow_nan=False, allow_infinity=False), min_size=2, max_size=20
    ),
)
@settings(max_examples=50)
def test_cliffs_delta_properties(sample1, sample2):
    """Cliff's delta should be in [-1, 1] and anti-symmetric under input swap."""
    n1, n2 = len(sample1), len(sample2)
    d1 = PETEventAnalyzer._cliffs_delta(np.array(sample1), np.array(sample2))
    d2 = PETEventAnalyzer._cliffs_delta(np.array(sample2), np.array(sample1))

    assert -1.0 <= d1 <= 1.0
    assert -1.0 <= d2 <= 1.0
    # Anti-symmetry: delta(A,B) = -delta(B,A)
    assert np.isclose(d1, -d2, atol=1e-9)


@given(st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_interpret_effect_size_categories(d):
    """Effect size interpretation should map to known categories."""
    result = PETEventAnalyzer._interpret_effect_size(d)
    assert result in {"negligible", "small", "medium", "large"}

    # Check boundaries correspond to thresholds
    d_abs = abs(d)
    if d_abs < 0.2:
        assert result == "negligible"
    elif d_abs < 0.5:
        assert result == "small"
    elif d_abs < 0.8:
        assert result == "medium"
    else:
        assert result == "large"


@given(
    st.lists(
        st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=30,
    )
)
@pytest.mark.property
def test_basic_stats_quantile_ordering(pet_values):
    """Quantiles must be ordered and IQR must equal q75 - q25.

    Ported from cleanup/system-reorganization (2892564) — that branch
    rewrote the whole file and dropped its @pytest.mark.property markers,
    so only the two genuinely-new assertions were kept.
    """
    analyzer, tmpdir = make_analyzer(pet_values)
    try:
        s = analyzer.basic_stats()
        assert s["count"] == len(pet_values)
        tol = 1e-9
        assert s["min"] - tol <= s["q25"] <= s["median"] <= s["q75"] <= s["max"] + tol
        assert s["min"] - tol <= s["mean"] <= s["max"] + tol
        assert abs(s["iqr"] - (s["q75"] - s["q25"])) < 1e-9
    finally:
        tmpdir.cleanup()


@given(
    st.lists(
        st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=30,
    )
)
@pytest.mark.property
def test_basic_stats_percentiles_monotonic(pet_values):
    """Percentile outputs must be non-decreasing across the p1..p99 range.

    Ported from cleanup/system-reorganization (2892564), same rationale.
    """
    analyzer, tmpdir = make_analyzer(pet_values)
    try:
        s = analyzer.basic_stats()
        pcts = [1, 5, 10, 90, 95, 99]
        values = [s[f"p{p}"] for p in pcts]
        assert values == sorted(values)
    finally:
        tmpdir.cleanup()
