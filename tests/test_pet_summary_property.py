
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.analysis.pet_summary import PETEventAnalyzer


def make_analyzer(pet_values):
    tmpdir = tempfile.TemporaryDirectory()
    csv_path = Path(tmpdir.name) / "test.csv"
    pd.DataFrame({"pet": pet_values}).to_csv(csv_path, index=False)
    return PETEventAnalyzer(csv_path), tmpdir

@given(st.lists(st.floats(min_value=0.1, max_value=9.9), min_size=2, max_size=30))
def test_basic_stats_count_matches_rows(pet_values):
    analyzer, tmpdir = make_analyzer(pet_values)
    try:
        stats = analyzer.basic_stats(ci=0.95)
        assert stats["count"] == len(pet_values)
    finally:
        tmpdir.cleanup()

@given(st.lists(st.floats(min_value=0.1, max_value=9.9), min_size=2, max_size=30))
def test_basic_stats_ci_bounds(pet_values):
    analyzer, tmpdir = make_analyzer(pet_values)
    try:
        stats = analyzer.basic_stats(ci=0.95)
        assert stats["ci_mean_lower"] <= stats["mean"] <= stats["ci_mean_upper"]
        assert stats["ci_mean_lower"] <= stats["ci_mean_upper"]
    finally:
        tmpdir.cleanup()

@given(st.integers(min_value=2, max_value=20).flatmap(
    lambda n: st.tuples(
        st.lists(st.floats(min_value=0.1, max_value=9.9), min_size=n, max_size=n),
        st.lists(st.floats(min_value=0.1, max_value=9.9), min_size=n, max_size=n)
    )
))
def test_cohens_d_non_negative(samples):
    a, b = samples
    d = PETEventAnalyzer._cohens_d(np.array(a), np.array(b))
    assert d >= 0

@given(st.lists(st.floats(min_value=0.1, max_value=9.9), min_size=1, max_size=20),
       st.lists(st.floats(min_value=0.1, max_value=9.9), min_size=1, max_size=20))
def test_cliffs_delta_in_range(a, b):
    delta = PETEventAnalyzer._cliffs_delta(np.array(a), np.array(b))
    assert -1.0 <= delta <= 1.0

@given(st.floats(min_value=-10, max_value=10))
def test_interpret_effect_size_valid(d):
    label = PETEventAnalyzer._interpret_effect_size(d)
    assert label in {"negligible", "small", "medium", "large"}
