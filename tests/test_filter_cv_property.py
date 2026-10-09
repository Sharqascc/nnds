"""Property tests for paper.analysis.filter_cv."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from paper.analysis import filter_cv as fc


def test_min_concurrent_dist_identical_tracks_is_zero():
    ta = {0: (10.0, 20.0), 1: (11.0, 21.0), 2: (12.0, 22.0)}
    assert fc._min_concurrent_dist(ta, ta) == pytest.approx(0.0)


def test_min_concurrent_dist_disjoint_frames_is_inf():
    ta = {0: (0.0, 0.0), 1: (1.0, 1.0)}
    tb = {10: (0.0, 0.0), 11: (1.0, 1.0)}
    assert fc._min_concurrent_dist(ta, tb) == float("inf")


def test_metrics_perfect_predictions():
    y = np.array([1, 0, 1, 0])
    m = fc._metrics(y, y)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["tp"] == 2
    assert m["tn"] == 2
    assert m["fp"] == 0
    assert m["fn"] == 0


@given(
    dist=st.floats(0.0, 100.0, allow_nan=False, allow_infinity=False),
    pet=st.floats(0.0, 5.0, allow_nan=False, allow_infinity=False),
)
def test_apply_rule_binary_output(dist, pet):
    df = pd.DataFrame(
        {
            "min_concurrent_dist_m": [dist, dist],
            "pet_s": [pet, pet],
        }
    )
    out = fc._apply_rule(df)
    assert out.shape == (2,)
    assert set(out.tolist()).issubset({0, 1})
    expected = 1 if (dist <= fc.MAX_CONCURRENT_DIST_M and pet <= fc.MAX_PET_S) else 0
    assert (out == expected).all()


@given(
    values=st.lists(
        st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=20,
    )
)
def test_bootstrap_ci_is_bounded_by_min_and_max(values):
    rng = np.random.default_rng(0)
    ci = fc._bootstrap_ci(values, 200, rng)
    lo, hi = ci["lo"], ci["hi"]
    assert lo <= hi
    assert lo <= ci["mean"] <= hi
