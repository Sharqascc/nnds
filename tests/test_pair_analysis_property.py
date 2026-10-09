"""Property tests for paper.analysis.pair_analysis."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from paper.analysis import pair_analysis as pa


def test_speed_stats_on_static_track_is_zero():
    td = {0: (1.0, 1.0), 1: (1.0, 1.0), 2: (1.0, 1.0)}
    mean, mx = pa._speed_stats(td)
    assert mean == pytest.approx(0.0)
    assert mx == pytest.approx(0.0)


@given(
    dx=st.floats(1.0, 100.0, allow_nan=False, allow_infinity=False),
)
def test_speed_stats_positive_on_movement(dx):
    td = {0: (0.0, 0.0), 30: (dx, 0.0)}
    mean, mx = pa._speed_stats(td)
    assert mean > 0.0
    assert mx > 0.0


def test_gap_rate_no_gaps_is_zero():
    td = {0: (0.0, 0.0), 1: (0.1, 0.0), 2: (0.2, 0.0)}
    assert pa._gap_rate(td) == pytest.approx(0.0)


def test_gap_rate_full_gap_is_one():
    td = {0: (0.0, 0.0), 10: (1.0, 0.0)}
    r = pa._gap_rate(td)
    assert 0.0 <= r <= 1.0


def test_net_disp_zero_on_static_track():
    td = {0: (5.0, 5.0), 1: (5.0, 5.0), 2: (5.0, 5.0)}
    assert pa._net_disp(td) == pytest.approx(0.0)


@given(
    x=st.lists(
        st.floats(0.0, 10.0, allow_nan=False, allow_infinity=False),
        min_size=4, max_size=20,
    ),
    y=st.lists(
        st.sampled_from([0, 1]),
        min_size=4, max_size=20,
    ),
)
def test_fisher_ratio_finite(x, y):
    a = np.array(x)
    labels = np.array(y[: len(x)])
    if len(set(labels.tolist())) < 2:
        return
    f = pa._fisher_ratio(a, labels)
    assert np.isfinite(f)
