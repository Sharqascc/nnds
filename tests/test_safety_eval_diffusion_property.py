
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.analysis.safety_eval_diffusion import compute_ttc_seq, first_below_threshold


@settings(deadline=None)
@given(st.lists(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False), min_size=0, max_size=50),
       st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False))
def test_first_below_threshold_returns_valid_index(dist_seq, thresh):
    seq = np.array(dist_seq)
    idx = first_below_threshold(seq, thresh)
    if idx is not None:
        assert 0 <= idx < len(seq)
        assert seq[int(idx)] < thresh
        assert not any(seq[:int(idx)] < thresh)
    else:
        assert not any(seq < thresh)

@settings(deadline=None)
@given(st.integers(min_value=2, max_value=20).flatmap(
    lambda n: st.tuples(
        st.lists(st.tuples(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                           st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)), min_size=n, max_size=n),
        st.lists(st.tuples(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                           st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)), min_size=n, max_size=n)
    )),
    st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False))
def test_compute_ttc_seq_length_and_values(samples, dt):
    pos1_list, _ = samples
    pos1 = np.array(pos1_list, dtype=float)
    pos2 = np.array(pos2_list, dtype=float)
    ttc = compute_ttc_seq(pos1, pos2, dt)
    assert len(ttc) == len(pos1) - 1
    for val in ttc:
        if val is not None:
            assert isinstance(val, float)
            assert val >= 0.0
        else:
            assert val is None

@settings(deadline=None)
@given(st.integers(min_value=2, max_value=20).flatmap(
    lambda n: st.tuples(
        st.lists(st.tuples(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                           st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)), min_size=n, max_size=n),
        st.lists(st.tuples(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                           st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)), min_size=n, max_size=n)
    )),
    st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False))
def test_compute_ttc_seq_zero_velocity_returns_none(samples, dt):
    pos1_list, _ = samples
    pos1 = np.array(pos1_list, dtype=float)
    pos2 = pos1.copy()
    ttc = compute_ttc_seq(pos1, pos2, dt)
    assert all(v is None for v in ttc)
