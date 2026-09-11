import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.analysis.ssm_agreement import agreement_metrics
from src.analysis.ssm_error import SsmEvent

event_st = st.builds(
    SsmEvent,
    track_a=st.integers(0, 5),
    track_b=st.integers(0, 5),
    pet=st.floats(0.05, 5.0, allow_nan=False, allow_infinity=False),
)
events_st = st.lists(event_st, min_size=0, max_size=12)


@given(events_st)
@settings(max_examples=50)
def test_identical_r2_is_one(events):
    m = agreement_metrics(events, events, "pet")
    if m["n"] >= 2:
        assert m["r2"] == pytest.approx(1.0, abs=1e-9)


@given(events_st)
@settings(max_examples=50)
def test_identical_spearman_is_one(events):
    m = agreement_metrics(events, events, "pet")
    if m["n"] >= 3:
        # need at least two distinct values for a defined rank correlation
        vals = {e.pet for e in events}
        if len(vals) >= 2:
            assert m["spearman"] == pytest.approx(1.0, abs=1e-9)


@given(events_st, events_st)
@settings(max_examples=50)
def test_mae_non_negative(pred, gt):
    m = agreement_metrics(pred, gt, "pet")
    assert m["mae"] >= 0.0
    assert m["rmse"] >= 0.0


@given(events_st, events_st)
@settings(max_examples=50)
def test_r2_bounded_above_by_one(pred, gt):
    m = agreement_metrics(pred, gt, "pet")
    assert m["r2"] <= 1.0 + 1e-9


@given(events_st, events_st)
@settings(max_examples=50)
def test_spearman_bounded(pred, gt):
    m = agreement_metrics(pred, gt, "pet")
    assert -1.0 - 1e-9 <= m["spearman"] <= 1.0 + 1e-9


@given(events_st, events_st)
@settings(max_examples=50)
def test_rmse_geq_mae(pred, gt):
    m = agreement_metrics(pred, gt, "pet")
    assert m["rmse"] + 1e-9 >= m["mae"]
