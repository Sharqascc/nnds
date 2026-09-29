import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.analysis.ssm_error import (
    SsmEvent,
    conflict_counts,
    conflict_prf,
    critical_conflict_recall,
)

event_st = st.builds(
    SsmEvent,
    track_a=st.integers(0, 5),
    track_b=st.integers(0, 5),
    pet=st.floats(0.05, 5.0, allow_nan=False, allow_infinity=False),
)
events_st = st.lists(event_st, min_size=0, max_size=12)


@given(events_st, events_st)
@settings(max_examples=50)
def test_counts_non_negative(pred, gt):
    c = conflict_counts(pred, gt)
    assert c["tp"] >= 0 and c["fp"] >= 0 and c["fn"] >= 0


@given(events_st, events_st)
@settings(max_examples=50)
def test_prf_bounded(pred, gt):
    m = conflict_prf(pred, gt)
    for k in ("precision", "recall", "f1"):
        assert 0.0 <= m[k] <= 1.0


@given(events_st)
@settings(max_examples=50)
def test_perfect_prf_one(events):
    m = conflict_prf(events, events)
    if events:
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0


@given(events_st, events_st)
@settings(max_examples=50)
def test_symmetric_counts(pred, gt):
    c1 = conflict_counts(pred, gt)
    c2 = conflict_counts(gt, pred)
    assert c1["tp"] == c2["tp"]
    assert c1["fp"] == c2["fn"]
    assert c1["fn"] == c2["fp"]


@given(events_st, events_st, st.floats(0.1, 3.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=50)
def test_critical_recall_bounded(pred, gt, thr):
    m = critical_conflict_recall(pred, gt, threshold=thr)
    assert 0.0 <= m["recall"] <= 1.0
    assert m["n_hit"] <= m["n_critical_gt"]


@given(events_st)
@settings(max_examples=50)
def test_critical_recall_perfect_when_identical(events):
    if not events:
        return
    m = critical_conflict_recall(events, events, threshold=10.0)
    if m["n_critical_gt"] > 0:
        assert m["recall"] == 1.0


@given(events_st, events_st)
@settings(max_examples=50)
def test_f1_symmetric(pred, gt):
    m1 = conflict_prf(pred, gt)
    m2 = conflict_prf(gt, pred)
    assert m1["f1"] == pytest.approx(m2["f1"], abs=1e-9)
