import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.analysis.ssm_error import SsmEvent, pair_key, pet_value_metrics, ttc_value_metrics

event_st = st.builds(
    SsmEvent,
    track_a=st.integers(0, 5),
    track_b=st.integers(0, 5),
    pet=st.floats(0.1, 5.0, allow_nan=False, allow_infinity=False),
    ttc=st.floats(0.1, 5.0, allow_nan=False, allow_infinity=False),
)
events_st = st.lists(event_st, min_size=0, max_size=15)


@given(events_st)
@settings(max_examples=50)
def test_pet_mae_non_negative(events):
    m = pet_value_metrics(events, events)
    assert m["mae"] >= 0.0
    assert m["rmse"] >= 0.0


@given(events_st)
@settings(max_examples=50)
def test_identical_events_zero_error(events):
    m = pet_value_metrics(events, events)
    assert m["mae"] == pytest.approx(0.0, abs=1e-12)
    assert m["n_pred_only"] == 0
    assert m["n_gt_only"] == 0


@given(events_st, events_st)
@settings(max_examples=50)
def test_mae_symmetric(pred, gt):
    m1 = pet_value_metrics(pred, gt)
    m2 = pet_value_metrics(gt, pred)
    assert m1["mae"] == pytest.approx(m2["mae"], abs=1e-9)
    assert m1["n_matched"] == m2["n_matched"]
    assert m1["n_pred_only"] == m2["n_gt_only"]
    assert m1["n_gt_only"] == m2["n_pred_only"]


@given(events_st, events_st)
@settings(max_examples=50)
def test_rmse_geq_mae(pred, gt):
    m = pet_value_metrics(pred, gt)
    assert m["rmse"] + 1e-9 >= m["mae"]


@given(events_st, events_st)
@settings(max_examples=50)
def test_ttc_mae_non_negative(pred, gt):
    m = ttc_value_metrics(pred, gt)
    assert m["mae"] >= 0.0
    assert m["rmse"] >= 0.0


@given(events_st)
@settings(max_examples=50)
def test_pair_key_idempotent(events):
    for e in events:
        k1 = pair_key(e.track_a, e.track_b)
        k2 = pair_key(e.track_b, e.track_a)
        assert k1 == k2
