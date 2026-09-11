import pytest

from src.analysis.ssm_error import (
    SsmEvent,
    pair_key,
    pet_value_metrics,
    ssm_value_metrics,
    ttc_value_metrics,
)


def _e(a, b, pet=None, ttc=None):
    return SsmEvent(a, b, pet=pet, ttc=ttc)


def test_pair_key_symmetric():
    assert pair_key(1, 2) == pair_key(2, 1)
    assert pair_key(1, 2) == (1, 2)


def test_pair_key_same_id():
    assert pair_key(5, 5) == (5, 5)


def test_pet_perfect_match_zero_error():
    events = [_e(1, 2, pet=1.5), _e(3, 4, pet=2.0)]
    m = pet_value_metrics(events, events)
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["n_matched"] == 2


def test_pet_known_error():
    pred = [_e(1, 2, pet=1.7)]
    gt = [_e(1, 2, pet=1.5)]
    m = pet_value_metrics(pred, gt)
    assert m["mae"] == pytest.approx(0.2)
    assert m["rmse"] == pytest.approx(0.2)


def test_pet_unordered_pair_matches():
    pred = [_e(2, 1, pet=1.5)]
    gt = [_e(1, 2, pet=1.5)]
    m = pet_value_metrics(pred, gt)
    assert m["n_matched"] == 1
    assert m["mae"] == 0.0


def test_pet_missing_prediction_counted_fn():
    gt = [_e(1, 2, pet=1.5), _e(3, 4, pet=2.0)]
    pred = [_e(1, 2, pet=1.5)]
    m = pet_value_metrics(pred, gt)
    assert m["n_matched"] == 1
    assert m["n_gt_only"] == 1
    assert m["n_pred_only"] == 0


def test_pet_missing_gt_counted_fp():
    gt = [_e(1, 2, pet=1.5)]
    pred = [_e(1, 2, pet=1.5), _e(3, 4, pet=2.0)]
    m = pet_value_metrics(pred, gt)
    assert m["n_matched"] == 1
    assert m["n_pred_only"] == 1
    assert m["n_gt_only"] == 0


def test_pet_skips_none_values():
    pred = [_e(1, 2, pet=None)]
    gt = [_e(1, 2, pet=1.5)]
    m = pet_value_metrics(pred, gt)
    assert m["n_matched"] == 0
    assert m["n_gt_only"] == 1


def test_pet_dedup_uses_minimum():
    pred = [_e(1, 2, pet=2.0), _e(1, 2, pet=1.5), _e(1, 2, pet=3.0)]
    gt = [_e(1, 2, pet=1.5)]
    m = pet_value_metrics(pred, gt)
    assert m["n_matched"] == 1
    assert m["mae"] == pytest.approx(0.0)


def test_pet_empty_both():
    m = pet_value_metrics([], [])
    assert m == {
        "mae": 0.0,
        "rmse": 0.0,
        "n_matched": 0,
        "n_pred_only": 0,
        "n_gt_only": 0,
    }


def test_ttc_known_error():
    pred = [_e(1, 2, ttc=0.5)]
    gt = [_e(1, 2, ttc=0.7)]
    m = ttc_value_metrics(pred, gt)
    assert m["mae"] == pytest.approx(0.2)
    assert m["n_matched"] == 1


def test_ssm_combined_returns_both():
    pred = [_e(1, 2, pet=1.5, ttc=0.6)]
    gt = [_e(1, 2, pet=1.5, ttc=0.6)]
    m = ssm_value_metrics(pred, gt)
    assert m["pet"]["mae"] == 0.0
    assert m["ttc"]["mae"] == 0.0
