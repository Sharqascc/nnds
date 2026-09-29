import numpy as np
import pytest

from src.analysis.ssm_agreement import (
    agreement_metrics,
    aligned_pairs,
    pet_vs_pet,
    ttc_vs_ttc,
)
from src.analysis.ssm_error import SsmEvent


def _e(a, b, pet=None, ttc=None):
    return SsmEvent(a, b, pet=pet, ttc=ttc)


def test_perfect_agreement():
    events = [_e(1, 2, pet=1.5), _e(3, 4, pet=2.0), _e(5, 6, pet=0.8)]
    m = pet_vs_pet(events, events)
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["r2"] == pytest.approx(1.0)
    assert m["n"] == 3


def test_known_linear_offset_r2_less_than_one():
    gt = [_e(1, 2, pet=1.0), _e(3, 4, pet=2.0), _e(5, 6, pet=3.0)]
    pred = [_e(1, 2, pet=1.2), _e(3, 4, pet=2.2), _e(5, 6, pet=3.2)]
    m = pet_vs_pet(pred, gt)
    assert m["mae"] == pytest.approx(0.2)
    assert m["rmse"] == pytest.approx(0.2)
    # R^2 = 1 - sum(err^2) / sum((gt - mean)^2) = 1 - 3*0.04 / 2 = 0.94
    assert m["r2"] == pytest.approx(0.94)
    assert m["spearman"] == pytest.approx(1.0)


def test_anti_correlated_gives_negative_spearman():
    gt = [_e(1, 2, pet=1.0), _e(3, 4, pet=2.0), _e(5, 6, pet=3.0)]
    pred = [_e(1, 2, pet=3.0), _e(3, 4, pet=2.0), _e(5, 6, pet=1.0)]
    m = pet_vs_pet(pred, gt)
    assert m["spearman"] < 0.0


def test_missing_matched_pairs_dropped():
    gt = [_e(1, 2, pet=1.0), _e(3, 4, pet=2.0)]
    pred = [_e(1, 2, pet=1.0)]  # only one common pair
    m = pet_vs_pet(pred, gt)
    assert m["n"] == 1


def test_empty_returns_zero():
    m = pet_vs_pet([], [])
    assert m == {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "spearman": 0.0, "n": 0}


def test_aligned_pairs_order_matches():
    gt = [_e(1, 2, pet=1.0), _e(3, 4, pet=2.0)]
    pred = [_e(3, 4, pet=2.5), _e(1, 2, pet=1.5)]
    pv, gv = aligned_pairs(pred, gt, "pet")
    # sorted by pair key: (1,2) then (3,4)
    np.testing.assert_allclose(pv, [1.5, 2.5])
    np.testing.assert_allclose(gv, [1.0, 2.0])


def test_ttc_field_used():
    gt = [_e(1, 2, ttc=0.5)]
    pred = [_e(1, 2, ttc=0.7)]
    m = ttc_vs_ttc(pred, gt)
    assert m["mae"] == pytest.approx(0.2)


def test_single_matched_pair_spearman_zero():
    gt = [_e(1, 2, pet=1.0)]
    pred = [_e(1, 2, pet=1.2)]
    m = pet_vs_pet(pred, gt)
    assert m["spearman"] == 0.0  # not enough data for rank correlation
    assert m["n"] == 1
