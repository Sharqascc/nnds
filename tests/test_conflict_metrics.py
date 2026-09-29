import pytest

from src.analysis.ssm_error import (
    SsmEvent,
    conflict_counts,
    conflict_prf,
    critical_conflict_recall,
)


def _e(a, b, pet=None, ttc=None):
    return SsmEvent(a, b, pet=pet, ttc=ttc)


def test_conflict_counts_perfect():
    events = [_e(1, 2, pet=1.0), _e(3, 4, pet=2.0)]
    c = conflict_counts(events, events)
    assert c == {"tp": 2, "fp": 0, "fn": 0}


def test_conflict_counts_mixed():
    pred = [_e(1, 2, pet=1.0), _e(3, 4, pet=2.0)]
    gt = [_e(1, 2, pet=1.0), _e(5, 6, pet=3.0)]
    c = conflict_counts(pred, gt)
    assert c == {"tp": 1, "fp": 1, "fn": 1}


def test_conflict_prf_perfect():
    events = [_e(1, 2, pet=1.0)]
    m = conflict_prf(events, events)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_conflict_prf_all_fp():
    pred = [_e(1, 2, pet=1.0)]
    gt = [_e(3, 4, pet=1.0)]
    m = conflict_prf(pred, gt)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_conflict_prf_half_precision():
    pred = [_e(1, 2, pet=1.0), _e(3, 4, pet=1.0)]
    gt = [_e(1, 2, pet=1.0)]
    m = conflict_prf(pred, gt)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(2 * 0.5 * 1.0 / 1.5)


def test_conflict_prf_empty_both():
    m = conflict_prf([], [])
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_unordered_pair_counts_as_match():
    pred = [_e(2, 1, pet=1.0)]
    gt = [_e(1, 2, pet=1.0)]
    c = conflict_counts(pred, gt)
    assert c == {"tp": 1, "fp": 0, "fn": 0}


def test_critical_conflict_recall_all_hit():
    gt = [_e(1, 2, pet=0.5), _e(3, 4, pet=1.0)]
    pred = [_e(1, 2, pet=0.5), _e(3, 4, pet=1.0)]
    m = critical_conflict_recall(pred, gt, threshold=1.5)
    assert m["recall"] == 1.0
    assert m["n_critical_gt"] == 2


def test_critical_conflict_recall_partial():
    gt = [_e(1, 2, pet=0.5), _e(3, 4, pet=1.0)]
    pred = [_e(1, 2, pet=0.5)]
    m = critical_conflict_recall(pred, gt, threshold=1.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["n_hit"] == 1


def test_critical_conflict_recall_no_critical_gt():
    gt = [_e(1, 2, pet=3.0)]
    pred = [_e(1, 2, pet=3.0)]
    m = critical_conflict_recall(pred, gt, threshold=1.5)
    assert m["recall"] == 0.0
    assert m["n_critical_gt"] == 0


def test_critical_conflict_recall_uses_ttc_field():
    gt = [_e(1, 2, ttc=0.4), _e(3, 4, ttc=2.0)]
    pred = [_e(1, 2, ttc=0.4)]
    m = critical_conflict_recall(pred, gt, field="ttc", threshold=1.5)
    assert m["recall"] == 1.0
    assert m["n_critical_gt"] == 1
