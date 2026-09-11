import pytest

from src.analysis.tracking_metrics import (
    Track,
    count_id_switches,
    hota,
    idf1,
    mota,
)


def _t(frame, tid, box):
    return Track(frame, tid, box)


def _perfect():
    gt = [_t(0, 1, (0, 0, 10, 10)), _t(1, 1, (1, 1, 11, 11)), _t(2, 1, (2, 2, 12, 12))]
    trk = [_t(0, 1, (0, 0, 10, 10)), _t(1, 1, (1, 1, 11, 11)), _t(2, 1, (2, 2, 12, 12))]
    return trk, gt


def test_perfect_mota():
    trk, gt = _perfect()
    assert mota(trk, gt) == pytest.approx(1.0)


def test_perfect_idf1():
    trk, gt = _perfect()
    assert idf1(trk, gt) == pytest.approx(1.0)


def test_perfect_hota():
    trk, gt = _perfect()
    assert hota(trk, gt) == pytest.approx(1.0)


def test_empty_preds_mota_zero():
    _, gt = _perfect()
    assert mota([], gt) == pytest.approx(0.0)


def test_empty_preds_idf1_zero():
    _, gt = _perfect()
    assert idf1([], gt) == pytest.approx(0.0)


def test_empty_preds_hota_zero():
    _, gt = _perfect()
    assert hota([], gt) == pytest.approx(0.0)


def test_both_empty_mota_one():
    assert mota([], []) == 1.0


def test_both_empty_idf1_one():
    assert idf1([], []) == 1.0


def test_id_switch_counted():
    gt = [_t(0, 1, (0, 0, 10, 10)), _t(1, 1, (1, 1, 11, 11))]
    trk = [_t(0, 1, (0, 0, 10, 10)), _t(1, 2, (1, 1, 11, 11))]  # ID changes 1 -> 2
    assert count_id_switches(trk, gt) == 1
    assert mota(trk, gt) < 1.0


def test_false_positive_reduces_mota():
    trk, gt = _perfect()
    trk = [*trk, _t(0, 99, (500, 500, 510, 510))]  # extra false detection
    assert mota(trk, gt) < 1.0


def test_false_negative_reduces_mota():
    _, gt = _perfect()
    trk = _perfect()[0][:-1]  # drop last frame
    assert mota(trk, gt) < 1.0


def test_idf1_bounded():
    trk, gt = _perfect()
    trk = [*trk, _t(5, 99, (0, 0, 1, 1))]
    v = idf1(trk, gt)
    assert 0.0 <= v <= 1.0


def test_hota_bounded():
    trk, gt = _perfect()
    v = hota(trk, gt)
    assert 0.0 <= v <= 1.0
