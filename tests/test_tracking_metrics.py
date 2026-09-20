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


def test_hota_is_not_idf1_on_split_ids():
    """Regression for the HOTA-vs-IDF1 confusion.

    A single GT track over 10 frames, predicted as two IDs of 5 frames each
    with perfect boxes. Correct HOTA:
      DetA = 10/10 = 1.0
      AssA = mean(5/10, 5/10) = 0.5
      HOTA = sqrt(1.0 * 0.5) = 0.7071

    The previous implementation used a global Jaccard on the ID-overlap
    matrix (IDF1-style), which yielded 0.5774. On this case IDF1 correctly
    returns 0.5 and must not equal HOTA.
    """
    import math

    gt = [Track(frame=f, track_id=1, box=(0.0, 0.0, 10.0, 10.0)) for f in range(10)]
    trk = [Track(frame=f, track_id=100, box=(0.0, 0.0, 10.0, 10.0)) for f in range(5)] + [
        Track(frame=f, track_id=200, box=(0.0, 0.0, 10.0, 10.0)) for f in range(5, 10)
    ]

    h = hota(trk, gt)
    i = idf1(trk, gt)

    assert h == pytest.approx(math.sqrt(0.5), abs=1e-3), (
        f"HOTA should be 0.7071 on this case, got {h:.4f}"
    )
    assert i == pytest.approx(0.5, abs=1e-3), f"IDF1 should be 0.5, got {i:.4f}"
    assert abs(h - i) > 0.05, f"HOTA ({h:.4f}) and IDF1 ({i:.4f}) should differ on split IDs"


def test_hota_matches_deta_when_assa_is_perfect():
    """If predictions perfectly match GT (no splits, no FPs), DetA=1 and
    AssA=1, so HOTA=1."""
    gt = [Track(frame=f, track_id=1, box=(0.0, 0.0, 10.0, 10.0)) for f in range(5)]
    trk = [Track(frame=f, track_id=1, box=(0.0, 0.0, 10.0, 10.0)) for f in range(5)]
    assert hota(trk, gt) == pytest.approx(1.0, abs=1e-9)


def test_hota_zero_when_no_true_positives():
    gt = [Track(frame=0, track_id=1, box=(0.0, 0.0, 10.0, 10.0))]
    trk = [Track(frame=0, track_id=1, box=(500.0, 500.0, 510.0, 510.0))]
    assert hota(trk, gt) == pytest.approx(0.0)
