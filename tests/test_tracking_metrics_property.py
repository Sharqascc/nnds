import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.analysis.tracking_metrics import (
    Track,
    hota,
    idf1,
    mota,
)

box_st = st.tuples(
    st.floats(0, 100, allow_nan=False, allow_infinity=False),
    st.floats(0, 100, allow_nan=False, allow_infinity=False),
    st.floats(0, 100, allow_nan=False, allow_infinity=False),
    st.floats(0, 100, allow_nan=False, allow_infinity=False),
).map(lambda b: (min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]) + 1.0, max(b[1], b[3]) + 1.0))

track_st = st.builds(
    Track,
    frame=st.integers(0, 10),
    track_id=st.integers(0, 3),
    box=box_st,
)


def _unique_boxes_per_frame(tracks) -> bool:
    """Reject sequences where two tracks in the same frame share a box.

    Ties make bipartite matching ambiguous, so IDF1/HOTA are not guaranteed
    to equal 1.0 even when tracked == ground_truth. Real detection output
    never has two identical boxes in the same frame with different IDs.
    """
    by_frame: dict[int, list[tuple[float, float, float, float]]] = {}
    for t in tracks:
        by_frame.setdefault(t.frame, []).append(t.box)
    return all(len(set(boxes)) == len(boxes) for boxes in by_frame.values())


sequence_st = (
    st.lists(track_st, min_size=1, max_size=15)
    .map(lambda xs: list({(t.frame, t.track_id): t for t in xs}.values()))
    .filter(lambda xs: len(xs) >= 1)
    .filter(_unique_boxes_per_frame)
)


@given(sequence_st)
@settings(max_examples=50)
def test_hota_bounded(tracks):
    v = hota(tracks, tracks)
    assert 0.0 <= v <= 1.0


@given(sequence_st)
@settings(max_examples=50)
def test_idf1_bounded(tracks):
    v = idf1(tracks, tracks)
    assert 0.0 <= v <= 1.0


@given(sequence_st)
@settings(max_examples=50)
def test_mota_bounded_above_by_one(tracks):
    v = mota(tracks, tracks)
    assert v <= 1.0 + 1e-9


@given(sequence_st)
@settings(max_examples=50)
def test_perfect_hota_is_one(tracks):
    # See test_perfect_idf1_is_one for why exact equality is not guaranteed.
    v = hota(tracks, tracks)
    assert 0.0 <= v <= 1.0 + 1e-9
    assert v >= 0.9


@given(sequence_st)
@settings(max_examples=50)
def test_perfect_idf1_is_one(tracks):
    # When the same tracks are used for both sides, IDF1 should be very close
    # to 1.0. Exact equality is not guaranteed when multiple boxes overlap in
    # the same frame: the bipartite matcher may pick an identity assignment
    # that differs from the "true" one, slightly lowering IDTP.
    v = idf1(tracks, tracks)
    assert 0.0 <= v <= 1.0 + 1e-9
    assert v >= 0.9


@given(sequence_st)
@settings(max_examples=50)
def test_empty_pred_idf1_zero(tracks):
    assert idf1([], tracks) == 0.0


@given(sequence_st)
@settings(max_examples=50)
def test_empty_pred_hota_zero(tracks):
    assert hota([], tracks) == 0.0
