"""Property tests for track splitting under anomalous input.

Covers jump handling (impossible spatial displacement) and lag handling
(out-of-order, duplicate, or gapped frames) in _split_tracks_by_gaps.
"""

from __future__ import annotations

import random

from hypothesis import given, settings
from hypothesis import strategies as st

from src.analysis.grid_trajectory.uvh_coco_fused_grid_pet import (
    TrackPoint,
    _split_tracks_by_gaps,
)

SMALL = st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False)


def _tp(frame: int, x: float, y: float) -> TrackPoint:
    return TrackPoint(frame=frame, x=x, y=y, cls_id=0, cls_name="c", conf=1.0)


def _key(p: TrackPoint) -> tuple:
    return (p.frame, p.x, p.y)


# ---------------------------------------------------------------------------
# Trivial cases
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty():
    assert _split_tracks_by_gaps({}) == {}


def test_empty_track_is_dropped():
    assert _split_tracks_by_gaps({5: []}) == {}


def test_single_point_yields_one_segment():
    out = _split_tracks_by_gaps({5: [_tp(0, 1.0, 2.0)]})
    assert len(out) == 1
    only = next(iter(out.values()))
    assert len(only) == 1
    assert _key(only[0]) == (0, 1.0, 2.0)


# ---------------------------------------------------------------------------
# Preservation invariants
# ---------------------------------------------------------------------------


@given(
    n=st.integers(min_value=1, max_value=50),
    tid=st.integers(min_value=0, max_value=100),
    xs=st.lists(SMALL, min_size=50, max_size=50),
)
@settings(max_examples=40, deadline=None)
def test_all_points_preserved_across_segments(n, tid, xs):
    points = [_tp(i, xs[i], 0.0) for i in range(n)]
    out = _split_tracks_by_gaps({tid: points})
    recovered = sorted(_key(p) for seg in out.values() for p in seg)
    expected = sorted(_key(p) for p in points)
    assert recovered == expected


@given(
    n=st.integers(min_value=2, max_value=30),
    tid=st.integers(min_value=0, max_value=100),
    vx=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    vy=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    x0=SMALL,
    y0=SMALL,
)
@settings(max_examples=40, deadline=None)
def test_clean_sequence_is_not_split(n, tid, vx, vy, x0, y0):
    points = [_tp(i, x0 + vx * i, y0 + vy * i) for i in range(n)]
    out = _split_tracks_by_gaps({tid: points})
    assert len(out) == 1
    only = next(iter(out.values()))
    assert len(only) == n


# ---------------------------------------------------------------------------
# Jump handling
# ---------------------------------------------------------------------------


def test_splits_on_large_frame_gap_with_bad_prediction():
    points = [
        _tp(0, 0.0, 0.0),
        _tp(1, 1.0, 1.0),
        _tp(100, 500.0, 500.0),
    ]
    out = _split_tracks_by_gaps(
        {1: points},
        max_frame_gap=10,
        max_spatial_jump=1e9,
        prediction_tolerance=80.0,
    )
    assert len(out) == 2


def test_splits_on_huge_spatial_jump_no_gap():
    points = [
        _tp(0, 0.0, 0.0),
        _tp(1, 1.0, 1.0),
        _tp(2, 1e6, 1e6),
    ]
    out = _split_tracks_by_gaps(
        {1: points},
        max_frame_gap=10,
        max_spatial_jump=50.0,
        prediction_tolerance=80.0,
    )
    assert len(out) == 2


def test_skips_split_when_prediction_matches():
    points = [
        _tp(0, 0.0, 0.0),
        _tp(1, 1.0, 1.0),
        _tp(20, 20.0, 20.0),
    ]
    out = _split_tracks_by_gaps(
        {1: points},
        max_frame_gap=10,
        max_spatial_jump=1e9,
        prediction_tolerance=80.0,
    )
    assert len(out) == 1


@given(dx=st.floats(min_value=200.0, max_value=1e5, allow_nan=False, allow_infinity=False))
@settings(max_examples=30, deadline=None)
def test_any_impossible_leap_is_split(dx):
    points = [
        _tp(0, 0.0, 0.0),
        _tp(1, 0.0, 0.0),
        _tp(2, dx, 0.0),
    ]
    out = _split_tracks_by_gaps(
        {1: points},
        max_frame_gap=10,
        max_spatial_jump=50.0,
        prediction_tolerance=80.0,
    )
    assert len(out) == 2


# ---------------------------------------------------------------------------
# Lag / ordering
# ---------------------------------------------------------------------------


@given(
    frames=st.lists(st.integers(min_value=0, max_value=1000), min_size=1, max_size=20, unique=True)
)
@settings(max_examples=40, deadline=None)
def test_shuffled_input_produces_sorted_segments(frames):
    shuffled = list(frames)
    random.Random(42).shuffle(shuffled)
    points = [_tp(f, float(f), 0.0) for f in shuffled]
    out = _split_tracks_by_gaps(
        {1: points},
        max_frame_gap=1e9,
        max_spatial_jump=1e9,
        prediction_tolerance=1e9,
    )
    for seg in out.values():
        seg_frames = [p.frame for p in seg]
        assert seg_frames == sorted(seg_frames)
    recovered = sorted(p.frame for seg in out.values() for p in seg)
    assert recovered == sorted(frames)


def test_duplicate_frames_do_not_crash():
    points = [
        _tp(0, 0.0, 0.0),
        _tp(0, 1.0, 1.0),
        _tp(1, 2.0, 2.0),
    ]
    out = _split_tracks_by_gaps({1: points})
    recovered = sorted(_key(p) for seg in out.values() for p in seg)
    assert recovered == sorted(_key(p) for p in points)


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------


@given(n=st.integers(min_value=1, max_value=20))
@settings(max_examples=20, deadline=None)
def test_min_frame_preserved(n):
    points = [_tp(i, 0.0, 0.0) for i in range(n)]
    out = _split_tracks_by_gaps({1: points})
    all_pts = [p for seg in out.values() for p in seg]
    assert min(p.frame for p in all_pts) == 0


def test_output_ids_follow_tid_times_1000_scheme():
    points = [_tp(i, 0.0, 0.0) for i in range(3)]
    out = _split_tracks_by_gaps({7: points})
    assert set(out.keys()) == {7000}


def test_multiple_tracks_have_distinct_output_keys():
    a = [_tp(i, 0.0, 0.0) for i in range(3)]
    b = [_tp(i, 10.0, 10.0) for i in range(3)]
    out = _split_tracks_by_gaps({1: a, 2: b})
    assert set(out.keys()) == {1000, 2000}
