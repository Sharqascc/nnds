from hypothesis import given, settings
from hypothesis import strategies as st

from src.analysis.track_stability import (
    TrackPointRow,
    box_size_metrics,
    gap_metrics,
    jump_metrics,
    length_histogram,
    smoothness_metrics,
)


def _straight_track(frames, vx):
    return [TrackPointRow(frame=f, cx=float(f) * vx, cy=0.0, w=10.0, h=10.0) for f in frames]


@given(
    st.lists(st.integers(0, 100), min_size=1, max_size=30, unique=True).map(sorted),
    st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50)
def test_smoothness_zero_for_constant_velocity(frames, vx):
    pts = _straight_track(frames, vx)
    tracks = {1: pts}
    s = smoothness_metrics(tracks)
    if len(frames) >= 3:
        # If frames are consecutive, second diff is 0 for a straight line
        consecutive = all(frames[i + 1] - frames[i] == 1 for i in range(len(frames) - 1))
        if consecutive:
            assert s["max_accel_px"] < 1e-6


@given(st.integers(1, 50))
@settings(max_examples=50)
def test_length_histogram_consistent(n):
    tracks = {1: _straight_track(list(range(n)), 1.0)}
    h = length_histogram(tracks)
    assert h["n_tracks"] == 1
    assert h["mean"] == n
    assert h["min"] == h["max"] == n


@given(st.integers(0, 5))
@settings(max_examples=30)
def test_gap_metrics_non_negative(n_gaps):
    frames = list(range(20))
    for g in range(n_gaps):
        # Insert a gap after position g
        frames[g + 5] += 10
    tracks = {1: _straight_track(sorted(frames), 1.0)}
    g = gap_metrics(tracks)
    assert g["n_gaps"] >= 0
    assert g["max_gap"] >= 0


@given(
    st.lists(
        st.tuples(
            st.integers(0, 200),
            st.floats(-500.0, 500.0, allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
        max_size=40,
    )
)
@settings(max_examples=50)
def test_jump_metrics_bounded(rows):
    pts = [TrackPointRow(frame=f, cx=x, cy=0.0, w=10.0, h=10.0) for f, x in rows]
    tracks = {1: pts}
    j = jump_metrics(tracks)
    assert j["mean_jump_px"] >= 0
    assert j["max_jump_px"] >= 0


@given(
    st.lists(
        st.floats(0.1, 500.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=30,
    )
)
@settings(max_examples=50)
def test_box_size_change_non_negative(ws):
    pts = [TrackPointRow(frame=i, cx=0.0, cy=0.0, w=w, h=w) for i, w in enumerate(ws)]
    tracks = {1: pts}
    b = box_size_metrics(tracks)
    assert b["mean_rel_change"] >= 0
    assert b["p95_rel_change"] >= 0
