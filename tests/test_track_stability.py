import pytest

from src.analysis.track_stability import (
    TrackPointRow,
    box_size_metrics,
    full_report,
    gap_metrics,
    jump_metrics,
    length_histogram,
    smoothness_metrics,
    tracks_from_rows,
)


def _pt(f, cx=100.0, cy=100.0, w=40.0, h=40.0):
    return TrackPointRow(frame=f, cx=cx, cy=cy, w=w, h=h)


def _straight_track(tid, n=20, vx=5.0):
    return [_pt(i, cx=100 + i * vx) for i in range(n)]


def test_length_histogram_basic():
    tracks = {1: _straight_track(1, 20), 2: _straight_track(2, 3)}
    h = length_histogram(tracks)
    assert h["n_tracks"] == 2
    assert h["min"] == 3
    assert h["max"] == 20
    assert h["n_short"] == 1
    assert h["frac_short"] == pytest.approx(0.5)


def test_length_histogram_empty():
    h = length_histogram({})
    assert h["n_tracks"] == 0


def test_gap_metrics_no_gaps():
    tracks = {1: _straight_track(1, 10)}
    g = gap_metrics(tracks)
    assert g["n_gaps"] == 0
    assert g["tracks_with_large_gap"] == 0


def test_gap_metrics_with_gap():
    pts = [_pt(0), _pt(1), _pt(2), _pt(10), _pt(11)]
    tracks = {1: pts}
    g = gap_metrics(tracks, max_gap=3)
    assert g["n_gaps"] == 1
    assert g["max_gap"] == 8
    assert g["tracks_with_large_gap"] == 1


def test_jump_metrics_constant_velocity():
    tracks = {1: _straight_track(1, n=20, vx=5.0)}
    j = jump_metrics(tracks, threshold_px=10.0)
    # Each step is 5 px, below the 10 px threshold
    assert j["max_jump_px"] == pytest.approx(5.0)
    assert j["n_large_jumps"] == 0


def test_jump_metrics_detects_spike():
    pts = [_pt(i, cx=100 + i * 5) for i in range(10)]
    pts.append(_pt(10, cx=500))  # sudden jump of ~355 px
    tracks = {1: pts}
    j = jump_metrics(tracks, threshold_px=50.0)
    assert j["n_large_jumps"] == 1


def test_smoothness_constant_velocity_zero_accel():
    tracks = {1: _straight_track(1, n=20, vx=5.0)}
    s = smoothness_metrics(tracks)
    assert s["max_accel_px"] == pytest.approx(0.0, abs=1e-9)


def test_smoothness_detects_zig_zag():
    # Alternating +10, -10 -> second difference magnitude is 20
    xs = [0, 10, 0, 10, 0, 10, 0]
    pts = [_pt(i, cx=float(x)) for i, x in enumerate(xs)]
    tracks = {1: pts}
    s = smoothness_metrics(tracks)
    assert s["max_accel_px"] == pytest.approx(20.0)


def test_box_size_stable():
    tracks = {1: _straight_track(1, n=10, vx=5.0)}
    b = box_size_metrics(tracks)
    assert b["n_large_changes"] == 0
    assert b["mean_rel_change"] == pytest.approx(0.0, abs=1e-9)


def test_box_size_jump_detected():
    pts = [_pt(i, w=40.0) for i in range(5)]
    pts.append(_pt(5, w=120.0))  # 200% jump
    tracks = {1: pts}
    b = box_size_metrics(tracks, threshold_frac=0.5)
    assert b["n_large_changes"] == 1


def test_tracks_from_rows():
    rows = [
        {"frame": 0, "track_id": 1, "cx": 10.0, "cy": 20.0, "w": 5.0, "h": 5.0},
        {"frame": 1, "track_id": 1, "cx": 12.0, "cy": 20.0, "w": 5.0, "h": 5.0},
        {"frame": 0, "track_id": 2, "cx": 50.0, "cy": 60.0, "w": 8.0, "h": 8.0},
    ]
    tracks = tracks_from_rows(rows)
    assert set(tracks.keys()) == {1, 2}
    assert len(tracks[1]) == 2


def test_full_report_shape():
    tracks = {1: _straight_track(1, n=10)}
    r = full_report(tracks)
    assert set(r.keys()) == {"length", "gaps", "jumps", "smoothness", "box_size"}
