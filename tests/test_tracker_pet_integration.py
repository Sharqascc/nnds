"""Integration: tracker output -> grid PET chain.

Feeds TrackPoint sequences through the geometry helpers in
yolo_cpu_grid_pet and verifies PET-style conflict detection works
end to end. Runs in PR CI.
"""

from __future__ import annotations

from src.analysis.grid_trajectory import yolo_cpu_grid_pet as yp


def _pt(frame: int, x: float, y: float) -> yp.TrackPoint:
    return yp.TrackPoint(frame=frame, x=x, y=y, cls_id=2, cls_name="car", conf=0.9)


def test_single_track_yields_entry_exit_interval():
    pts = [_pt(i, float(i), 0.0) for i in range(20)]
    e = yp._entry_exit_frames(pts, 10.0, 0.0, 5.0)
    assert e is not None
    entry, exit_ = e
    assert entry <= exit_
    assert entry >= 0


def test_two_crossing_tracks_produce_conflict_point():
    a = [_pt(0, 0.0, 0.0), _pt(1, 5.0, 5.0)]
    b = [_pt(0, 0.0, 5.0), _pt(1, 5.0, 0.0)]
    inter = yp._pair_conflict_point(a, b)
    assert inter is not None
    assert abs(inter[0] - 2.5) < 1e-6
    assert abs(inter[1] - 2.5) < 1e-6


def test_no_conflict_for_parallel_tracks():
    a = [_pt(0, 0.0, 0.0), _pt(1, 1.0, 0.0), _pt(2, 2.0, 0.0)]
    b = [_pt(0, 0.0, 10.0), _pt(1, 1.0, 10.0), _pt(2, 2.0, 10.0)]
    assert yp._pair_conflict_point(a, b) is None


def test_allowed_classes_are_the_expected_set():
    assert set(yp.ALLOWED_CLASSES.keys()) == {0, 1, 2, 3, 5, 7}
    assert yp.ALLOWED_CLASSES[0] == "person"
    assert yp.ALLOWED_CLASSES[2] == "car"


def test_trackpoint_dataclass_fields():
    p = _pt(5, 1.0, 2.0)
    assert p.frame == 5
    assert p.x == 1.0
    assert p.y == 2.0
    assert p.cls_id == 2
