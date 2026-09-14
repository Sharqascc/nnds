import pytest

from src.analysis.grid_trajectory.uvh_coco_fused_grid_pet import (
    TrackPoint,
    _angle_diff_deg,
    _heading_near_point,
    _track_missing_ratio,
)


@pytest.mark.parametrize(
    "min_f, max_f, n, expected",
    [
        (0, 99, 100, 0.0),
        (0, 99, 90, 0.10),
        (0, 99, 89, 0.11),
        (0, 99, 50, 0.50),
        (5, 5, 1, 0.0),
        (5, 5, 0, 1.0),
        (10, 5, 3, 1.0),
    ],
)
def test_track_missing_ratio(min_f, max_f, n, expected):
    meta = {"min_frame": min_f, "max_frame": max_f}
    assert _track_missing_ratio(meta, n) == pytest.approx(expected)


def _pt(frame: int, x: float, y: float) -> TrackPoint:
    """Convenience constructor for a TrackPoint in tests."""
    return TrackPoint(
        frame=frame,
        x=x,
        y=y,
        cls_id=0,
        cls_name="test",
        conf=1.0,
    )


def _line(frame_end: int, dx: float, dy: float) -> list[TrackPoint]:
    """A straight-line track from (0,0) to (frame_end*dx, frame_end*dy)."""
    return [_pt(f, f * dx, f * dy) for f in range(frame_end + 1)]


@pytest.mark.parametrize(
    "dx, dy, expected",
    [
        (1.0, 0.0, 0.0),  # right
        (0.0, 1.0, 90.0),  # down
        (-1.0, 0.0, 180.0),  # left
        (0.0, -1.0, 270.0),  # up
        (1.0, 1.0, 45.0),  # down-right
        (-1.0, 1.0, 135.0),  # down-left
        (1.0, -1.0, 315.0),  # up-right
        (-1.0, -1.0, 225.0),  # up-left
    ],
)
def test_heading_near_point_directions(dx, dy, expected):
    pts = _line(20, dx, dy)
    # cx, cy near the middle of the line
    cx, cy = 10 * dx, 10 * dy
    h = _heading_near_point(pts, cx, cy)
    assert h is not None
    assert h == pytest.approx(expected)


def test_heading_near_point_empty():
    assert _heading_near_point([], 0.0, 0.0) is None


def test_heading_near_point_single():
    assert _heading_near_point([_pt(0, 5.0, 5.0)], 5.0, 5.0) is None


def test_heading_near_point_too_few_in_window():
    pts = [_pt(0, 0.0, 0.0), _pt(1, 10.0, 0.0)]
    assert _heading_near_point(pts, 0.0, 0.0, half_window=1) is None


def test_heading_near_point_stationary():
    pts = [_pt(f, 5.0, 5.0) for f in range(10)]
    assert _heading_near_point(pts, 5.0, 5.0) is None


def test_heading_near_point_window_truncates_at_start():
    # closest point is index 0, window covers indices 0..15
    pts = _line(20, 1.0, 0.0)
    h = _heading_near_point(pts, 0.0, 0.0)
    assert h is not None
    assert h == pytest.approx(0.0)


def test_heading_near_point_window_truncates_at_end():
    # closest point is index 20, window covers indices 5..20
    pts = _line(20, 1.0, 0.0)
    h = _heading_near_point(pts, 20.0, 0.0)
    assert h is not None
    assert h == pytest.approx(0.0)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (0.0, 0.0, 0.0),
        (0.0, 90.0, 90.0),
        (0.0, 180.0, 180.0),
        (0.0, 270.0, 90.0),
        (10.0, 350.0, 20.0),
        (350.0, 10.0, 20.0),
        (179.0, 1.0, 178.0),
        (0.0, 359.0, 1.0),
        (359.0, 1.0, 2.0),
    ],
)
def test_angle_diff_deg(a, b, expected):
    assert _angle_diff_deg(a, b) == pytest.approx(expected)


@pytest.mark.parametrize(
    "a, b",
    [(None, 0.0), (0.0, None), (None, None)],
)
def test_angle_diff_deg_none(a, b):
    assert _angle_diff_deg(a, b) is None
