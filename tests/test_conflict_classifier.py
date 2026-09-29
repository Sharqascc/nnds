from src.analysis.conflict_classifier import (
    _angle_between,
    _get_velocity_vector,
    classify_conflict_geometry,
)


def test_get_velocity_vector_insufficient_points():
    """Line 28: fewer than 2 points before before_frame."""
    points = [{"frame": 5, "x_pixel": 10, "y_pixel": 10}]
    assert _get_velocity_vector(points, before_frame=10) == (0.0, 0.0)


def test_get_velocity_vector_window_one():
    """Line 35: after slicing last window=1, fewer than 2 points remain."""
    points = [
        {"frame": 1, "x_pixel": 0, "y_pixel": 0},
        {"frame": 2, "x_pixel": 10, "y_pixel": 10},
    ]
    # window=1 reduces to last 1 point, triggering len(pts) < 2
    assert _get_velocity_vector(points, before_frame=3, window=1) == (0.0, 0.0)


def test_get_velocity_vector_pixel_fallback():
    """Lines 42-43: no world coordinates, use pixel coords."""
    points = [
        {"frame": 1, "x_pixel": 0, "y_pixel": 0},
        {"frame": 2, "x_pixel": 10, "y_pixel": 20},
    ]
    vx, vy = _get_velocity_vector(points, before_frame=3)
    assert vx == 10.0
    assert vy == 20.0


def test_angle_between_zero_magnitude():
    """Line 54: one vector has zero magnitude."""
    assert _angle_between((0, 0), (1, 1)) == 0.0


def test_classify_conflict_geometry_invalid_json():
    """Lines 74-75: invalid JSON returns 'other'."""
    assert classify_conflict_geometry("not json", "{}", conflict_frame=10) == "other"


def _make_traj(points):
    import json

    return json.dumps(points)


def test_classify_rear_end():
    """Lines 95-96: same direction, significant speed difference."""
    # Track A fast, Track B slow, same direction along x-axis
    traj_a = _make_traj(
        [
            {"frame": 1, "x_pixel": 0, "y_pixel": 0, "world_x": 0, "world_y": 0},
            {"frame": 2, "x_pixel": 100, "y_pixel": 0, "world_x": 100, "world_y": 0},
        ]
    )
    traj_b = _make_traj(
        [
            {"frame": 1, "x_pixel": 0, "y_pixel": 0, "world_x": 0, "world_y": 0},
            {"frame": 2, "x_pixel": 10, "y_pixel": 0, "world_x": 10, "world_y": 0},
        ]
    )
    result = classify_conflict_geometry(traj_a, traj_b, conflict_frame=3)
    assert result == "rear_end"


def test_classify_side_swipe():
    """Lines 97-98: same direction, similar speeds."""
    traj_a = _make_traj(
        [
            {"frame": 1, "x_pixel": 0, "y_pixel": 0, "world_x": 0, "world_y": 0},
            {"frame": 2, "x_pixel": 10, "y_pixel": 0, "world_x": 10, "world_y": 0},
        ]
    )
    traj_b = _make_traj(
        [
            {"frame": 1, "x_pixel": 0, "y_pixel": 0, "world_x": 0, "world_y": 0},
            {"frame": 2, "x_pixel": 12, "y_pixel": 0, "world_x": 12, "world_y": 0},
        ]
    )
    result = classify_conflict_geometry(traj_a, traj_b, conflict_frame=3)
    assert result == "side_swipe"


def test_classify_other_angle():
    """Line 104: angle not in any specific category."""
    traj_a = _make_traj(
        [
            {"frame": 1, "x_pixel": 0, "y_pixel": 0, "world_x": 0, "world_y": 0},
            {"frame": 2, "x_pixel": 10, "y_pixel": 0, "world_x": 10, "world_y": 0},
        ]
    )
    traj_b = _make_traj(
        [
            {"frame": 1, "x_pixel": 0, "y_pixel": 0, "world_x": 0, "world_y": 0},
            {"frame": 2, "x_pixel": 7, "y_pixel": 7, "world_x": 7, "world_y": 7},
        ]
    )
    result = classify_conflict_geometry(traj_a, traj_b, conflict_frame=3)
    assert result == "other"


def test_get_velocity_vector_same_frame():
    """Line 35: two points with same frame -> dt <= 0."""
    points = [
        {"frame": 1, "x_pixel": 0, "y_pixel": 0},
        {"frame": 1, "x_pixel": 10, "y_pixel": 10},
    ]
    assert _get_velocity_vector(points, before_frame=2) == (0.0, 0.0)


def test_classify_conflict_geometry_empty_input():
    """Line 69: empty trajectory strings."""
    assert classify_conflict_geometry("", "{}", conflict_frame=10) == "other"
    assert classify_conflict_geometry("{}", "", conflict_frame=10) == "other"
    assert classify_conflict_geometry("", "", conflict_frame=10) == "other"


def test_classify_conflict_geometry_zero_velocity():
    """Line 81: one track has zero velocity vector."""
    traj_a = _make_traj(
        [
            {"frame": 1, "x_pixel": 0, "y_pixel": 0, "world_x": 0, "world_y": 0},
            {"frame": 2, "x_pixel": 10, "y_pixel": 0, "world_x": 10, "world_y": 0},
        ]
    )
    # Track B has only one point before conflict_frame -> zero velocity
    traj_b = _make_traj(
        [
            {"frame": 1, "x_pixel": 0, "y_pixel": 0, "world_x": 0, "world_y": 0},
        ]
    )
    result = classify_conflict_geometry(traj_a, traj_b, conflict_frame=3)
    assert result == "other"


def test_classify_head_on():
    """Line 100: angle > 150 degrees."""
    traj_a = _make_traj(
        [
            {"frame": 1, "x_pixel": 0, "y_pixel": 0, "world_x": 0, "world_y": 0},
            {"frame": 2, "x_pixel": 10, "y_pixel": 0, "world_x": 10, "world_y": 0},
        ]
    )
    traj_b = _make_traj(
        [
            {"frame": 1, "x_pixel": 0, "y_pixel": 0, "world_x": 0, "world_y": 0},
            {"frame": 2, "x_pixel": -10, "y_pixel": 0, "world_x": -10, "world_y": 0},
        ]
    )
    result = classify_conflict_geometry(traj_a, traj_b, conflict_frame=3)
    assert result == "head_on"


def test_classify_crossing():
    """Line 102: angle between 60 and 120 degrees."""
    traj_a = _make_traj(
        [
            {"frame": 1, "x_pixel": 0, "y_pixel": 0, "world_x": 0, "world_y": 0},
            {"frame": 2, "x_pixel": 10, "y_pixel": 0, "world_x": 10, "world_y": 0},
        ]
    )
    traj_b = _make_traj(
        [
            {"frame": 1, "x_pixel": 0, "y_pixel": 0, "world_x": 0, "world_y": 0},
            {"frame": 2, "x_pixel": 0, "y_pixel": 10, "world_x": 0, "world_y": 10},
        ]
    )
    result = classify_conflict_geometry(traj_a, traj_b, conflict_frame=3)
    assert result == "crossing"
