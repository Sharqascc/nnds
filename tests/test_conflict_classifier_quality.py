import json
import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.analysis.conflict_classifier import (
    _angle_between,
    _get_velocity_vector,
    classify_conflict_geometry,
)


def _traj(points):
    return json.dumps(points)


def _straight(
    start_x,
    start_y,
    vx,
    vy,
    frames=(0, 1, 2),
    world=True,
):
    points = []
    for index, frame in enumerate(frames):
        x = start_x + vx * index
        y = start_y + vy * index
        point = {
            "frame": frame,
            "x_pixel": x,
            "y_pixel": y,
        }
        if world:
            point.update(
                {
                    "world_x": 1000.0 + x,
                    "world_y": 2000.0 + y,
                }
            )
        points.append(point)
    return _traj(points)


class TestVelocityEstimation:
    def test_uses_first_and_last_points_in_window(self):
        points = [
            {"frame": 0, "x_pixel": 0, "y_pixel": 0},
            {"frame": 1, "x_pixel": 10, "y_pixel": 0},
            {"frame": 2, "x_pixel": 30, "y_pixel": 0},
            {"frame": 3, "x_pixel": 60, "y_pixel": 0},
        ]

        velocity = _get_velocity_vector(points, before_frame=3)

        # Frames 0, 1, and 2 are before frame 3.
        # The implementation uses frame 0 and frame 2:
        # (30 - 0) / (2 - 0) = 15.
        assert velocity == (15.0, 0.0)

    def test_world_coordinates_take_priority(self):
        points = [
            {
                "frame": 0,
                "x_pixel": 0,
                "y_pixel": 0,
                "world_x": 0,
                "world_y": 0,
            },
            {
                "frame": 2,
                "x_pixel": 10,
                "y_pixel": 0,
                "world_x": 20,
                "world_y": 0,
            },
        ]

        velocity = _get_velocity_vector(points, before_frame=3)

        assert velocity == (10.0, 0.0)

    def test_pixel_fallback_when_world_incomplete(self):
        points = [
            {
                "frame": 0,
                "x_pixel": 0,
                "y_pixel": 0,
                "world_x": 0,
            },
            {
                "frame": 2,
                "x_pixel": 10,
                "y_pixel": 0,
                "world_x": 20,
                "world_y": 0,
            },
        ]

        velocity = _get_velocity_vector(points, before_frame=3)

        assert velocity == (5.0, 0.0)

    def test_conflict_frame_point_is_excluded(self):
        points = [
            {"frame": 0, "x_pixel": 0, "y_pixel": 0},
            {"frame": 5, "x_pixel": 50, "y_pixel": 0},
        ]

        velocity = _get_velocity_vector(points, before_frame=5)

        assert velocity == (0.0, 0.0)


class TestDocumentedCurrentBehavior:
    def test_parallel_similar_speed_is_side_swipe(self):
        trajectory_a = _straight(0, 0, 10, 0)
        trajectory_b = _straight(0, 10, 10, 0)

        result = classify_conflict_geometry(
            trajectory_a,
            trajectory_b,
            conflict_frame=3,
        )

        assert result == "side_swipe"

    def test_parallel_different_speed_is_rear_end(self):
        trajectory_a = _straight(0, 0, 20, 0)
        trajectory_b = _straight(0, 10, 5, 0)

        result = classify_conflict_geometry(
            trajectory_a,
            trajectory_b,
            conflict_frame=3,
        )

        assert result == "rear_end"

    def test_perpendicular_is_crossing(self):
        trajectory_a = _straight(0, 0, 10, 0)
        trajectory_b = _straight(0, 0, 0, 10)

        result = classify_conflict_geometry(
            trajectory_a,
            trajectory_b,
            conflict_frame=3,
        )

        assert result == "crossing"

    def test_opposite_direction_is_head_on(self):
        trajectory_a = _straight(0, 0, 10, 0)
        trajectory_b = _straight(100, 0, -10, 0)

        result = classify_conflict_geometry(
            trajectory_a,
            trajectory_b,
            conflict_frame=3,
        )

        assert result == "head_on"

    def test_oblique_angle_is_other(self):
        trajectory_a = _straight(0, 0, 10, 0)
        trajectory_b = _straight(0, 0, 10, 10)

        result = classify_conflict_geometry(
            trajectory_a,
            trajectory_b,
            conflict_frame=3,
        )

        assert result == "other"


class TestClassifierBoundaries:
    @pytest.mark.parametrize(
        ("angle_degrees", "expected"),
        [
            (29.0, "same-direction"),
            (30.0, "other"),
            (31.0, "other"),
            (59.0, "other"),
            (60.0, "crossing"),
            (90.0, "crossing"),
            (120.0, "crossing"),
            (121.0, "other"),
            (149.0, "other"),
            (150.0, "other"),
            (151.0, "head_on"),
        ],
    )
    def test_angle_boundaries(self, angle_degrees, expected):
        radians = math.radians(angle_degrees)
        trajectory_a = _straight(0, 0, 10, 0)
        trajectory_b = _straight(
            0,
            0,
            10 * math.cos(radians),
            10 * math.sin(radians),
        )

        result = classify_conflict_geometry(
            trajectory_a,
            trajectory_b,
            conflict_frame=3,
        )

        if expected == "same-direction":
            assert result in {"rear_end", "side_swipe"}
        else:
            assert result == expected

    def test_rear_end_speed_boundary(self):
        # speed_a = 10, speed_b = 4.
        # speed_diff = 6 > 0.5 * max(10, 4) = 5, so rear_end.
        trajectory_a = _straight(0, 0, 10, 0)
        trajectory_b = _straight(0, 10, 4, 0)

        assert (
            classify_conflict_geometry(
                trajectory_a,
                trajectory_b,
                conflict_frame=3,
            )
            == "rear_end"
        )

    def test_side_swipe_speed_boundary(self):
        trajectory_a = _straight(0, 0, 10, 0)
        trajectory_b = _straight(0, 10, 6, 0)

        assert (
            classify_conflict_geometry(
                trajectory_a,
                trajectory_b,
                conflict_frame=3,
            )
            == "side_swipe"
        )


class TestRobustness:
    @pytest.mark.parametrize(
        "trajectory_a",
        ["", "not json", "null", "[]", "{}", "123"],
    )
    def test_invalid_first_trajectory_returns_other(self, trajectory_a):
        trajectory_b = _straight(0, 0, 10, 0)

        result = classify_conflict_geometry(
            trajectory_a,
            trajectory_b,
            conflict_frame=3,
        )

        assert result == "other"

    @pytest.mark.parametrize(
        "trajectory_b",
        ["", "not json", "null", "[]", "{}", "123"],
    )
    def test_invalid_second_trajectory_returns_other(self, trajectory_b):
        trajectory_a = _straight(0, 0, 10, 0)

        result = classify_conflict_geometry(
            trajectory_a,
            trajectory_b,
            conflict_frame=3,
        )

        assert result == "other"

    def test_stationary_trajectory_returns_other(self):
        trajectory_a = _traj(
            [
                {"frame": 0, "x_pixel": 0, "y_pixel": 0},
                {"frame": 1, "x_pixel": 0, "y_pixel": 0},
            ]
        )
        trajectory_b = _straight(0, 10, 10, 0)

        result = classify_conflict_geometry(
            trajectory_a,
            trajectory_b,
            conflict_frame=3,
        )

        assert result == "other"

    def test_duplicate_frames_return_other(self):
        trajectory_a = _traj(
            [
                {"frame": 1, "x_pixel": 0, "y_pixel": 0},
                {"frame": 1, "x_pixel": 10, "y_pixel": 0},
            ]
        )
        trajectory_b = _straight(0, 10, 10, 0)

        result = classify_conflict_geometry(
            trajectory_a,
            trajectory_b,
            conflict_frame=3,
        )

        assert result == "other"

    def test_unsorted_points_are_sorted(self):
        trajectory_a = _traj(
            [
                {"frame": 2, "x_pixel": 20, "y_pixel": 0},
                {"frame": 0, "x_pixel": 0, "y_pixel": 0},
                {"frame": 1, "x_pixel": 10, "y_pixel": 0},
            ]
        )
        trajectory_b = _straight(0, 10, 10, 0)

        result = classify_conflict_geometry(
            trajectory_a,
            trajectory_b,
            conflict_frame=3,
        )

        assert result == "side_swipe"


class TestPropertyInvariants:
    @settings(deadline=None)
    @given(
        st.floats(min_value=-1000, max_value=1000),
        st.floats(min_value=-1000, max_value=1000),
        st.floats(min_value=-1000, max_value=1000),
        st.floats(min_value=-1000, max_value=1000),
    )
    def test_classification_is_always_valid(
        self,
        velocity_a_x,
        velocity_a_y,
        velocity_b_x,
        velocity_b_y,
    ):
        trajectory_a = _straight(0, 0, velocity_a_x, velocity_a_y)
        trajectory_b = _straight(0, 10, velocity_b_x, velocity_b_y)

        result = classify_conflict_geometry(
            trajectory_a,
            trajectory_b,
            conflict_frame=3,
        )

        assert result in {
            "rear_end",
            "head_on",
            "crossing",
            "side_swipe",
            "other",
        }

    @settings(deadline=None)
    @given(
        st.floats(min_value=-100, max_value=100),
        st.floats(min_value=-100, max_value=100),
        st.floats(min_value=-100, max_value=100),
        st.floats(min_value=-100, max_value=100),
    )
    def test_classification_is_deterministic(
        self,
        velocity_a_x,
        velocity_a_y,
        velocity_b_x,
        velocity_b_y,
    ):
        trajectory_a = _straight(0, 0, velocity_a_x, velocity_a_y)
        trajectory_b = _straight(0, 10, velocity_b_x, velocity_b_y)

        first = classify_conflict_geometry(
            trajectory_a,
            trajectory_b,
            conflict_frame=3,
        )
        second = classify_conflict_geometry(
            trajectory_a,
            trajectory_b,
            conflict_frame=3,
        )

        assert first == second
