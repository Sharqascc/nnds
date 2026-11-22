import json
import unittest
from unittest.mock import patch

from src.analysis import conflict_classifier as classifier


def trajectory(vx, vy):
    return json.dumps(
        [
            {"frame": 1, "x_pixel": 10.0, "y_pixel": 20.0},
            {"frame": 3, "x_pixel": 10.0 + 2 * vx, "y_pixel": 20.0 + 2 * vy},
        ]
    )


class TestRemainingAudit(unittest.TestCase):
    def assertVectorAlmostEqual(self, actual, expected):
        self.assertEqual(len(actual), 2)
        for obtained, wanted in zip(actual, expected, strict=True):
            self.assertAlmostEqual(obtained, wanted)

    def test_fractional_pixel_velocity_in_both_components(self):
        points = [
            {"frame": 2, "x_pixel": 7, "y_pixel": 9},
            {"frame": 6, "x_pixel": 17, "y_pixel": -2},
        ]
        self.assertVectorAlmostEqual(classifier._get_velocity_vector(points, 7), (2.5, -2.75))

    def test_fractional_world_velocity_in_both_components(self):
        points = [
            {"frame": 2, "x_pixel": 100, "y_pixel": 200, "world_x": 7, "world_y": 9},
            {"frame": 6, "x_pixel": 900, "y_pixel": 800, "world_x": 17, "world_y": -2},
        ]
        self.assertVectorAlmostEqual(classifier._get_velocity_vector(points, 7), (2.5, -2.75))

    def test_partial_world_coordinates_use_pixel_fallback(self):
        for missing_endpoint in (0, 1):
            for missing_key in ("world_x", "world_y"):
                for missing_value in ("absent", "none"):
                    with self.subTest(
                        endpoint=missing_endpoint, key=missing_key, value=missing_value
                    ):
                        points = [
                            {
                                "frame": 2,
                                "x_pixel": 7,
                                "y_pixel": 9,
                                "world_x": 100,
                                "world_y": 200,
                            },
                            {
                                "frame": 6,
                                "x_pixel": 17,
                                "y_pixel": -2,
                                "world_x": 900,
                                "world_y": 800,
                            },
                        ]
                        if missing_value == "absent":
                            del points[missing_endpoint][missing_key]
                        else:
                            points[missing_endpoint][missing_key] = None
                        self.assertVectorAlmostEqual(
                            classifier._get_velocity_vector(points, 7), (2.5, -2.75)
                        )

    def test_classifier_requests_five_point_window_for_both_actors(self):
        first = trajectory(3, 0)
        second = trajectory(0, 3)
        with patch.object(
            classifier, "_get_velocity_vector", wraps=classifier._get_velocity_vector
        ) as helper:
            self.assertEqual(classifier.classify_conflict_geometry(first, second, 4), "crossing")
        self.assertEqual(helper.call_count, 2)
        for call in helper.call_args_list:
            self.assertEqual(call.kwargs.get("window"), 5)

    def test_stationary_actor_returns_other_in_both_orders(self):
        stationary = trajectory(0, 0)
        moving = trajectory(3, 0)
        for first, second in [(stationary, moving), (moving, stationary), (stationary, stationary)]:
            with self.subTest(first=first, second=second):
                self.assertEqual(classifier.classify_conflict_geometry(first, second, 4), "other")

    def test_nonzero_unit_velocities_are_not_stationary(self):
        for vx, vy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            with self.subTest(vx=vx, vy=vy):
                track = trajectory(vx, vy)
                self.assertEqual(
                    classifier.classify_conflict_geometry(track, track, 4), "side_swipe"
                )
