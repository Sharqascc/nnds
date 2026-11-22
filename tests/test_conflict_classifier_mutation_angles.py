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


class TestAngleAudit(unittest.TestCase):
    def test_known_angles(self):
        cases = [
            ((0, 2), (0, 3), 0.0),
            ((0, 2), (0, -3), 180.0),
            ((1, 0), (0, 1), 90.0),
            ((1, 0), (1, 1), 45.0),
            ((1, 0), (-1, 1), 135.0),
            ((3, 4), (4, -3), 90.0),
        ]
        for first, second, expected in cases:
            with self.subTest(first=first, second=second):
                self.assertAlmostEqual(classifier._angle_between(first, second), expected, places=6)

    def test_zero_vectors_in_either_position(self):
        for first, second in [((0, 0), (2, 3)), ((2, 3), (0, 0)), ((0, 0), (0, 0))]:
            with self.subTest(first=first, second=second):
                self.assertEqual(classifier._angle_between(first, second), 0.0)

    def test_nonzero_unit_vectors_are_not_zero_vectors(self):
        self.assertAlmostEqual(classifier._angle_between((1, 0), (0, 1)), 90.0)
        self.assertAlmostEqual(classifier._angle_between((0, 1), (-1, 0)), 90.0)

    def test_tiny_perpendicular_vectors_return_right_angle(self):
        self.assertEqual(classifier._angle_between((1e-200, 0), (0, 1e-200)), 90.0)

    def test_exact_angle_decision_boundaries(self):
        cases = [
            (29.999, "side_swipe"),
            (30.0, "other"),
            (30.001, "other"),
            (59.999, "other"),
            (60.0, "crossing"),
            (60.001, "crossing"),
            (119.999, "crossing"),
            (120.0, "crossing"),
            (120.001, "other"),
            (149.999, "other"),
            (150.0, "other"),
            (150.001, "head_on"),
        ]
        first = trajectory(10, 0)
        second = trajectory(10, 0)
        for angle, expected in cases:
            with self.subTest(angle=angle):
                with patch.object(classifier, "_angle_between", return_value=angle):
                    self.assertEqual(
                        classifier.classify_conflict_geometry(first, second, conflict_frame=4),
                        expected,
                    )

    def test_speed_difference_boundary_in_both_actor_orders(self):
        cases = [(4.999, "rear_end"), (5.0, "side_swipe"), (5.001, "side_swipe")]
        for slower_speed, expected in cases:
            for speed_a, speed_b in [(10.0, slower_speed), (slower_speed, 10.0)]:
                with self.subTest(speed_a=speed_a, speed_b=speed_b):
                    self.assertEqual(
                        classifier.classify_conflict_geometry(
                            trajectory(speed_a, 0), trajectory(speed_b, 0), conflict_frame=4
                        ),
                        expected,
                    )
