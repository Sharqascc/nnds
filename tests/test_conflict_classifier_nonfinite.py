import json
import math
import unittest

from src.analysis import conflict_classifier as classifier


def trajectory(vx, vy, world=False):
    if world:
        return json.dumps(
            [
                {
                    "frame": 1,
                    "x_pixel": 0.0,
                    "y_pixel": 0.0,
                    "world_x": 0.0,
                    "world_y": 0.0,
                },
                {
                    "frame": 2,
                    "x_pixel": 1.0,
                    "y_pixel": 1.0,
                    "world_x": vx,
                    "world_y": vy,
                },
            ]
        )
    return json.dumps(
        [
            {"frame": 1, "x_pixel": 0.0, "y_pixel": 0.0},
            {"frame": 2, "x_pixel": vx, "y_pixel": vy},
        ]
    )


class TestNonfinitePolicy(unittest.TestCase):
    def test_private_angle_rejects_nonfinite_components(self):
        for value in (math.nan, math.inf, -math.inf):
            for bad_vector in ((value, 1.0), (1.0, value)):
                for first, second in [
                    (bad_vector, (1.0, 0.0)),
                    ((1.0, 0.0), bad_vector),
                    ((0.0, 0.0), bad_vector),
                ]:
                    with self.subTest(first=first, second=second):
                        with self.assertRaises(ValueError):
                            classifier._angle_between(first, second)

    def test_public_classifier_returns_other_for_nonfinite_velocity(self):
        for world in (False, True):
            valid = trajectory(1.0, 0.0, world=world)
            for value in (math.nan, math.inf, -math.inf):
                for vx, vy in ((value, 1.0), (1.0, value)):
                    invalid = trajectory(vx, vy, world=world)
                    for first, second in [
                        (invalid, valid),
                        (valid, invalid),
                    ]:
                        with self.subTest(
                            world=world,
                            value=value,
                            vx=vx,
                            vy=vy,
                            invalid_actor="A" if first == invalid else "B",
                        ):
                            self.assertEqual(
                                classifier.classify_conflict_geometry(
                                    first, second, conflict_frame=3
                                ),
                                "other",
                            )

    def test_public_classifier_handles_finite_extreme_geometry(self):
        cases = [
            (
                trajectory(1e-200, 0.0),
                trajectory(0.0, 1e-200),
                "crossing",
            ),
            (
                trajectory(1e200, 1e200),
                trajectory(-1e200, -1e200),
                "head_on",
            ),
        ]
        for first, second, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(
                    classifier.classify_conflict_geometry(first, second, conflict_frame=3),
                    expected,
                )
