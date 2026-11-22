import unittest

from src.analysis import conflict_classifier as classifier

velocity = classifier._get_velocity_vector


class TestVelocityAudit(unittest.TestCase):
    def assertVectorAlmostEqual(self, actual, expected):
        self.assertEqual(len(actual), 2)
        for obtained, wanted in zip(actual, expected, strict=True):
            self.assertAlmostEqual(obtained, wanted)

    def test_default_window_uses_last_five_sorted_points(self):
        points = [
            {"frame": frame, "x_pixel": frame * frame, "y_pixel": 3 - 2 * frame * frame}
            for frame in [6, 1, 4, 2, 5, 3]
        ]
        self.assertVectorAlmostEqual(velocity(points, 7), (8.0, -16.0))

    def test_invalid_container_returns_exact_zero(self):
        for points in [None, 123, "invalid", {"frame": 1}]:
            with self.subTest(points=points):
                self.assertEqual(velocity(points, 10), (0.0, 0.0))

    def test_excludes_points_at_and_after_cutoff(self):
        points = [
            {"frame": 6, "x_pixel": 1000, "y_pixel": -900},
            {"frame": 1, "x_pixel": 10, "y_pixel": 20},
            {"frame": 5, "x_pixel": 500, "y_pixel": 600},
            {"frame": 3, "x_pixel": 18, "y_pixel": 14},
        ]
        self.assertVectorAlmostEqual(velocity(points, 5), (4.0, -3.0))

    def test_ignores_non_dictionary_entries(self):
        points = [
            {"frame": 1, "x_pixel": 10, "y_pixel": 20},
            None,
            "noise",
            {"frame": 3, "x_pixel": 18, "y_pixel": 14},
        ]
        self.assertVectorAlmostEqual(velocity(points, 5), (4.0, -3.0))

    def test_uncomparable_frame_uses_existing_zero_fallback(self):
        points = [
            {"frame": 1, "x_pixel": 10, "y_pixel": 20},
            {"frame": None, "x_pixel": 18, "y_pixel": 14},
        ]
        self.assertEqual(velocity(points, 5), (0.0, 0.0))

    def test_prefers_complete_world_coordinates(self):
        points = [
            {"frame": 2, "x_pixel": 100, "y_pixel": 200, "world_x": 7, "world_y": -2},
            {"frame": 6, "x_pixel": 900, "y_pixel": 800, "world_x": 17, "world_y": -14},
        ]
        self.assertVectorAlmostEqual(velocity(points, 7), (2.5, -3.0))
