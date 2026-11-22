import unittest

from src.analysis import conflict_classifier as classifier


class TestClampAudit(unittest.TestCase):
    def test_nearly_out_of_range_cosine_for_opposite_vectors(self):
        first = (4.143968989934873e-38, 4.219205981879423e-38)
        second = (-first[0], -first[1])
        self.assertAlmostEqual(classifier._angle_between(first, second), 180.0, places=6)
