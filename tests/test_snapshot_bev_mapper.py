import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.bev.bev_mapper import BEVMapper


def test_bev_mapping_snapshot(snapshot):
    H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    bounds = {"x_min": 0, "x_max": 10, "y_min": 0, "y_max": 10}
    res = (100, 100)
    mapper = BEVMapper(H, bounds, res)
    # Test world_to_bev for a few points
    points = [(0, 0), (5, 5), (10, 10), (2.5, 7.5)]
    result = [mapper.world_to_bev(p) for p in points]
    assert result == snapshot
