
import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from src.bev.bev_mapper import BEVMapper


def _make_mapper():
    H = np.eye(3, dtype=np.float32)
    bounds = {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100}
    res = (100, 100)
    return BEVMapper(H, bounds, res)

@given(st.floats(min_value=0, max_value=99.9), st.floats(min_value=0, max_value=99.9))
def test_pixel_to_world_identity(x, y):
    mapper = _make_mapper()
    world = mapper.pixel_to_world((x, y))
    assert world is not None
    assert abs(world[0] - x) < 1e-3
    assert abs(world[1] - y) < 1e-3

@given(st.floats(min_value=0, max_value=99.9), st.floats(min_value=0, max_value=99.9))
def test_world_to_bev_bounds(wx, wy):
    mapper = _make_mapper()
    u, v = mapper.world_to_bev((wx, wy))
    assert 0 <= u < mapper.bev_w
    assert 0 <= v < mapper.bev_h
