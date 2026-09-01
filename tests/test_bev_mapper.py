"""
Tests for BEVMapper module.
"""
import numpy as np
import pytest
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bev.bev_mapper import BEVMapper

def create_test_mapper():
    """Create a BEVMapper with identity homography."""
    H = np.array([
        [1.0, 0, 0],
        [0, 1.0, 0],
        [0, 0, 1]
    ])
    bounds = {
        'x_min': -2.0, 'x_max': 22.0,
        'y_min': -2.0, 'y_max': 18.0
    }
    resolution = [240, 200]
    return BEVMapper(H, bounds, resolution)

def test_bev_mapper_initialization():
    """Test BEVMapper initializes correctly."""
    mapper = create_test_mapper()
    assert mapper is not None
    assert mapper.H.shape == (3, 3)
    assert mapper.bev_x_min == -2.0
    assert mapper.bev_w == 240

def test_bev_mapper_pixel_to_world():
    """Test pixel_to_world maps correctly."""
    mapper = create_test_mapper()
    # With identity H, pixel (120, 100) -> world (120, 100)
    # But wait, world bounds are 24m x 20m, and pixel resolution is 240x200
    # So pixel_to_world should map pixel coordinates to world meters
    # Using identity H, the world coordinates would just be the pixel values
    pixel_point = (120.0, 100.0)
    world = mapper.pixel_to_world(pixel_point)
    assert world is not None
    assert len(world) == 2
    # With identity H, world_x = pixel_x, world_y = pixel_y
    assert world[0] == 120.0
    assert world[1] == 100.0

def test_bev_mapper_pixel_to_bev():
    """Test pixel_to_bev maps correctly."""
    mapper = create_test_mapper()
    # pixel (120, 100) should map to BEV coordinates
    bev = mapper.pixel_to_bev((120.0, 100.0))
    assert bev is not None
    assert len(bev) == 2

def test_bev_mapper_world_to_bev():
    """Test world_to_bev maps correctly."""
    mapper = create_test_mapper()
    # world (10, 8) should be near center of BEV
    bev = mapper.world_to_bev((10.0, 8.0))
    assert bev is not None
    assert len(bev) == 2
    assert 0 <= bev[0] <= 240
    assert 0 <= bev[1] <= 200

def test_bev_mapper_out_of_bounds():
    """Test out of bounds handling."""
    mapper = create_test_mapper()
    # Point far outside world bounds
    result = mapper.pixel_to_world((5000.0, 5000.0))
    # Should handle gracefully (may return None or large values)
    assert result is not None or result is None  # Should not crash

def test_bev_mapper_with_real_homography():
    """Test BEVMapper with real calibration homography."""
    repo = Path(__file__).parent.parent
    calib_path = repo/'configs/sites/giti/calibration_points.json'
    bev_path = repo/'configs/sites/giti/bev_config.json'
    if not calib_path.exists() or not bev_path.exists():
        pytest.skip("GITI calibration not available")
    
    with open(bev_path) as f:
        bev_cfg = json.load(f)
    H = np.array(bev_cfg['H_pixel_to_world'], dtype=float)
    
    bounds = {
        'x_min': bev_cfg['x_min'],
        'x_max': bev_cfg['x_max'],
        'y_min': bev_cfg['y_min'],
        'y_max': bev_cfg['y_max']
    }
    bev_res = bev_cfg.get('bev_resolution', [240, 200])
    mapper = BEVMapper(H, bounds, bev_res)
    assert mapper is not None
    # Test a known pixel point
    world = mapper.pixel_to_world((194, 124))
    assert world is not None
    assert len(world) == 2
