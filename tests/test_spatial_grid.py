"""
Tests for spatial grid module.
"""
import numpy as np
import pytest
import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.grid_trajectory.spatial_grid import SpatialGrid

def create_test_grid():
    """Create a SpatialGrid with known corners."""
    tmp_dir = Path(tempfile.mkdtemp())
    grid_config = {
        "corners": {
            "top_left": [0, 0],
            "top_right": [1000, 0],
            "bottom_left": [0, 500],
            "bottom_right": [1000, 500]
        },
        "configuration": {
            "cell_size": 100,
            "naming_style": "cell_{col}_{row}"
        }
    }
    config_path = tmp_dir / 'grid_config.json'
    with open(config_path, 'w') as f:
        json.dump(grid_config, f)
    return SpatialGrid(config_path), config_path

def test_spatial_grid_initialization():
    """Test SpatialGrid initializes correctly."""
    grid, _ = create_test_grid()
    assert grid.corners is not None
    assert grid.cell_size == 100
    assert grid.naming_style == 'cell_{col}_{row}'

def test_spatial_grid_cell_detection():
    """Test cell detection from pixel coordinates."""
    grid, _ = create_test_grid()
    cell = grid.get_cell_from_pixels(50, 50)
    assert cell == 'cell_A_1'
    
    cell = grid.get_cell_from_pixels(150, 150)
    assert cell == 'cell_B_2'

def test_spatial_grid_out_of_bounds():
    """Test out of bounds handling."""
    grid, _ = create_test_grid()
    cell = grid.get_cell_from_pixels(1500, 1500)
    assert cell == 'OUT_OF_BOUNDS'

def test_spatial_grid_boundary_cell():
    """Test boundary conditions."""
    grid, _ = create_test_grid()
    cell = grid.get_cell_from_pixels(100, 100)
    assert cell in ['cell_B_2', 'cell_A_1', 'cell_B_1', 'cell_A_2']

def test_spatial_grid_naming():
    """Test cell naming follows naming_style."""
    grid, _ = create_test_grid()
    cell = grid.get_cell_from_pixels(250, 250)
    assert cell == 'cell_C_3'

def test_spatial_grid_get_cell_center():
    """Test cell center retrieval."""
    grid, _ = create_test_grid()
    center = grid.get_cell_center('cell_A_1')
    assert center is not None
    x, y = center
    assert x == 50
    assert y == 50

def test_spatial_grid_get_cell_bounds():
    """Test cell bounds retrieval."""
    grid, _ = create_test_grid()
    bounds = grid.get_cell_bounds('cell_A_1')
    assert bounds is not None
    x1, y1, x2, y2 = bounds
    assert x2 - x1 == 100
    assert y2 - y1 == 100

def test_spatial_grid_get_stats():
    """Test get_stats returns dict."""
    grid, _ = create_test_grid()
    stats = grid.get_stats()
    assert isinstance(stats, dict)
