"""
Tests for BEV calibration functions.
"""
import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_grid_validation_import():
    """Test grid validation functions can be imported."""
    from src.bev.calibration.grid_validation_calibration import (
        load_grid_config, export_results, reprojection_errors
    )
    assert callable(load_grid_config)
    assert callable(export_results)
    assert callable(reprojection_errors)

def test_monte_carlo_benchmark_import():
    """Test Monte Carlo calibration benchmark can be imported."""
    from src.bev.calibration.monte_carlo_calibration_benchmark import (
        run_monte_carlo, run_single_scenario, run_single_trial,
        apply_homography, estimate_homography, compare_methods
    )
    assert callable(run_monte_carlo)
    assert callable(run_single_scenario)
    assert callable(run_single_trial)
    assert callable(apply_homography)
    assert callable(estimate_homography)
    assert callable(compare_methods)

def test_reprojection_errors():
    """Test reprojection error computation returns errors and projected points."""
    from src.bev.calibration.grid_validation_calibration import reprojection_errors
    
    # Create synthetic homography and points
    H = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    pixel_pts = np.array([[10, 20], [30, 40], [50, 60]], dtype=float)
    world_pts = np.array([[10, 20], [30, 40], [50, 60]], dtype=float)
    
    result = reprojection_errors(pixel_pts, world_pts, H)
    assert result is not None
    # Function returns (errors, projected_world) tuple
    errors = result[0]
    projected = result[1]
    assert len(errors) == 3
    assert projected.shape == (3, 2)
