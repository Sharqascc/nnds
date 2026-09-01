"""
Tests for giti_bev_calib module.
"""
import numpy as np
import pytest
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bev.giti_bev_calib import load_giti_homography

def test_load_giti_homography_import():
    """Test that load_giti_homography can be imported."""
    assert callable(load_giti_homography)

def test_load_giti_homography_with_calibration():
    """Test loading homography from calibration file."""
    repo = Path(__file__).parent.parent
    calib_path = repo/'configs/sites/giti/calibration_points.json'
    if not calib_path.exists():
        pytest.skip("GITI calibration not available")
    
    H, pixel_pts, world_pts = load_giti_homography(str(calib_path))
    assert H is not None
    assert H.shape == (3, 3)
    assert pixel_pts is not None
    assert world_pts is not None

def test_load_giti_homography_invalid_file():
    """Test loading homography from invalid file raises exception."""
    with pytest.raises(Exception):
        load_giti_homography('/path/to/nonexistent.json')

def test_load_giti_homography_with_stats():
    """Test loading homography with stats."""
    repo = Path(__file__).parent.parent
    calib_path = repo/'configs/sites/giti/calibration_points.json'
    if not calib_path.exists():
        pytest.skip("GITI calibration not available")
    
    result = load_giti_homography(str(calib_path), return_stats=True)
    assert isinstance(result, tuple)
    assert len(result) == 4
    # Stats dict has these keys
    stats = result[3]
    assert 'mean_error' in stats
    assert 'max_error' in stats
    assert 'inlier_ratio' in stats
