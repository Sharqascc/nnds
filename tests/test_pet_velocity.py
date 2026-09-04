import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.pet_conflict_checker import PETConflictChecker


def create_checker():
    """Create a minimal PETConflictChecker for testing _estimate_velocity."""
    checker = PETConflictChecker.__new__(PETConflictChecker)
    return checker


def test_velocity_constant():
    """Test constant velocity estimation from trajectory (5 m/s)."""
    checker = create_checker()
    traj = pd.DataFrame(
        {"x": np.arange(10) * 0.5, "y": np.zeros(10), "timestamp": np.arange(10) * 0.1}
    )
    vel = checker._estimate_velocity(traj)
    assert abs(vel - 5.0) < 0.1, f"Expected ~5 m/s, got {vel}"


def test_velocity_varying():
    """Test velocity estimation with varying speed."""
    checker = create_checker()
    x = []
    ts = []
    speed = 1.0
    for i in range(10):
        x.append(speed * i * 0.1)
        ts.append(i * 0.1)
        speed += 1.0
    traj = pd.DataFrame({"x": x, "y": np.zeros(10), "timestamp": ts})
    vel = checker._estimate_velocity(traj)
    assert 1.0 < vel <= 10.5, f"Expected velocity between 1-10.5 m/s, got {vel}"


def test_velocity_nan_handling():
    """Test that NaN values are handled."""
    checker = create_checker()
    traj = pd.DataFrame(
        {"x": [0, np.nan, 2, 3, 4], "y": [0, 0, 0, 0, 0], "timestamp": [0, 0.1, 0.2, 0.3, 0.4]}
    )
    vel = checker._estimate_velocity(traj)
    assert np.isfinite(vel)


def test_velocity_insufficient_points():
    """Test fallback when trajectory has too few points."""
    checker = create_checker()
    traj = pd.DataFrame({"x": [0, 1], "y": [0, 0], "timestamp": [0, 0.1]})
    vel = checker._estimate_velocity(traj)
    assert np.isfinite(vel)


def test_velocity_zero_time_diff():
    """Test that zero time differences are handled."""
    checker = create_checker()
    traj = pd.DataFrame(
        {
            "x": [0, 1, 2, 3, 4],
            "y": [0, 0, 0, 0, 0],
            "timestamp": [0, 0, 0.1, 0.2, 0.3],  # duplicate timestamp
        }
    )
    vel = checker._estimate_velocity(traj)
    assert np.isfinite(vel)
    assert vel > 0


def test_velocity_frame_fallback():
    """Test fallback to frame/30 when timestamp not available."""
    checker = create_checker()
    traj = pd.DataFrame(
        {
            "x": np.arange(10) * 0.5,
            "y": np.zeros(10),
            "frame": np.arange(10),  # 30 fps
        }
    )
    vel = checker._estimate_velocity(traj)
    # 0.5 m per frame at 30fps = 15 m/s
    assert abs(vel - 15.0) < 0.1, f"Expected ~15 m/s, got {vel}"
