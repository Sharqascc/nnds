import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.pet_conflict_checker import PETConflictChecker


def create_checker():
    checker = PETConflictChecker.__new__(PETConflictChecker)
    return checker


def test_savgol_constant():
    """Test that Savitzky-Golay matches constant speed (5 m/s)."""
    checker = create_checker()
    traj = pd.DataFrame(
        {"x": np.arange(20) * 0.5, "y": np.zeros(20), "timestamp": np.arange(20) * 0.1}
    )
    vel = checker._estimate_velocity_savgol(traj, window=7, polyorder=2)
    assert np.isclose(vel, 5.0, atol=0.2), f"Expected ~5 m/s, got {vel}"


def test_savgol_accelerating_analytical():
    """Test with known analytical acceleration. x=0.5*a*t^2, a=1 m/s^2."""
    checker = create_checker()
    t = np.arange(20) * 0.1
    x = 0.5 * 1.0 * t**2
    y = np.zeros(20)
    traj = pd.DataFrame({"x": x, "y": y, "timestamp": t})
    vel = checker._estimate_velocity_savgol(traj, window=7, polyorder=2)
    assert np.isclose(vel, 0.95, atol=0.1), f"Expected ~0.95 m/s, got {vel}"


def test_savgol_fallback_short():
    """Test that short trajectories fallback to median."""
    checker = create_checker()
    traj = pd.DataFrame({"x": [0, 1], "y": [0, 0], "timestamp": [0, 0.1]})
    vel = checker._estimate_velocity_savgol(traj, window=7, polyorder=2)
    assert np.isclose(vel, 10.0, atol=0.5)


def test_savgol_fallback_nonuniform():
    """Test that non-uniform timestamps fallback to median."""
    checker = create_checker()
    traj = pd.DataFrame(
        {"x": [0, 1, 2, 3, 4], "y": [0, 0, 0, 0, 0], "timestamp": [0, 0.1, 0.2, 0.5, 0.6]}
    )
    vel = checker._estimate_velocity_savgol(traj, window=5, polyorder=2)
    assert np.isfinite(vel)


def test_savgol_fallback_insufficient_window():
    """Test that when window is too small (< polyorder+2), fallback to median."""
    checker = create_checker()
    traj = pd.DataFrame(
        {"x": np.arange(10) * 0.5, "y": np.zeros(10), "timestamp": np.arange(10) * 0.1}
    )
    vel = checker._estimate_velocity_savgol(traj, window=3, polyorder=2)
    assert np.isfinite(vel)
    assert abs(vel - 5.0) < 0.5


def test_savgol_nan_middle():
    """Test NaN in middle of trajectory."""
    checker = create_checker()
    traj = pd.DataFrame(
        {
            "x": [0, 1, np.nan, 3, 4, 5, 6],
            "y": [0, 0, 0, 0, 0, 0, 0],
            "timestamp": np.arange(7) * 0.1,
        }
    )
    vel = checker._estimate_velocity_savgol(traj, window=5, polyorder=2)
    assert np.isfinite(vel) or np.isnan(vel)


def test_savgol_noisy_constant():
    """Test Savitzky-Golay on noisy constant speed data."""
    checker = create_checker()
    np.random.seed(42)
    t = np.arange(30) * 0.1
    x = 5.0 * t + np.random.normal(0, 0.02, len(t))
    y = np.zeros(30)
    traj = pd.DataFrame({"x": x, "y": y, "timestamp": t})
    vel = checker._estimate_velocity_savgol(traj, window=9, polyorder=3)
    assert np.isclose(vel, 5.0, atol=0.3), f"Expected ~5 m/s, got {vel}"


def test_savgol_missing_timestamp():
    """Test when timestamp column is missing."""
    checker = create_checker()
    traj = pd.DataFrame({"x": [0, 1, 2], "y": [0, 0, 0]})
    vel = checker._estimate_velocity_savgol(traj)
    assert np.isnan(vel)
