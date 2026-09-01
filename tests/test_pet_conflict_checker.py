"""
Tests for PET conflict checker.
"""
import numpy as np
import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.pet_conflict_checker import (
    PETConflictChecker,
    PETEvent,
    ConflictResult,
    ConflictSeverity,
    PETUncertainty,
    Trajectory,
    WorldPoint
)

def create_test_trajectory(track_id=1):
    """Create a simple test trajectory."""
    points = (
        WorldPoint(t=0.0, x=0.0, y=0.0),
        WorldPoint(t=1/30, x=1.0, y=1.0),
        WorldPoint(t=2/30, x=2.0, y=2.0)
    )
    return Trajectory(track_id=track_id, points=points)

def test_pet_conflict_checker_import():
    """Test PETConflictChecker can be imported."""
    assert PETConflictChecker is not None

def test_pet_conflict_checker_initialization():
    """Test PETConflictChecker initializes."""
    checker = PETConflictChecker()
    assert checker is not None

def test_pet_conflict_checker_filter_by_roi():
    """Test filter_by_roi with correct ROI format."""
    checker = PETConflictChecker()
    df = pd.DataFrame({
        'x': [10.0, 20.0, 30.0, 40.0],
        'y': [10.0, 20.0, 30.0, 40.0]
    })
    roi = {'xmin': 0.0, 'xmax': 50.0, 'ymin': 0.0, 'ymax': 50.0}
    result = checker.filter_by_roi(df, roi)
    assert result is not None
    assert len(result) == 4

def test_pet_event_dataclass():
    """Test PETEvent dataclass."""
    traj = create_test_trajectory(track_id=1)
    event = PETEvent(
        event_id=1,
        pet=0.5,
        track_a=10,
        track_b=20,
        conflict_type='crossing',
        world_traj_i=traj,
        world_traj_j=traj
    )
    assert event.event_id == 1
    assert event.pet == 0.5

def test_conflict_result():
    """Test ConflictResult."""
    result = ConflictResult(
        id_a=10,
        id_b=20,
        pet=0.5,
        severity=ConflictSeverity.MODERATE
    )
    assert result.id_a == 10
    assert result.id_b == 20
    assert result.pet == 0.5

def test_pet_uncertainty():
    """Test PETUncertainty with correct fields."""
    uncertainty = PETUncertainty(
        nominal_pet=0.5,
        uncertainty_std=0.05,
        error_sources={'detection': 0.03, 'homography': 0.02}
    )
    assert uncertainty.nominal_pet == 0.5
    assert uncertainty.uncertainty_std == 0.05
    assert 'detection' in uncertainty.error_sources

def test_pet_uncertainty_ci95():
    """Test confidence interval calculation."""
    uncertainty = PETUncertainty(
        nominal_pet=0.5,
        uncertainty_std=0.05,
        error_sources={}
    )
    ci = uncertainty.confidence_interval_95
    assert ci[0] >= 0
    assert ci[1] > ci[0]

def test_trajectory_duration():
    """Test trajectory duration property."""
    traj = create_test_trajectory()
    assert traj.duration > 0
    assert traj.track_id == 1

def test_pet_conflict_checker_extract_trajectories():
    """Test trajectory extraction from DataFrame."""
    checker = PETConflictChecker()
    df = pd.DataFrame({
        'track_id': [1, 1, 1, 2, 2, 2],
        'frame': [0, 1, 2, 0, 1, 2],
        'x': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    })
    trajectories = checker.extract_trajectories(df)
    assert trajectories is not None
    assert len(trajectories) == 2
