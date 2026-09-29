from types import SimpleNamespace

import numpy as np
import pytest

from src.core.types import (
    Trajectory,
    TrajectoryBatch,
    WorldPoint,
)
from src.core.validation import (
    compute_error_metrics,
    validate_bev_result,
    validate_numeric_array,
)
from src.utils.seed import get_seed, set_seed


# ---------- core.types ----------
def test_world_point_creation():
    wp = WorldPoint(t=1.0, x=2.0, y=3.0)
    assert wp.t == 1.0
    assert wp.x == 2.0
    assert wp.y == 3.0


def test_trajectory_duration_empty():
    traj = Trajectory(track_id=1, points=())
    assert traj.duration == 0.0


def test_trajectory_duration_nonempty():
    wp1 = WorldPoint(t=0.0, x=0, y=0)
    wp2 = WorldPoint(t=2.5, x=1, y=1)
    traj = Trajectory(track_id=1, points=(wp1, wp2))
    assert traj.duration == 2.5


def test_trajectory_batch_properties():
    fake_inputs = SimpleNamespace(shape=(4, 10, 2))
    fake_targets = SimpleNamespace(shape=(4, 5, 2))
    batch = TrajectoryBatch(inputs=fake_inputs, targets=fake_targets, meta={}, fps=30.0)
    assert batch.batch_size == 4
    assert batch.input_length == 10
    assert batch.target_length == 5


# ---------- core.validation ----------
def test_compute_error_metrics_empty_raises():
    with pytest.raises(ValueError):
        compute_error_metrics([])


def test_compute_error_metrics_valid():
    metrics = compute_error_metrics([1.0, 2.0, 3.0])
    assert metrics.mean_error == pytest.approx(2.0)
    assert metrics.max_error == 3.0
    assert metrics.rmse == pytest.approx(np.sqrt((1 + 4 + 9) / 3))
    assert metrics.num_samples == 3


def test_validate_numeric_array_empty():
    with pytest.raises(ValueError):
        validate_numeric_array("test", [])


def test_validate_numeric_array_nonfinite():
    with pytest.raises(ValueError):
        validate_numeric_array("test", [1.0, np.nan])


def test_validate_numeric_array_ndim_mismatch():
    with pytest.raises(ValueError):
        validate_numeric_array("test", [1.0, 2.0], ndim=2)


def test_validate_numeric_array_valid():
    arr = validate_numeric_array("test", [1.0, 2.0], ndim=1)
    assert arr.ndim == 1
    assert len(arr) == 2


def test_validate_bev_result_missing_keys():
    with pytest.raises(ValueError):
        validate_bev_result({"pointerrors": [], "meanerrorall": 1})


def test_validate_bev_result_valid():
    result = {
        "pointerrors": [{"error": 0.1}, {"error": 0.2}],
        "meanerrorall": 0.15,
        "meanerrorinliers": 0.1,
        "stderrorall": 0.05,
        "maxerror": 0.2,
        "rmse": 0.12,
    }
    # Should not raise
    validate_bev_result(result)


# ---------- utils.seed ----------
def test_set_seed_basic():
    set_seed(42)
    assert get_seed() == 42


def test_set_seed_cuda_available(monkeypatch):
    called = False

    def mock_manual_seed_all(seed):
        nonlocal called
        called = True

    # Simulate cuda available and patch manual_seed_all
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", mock_manual_seed_all)
    set_seed(123)
    assert called
