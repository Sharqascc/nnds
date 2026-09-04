import runpy
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.analysis.pet_conflict_checker import PETConflictChecker, compute_pet


def test_compute_pet_near_zero_warns_extra():
    with pytest.warns(RuntimeWarning):
        compute_pet([0.001], [0.002], min_valid_pet=0.01)


def test_batch_logger_and_skip(tmp_path):
    checker = PETConflictChecker(enable_logging=True, log_dir=str(tmp_path))
    import logging

    logging.getLogger("pet_conflict_checker").handlers.clear()
    traj_a = pd.DataFrame({"timestamp": [0.0], "frame": [0], "x": [0], "y": [0], "track_id": [1]})
    traj_b = pd.DataFrame(
        {"timestamp": [100.0], "frame": [100], "x": [10], "y": [10], "track_id": [2]}
    )
    with patch("src.analysis.pet_conflict_checker.compute_pet", return_value=10.0):
        results = checker.detect_from_trajectories_batch([(traj_a, traj_b)], fps=30.0)
    assert len(results) == 0
    logging.getLogger("pet_conflict_checker").handlers.clear()


def test_batch_exception(tmp_path):
    checker = PETConflictChecker(enable_logging=True, log_dir=str(tmp_path))
    import logging

    logging.getLogger("pet_conflict_checker").handlers.clear()
    traj_a = pd.DataFrame({"frame": [0], "x": [0], "y": [0]})
    traj_b = pd.DataFrame({"frame": [0], "x": [1], "y": [1]})
    with patch("src.analysis.pet_conflict_checker.compute_pet", side_effect=ValueError("boom")):
        results = checker.detect_from_trajectories_batch([(traj_a, traj_b)], fps=30.0)
    assert len(results) == 0
    logging.getLogger("pet_conflict_checker").handlers.clear()


def test_estimate_velocity_all_invalid():
    checker = PETConflictChecker(enable_logging=False)
    traj = pd.DataFrame({"x": [0, np.nan, 2], "y": [0, np.nan, 2], "timestamp": [0, 1, 2]})
    assert checker._estimate_velocity(traj) == 5.0


def test_estimate_velocity_savgol_even_window():
    checker = PETConflictChecker(enable_logging=False)
    traj = pd.DataFrame(
        {"x": [0, 1, 2, 3, 4, 5, 6], "y": [0, 1, 2, 3, 4, 5, 6], "timestamp": [0, 1, 2, 3, 4, 5, 6]}
    )
    vel = checker._estimate_velocity_savgol(traj, window=4, polyorder=3)
    assert vel > 0


def test_estimate_velocity_savgol_no_finite():
    checker = PETConflictChecker(enable_logging=False)
    traj = pd.DataFrame(
        {"x": [0, 1, 2, 3, 4, 5, 6], "y": [0, 1, 2, 3, 4, 5, 6], "timestamp": [0, 1, 2, 3, 4, 5, 6]}
    )
    with patch(
        "src.analysis.pet_conflict_checker.savgol_filter",
        return_value=np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
    ):
        vel = checker._estimate_velocity_savgol(traj, window=5, polyorder=2)
    assert np.isnan(vel)


def test_process_video_stub_logger(tmp_path):
    checker = PETConflictChecker(enable_logging=True, log_dir=str(tmp_path))
    import logging

    logging.getLogger("pet_conflict_checker").handlers.clear()
    out = checker.process_video("dummy.mp4", "weights.pt")
    assert out.empty
    logging.getLogger("pet_conflict_checker").handlers.clear()


def test_cli_main_guard(monkeypatch, tmp_path):
    csv = tmp_path / "events.csv"
    csv.write_text("pet\n0.5\n4.0\n")
    monkeypatch.setattr(sys, "argv", ["prog", "--csv", str(csv), "--no-uncertainty"])
    with (
        patch(
            "src.analysis.pet_conflict_checker.PETConflictChecker.detect_from_csv",
            return_value=pd.DataFrame(columns=["pet"]),
        ),
        pytest.raises(SystemExit) as e,
    ):
        runpy.run_module("src.analysis.pet_conflict_checker", run_name="__main__")
    assert e.value.code == 0


def test_compute_pet_times_b_nonmonotonic_warns():
    with pytest.warns(RuntimeWarning):
        compute_pet([1.0, 2.0], [3.0, 2.5])


def test_cli_main_block(monkeypatch, tmp_path):
    csv = tmp_path / "events.csv"
    csv.write_text("pet\n0.5\n4.0\n")
    monkeypatch.setattr(sys, "argv", ["prog", "--csv", str(csv), "--no-uncertainty"])
    with (
        patch(
            "src.analysis.pet_conflict_checker.PETConflictChecker.detect_from_csv",
            return_value=pd.DataFrame(columns=["pet"]),
        ),
        pytest.raises(SystemExit) as e,
    ):
        runpy.run_module("src.analysis.pet_conflict_checker", run_name="__main__")
    assert e.value.code == 0
