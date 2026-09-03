
import logging
import importlib
import types
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.analysis.pet_conflict_checker import (
    ConflictSeverity,
    PETUncertainty,
    ConflictResult,
    _find_pet_column,
    classify_pet_severity,
    setup_conflict_logger,
    compute_pet,
    compute_pet_batch,
    compute_grid_pet,
    estimate_pet_uncertainty,
    filter_by_roi,
    get_trajectory_pairs,
    detect_conflicts,
    _row_to_trajectory,
    dataframe_to_pet_events,
    PETConflictChecker,
    Trajectory,
    WorldPoint,
    PETEvent,
)


# ---------- _find_pet_column ----------
def test_find_pet_column_prefers_first_candidate():
    df = pd.DataFrame(columns=["foo", "pet_sec", "pet"])
    assert _find_pet_column(df) == "pet"  # first candidate 'pet' is checked first

def test_find_pet_column_fallback():
    df = pd.DataFrame(columns=["pet_sample_sec"])
    assert _find_pet_column(df) == "pet_sample_sec"

def test_find_pet_column_none():
    df = pd.DataFrame(columns=["x", "y"])
    assert _find_pet_column(df) is None


# ---------- classify_pet_severity ----------
@pytest.mark.parametrize("val,expected", [
    (0.5, ConflictSeverity.CRITICAL),
    (1.2, ConflictSeverity.SERIOUS),
    (2.0, ConflictSeverity.MODERATE),
    (4.0, ConflictSeverity.MINOR),
    (6.0, ConflictSeverity.SAFE),
])
def test_classify_pet_severity(val, expected):
    assert classify_pet_severity(val) == expected


# ---------- setup_conflict_logger ----------
def test_setup_logger_duplicate_handlers(tmp_path):
    logger = setup_conflict_logger(log_dir=str(tmp_path), logger_name="testlogger")
    logger2 = setup_conflict_logger(log_dir=str(tmp_path), logger_name="testlogger")
    assert logger.handlers == logger2.handlers
    # Clean up handlers to avoid side effects
    logging.getLogger("testlogger").handlers.clear()


# ---------- compute_pet ----------
def test_compute_pet_empty():
    assert compute_pet([], []) == np.inf
    assert compute_pet([1.0], []) == np.inf

def test_compute_pet_nan_raises():
    with pytest.raises(ValueError):
        compute_pet([np.nan], [1.0])

def test_compute_pet_nonmonotonic_warns():
    with pytest.warns(RuntimeWarning):
        compute_pet([1.0, 0.5], [2.0, 3.0])

def test_compute_pet_near_zero_warns():
    with pytest.warns(RuntimeWarning):
        compute_pet([0.001], [0.002], min_valid_pet=0.01)

def test_compute_pet_basic():
    assert compute_pet([1.0, 2.0], [2.5, 3.5]) == 0.5


# ---------- compute_pet_batch ----------
def test_compute_pet_batch_length_mismatch():
    with pytest.raises(ValueError):
        compute_pet_batch([np.array([1.0])], [])

def test_compute_pet_batch_success():
    a = [np.array([1.0, 2.0]), np.array([5.0])]
    b = [np.array([2.5, 3.0]), np.array([6.0])]
    pets = compute_pet_batch(a, b)
    assert pets[0] == 0.5
    assert pets[1] == 1.0


# ---------- compute_grid_pet ----------
def test_compute_grid_pet_shape_mismatch():
    with pytest.raises(ValueError):
        compute_grid_pet(np.ones((3,4,4)), np.ones((4,4,4)), fps=30)

def test_compute_grid_pet_empty_occupancy():
    grid = np.zeros((3,4,4))
    assert compute_grid_pet(grid, grid, fps=30) == np.inf

def test_compute_grid_pet_positive():
    grid_a = np.zeros((5,4,4))
    grid_b = np.zeros((5,4,4))
    grid_a[1] = 1
    grid_b[3] = 1
    assert compute_grid_pet(grid_a, grid_b, fps=10) == 2/10


# ---------- estimate_pet_uncertainty ----------
def test_estimate_pet_uncertainty_zero_velocity():
    unc = estimate_pet_uncertainty(2.0, velocity_mps=0.0)
    assert unc.uncertainty_std > 0
    assert set(unc.error_sources.keys()) == {"detection", "homography", "tracking"}

def test_estimate_pet_uncertainty_normal():
    unc = estimate_pet_uncertainty(2.5, velocity_mps=10.0)
    assert unc.nominal_pet == 2.5
    assert unc.uncertainty_std == pytest.approx(np.sqrt(0.2**2+0.3**2+0.1**2)/10.0)


# ---------- filter_by_roi ----------
def test_filter_by_roi_missing_keys():
    with pytest.raises(ValueError):
        filter_by_roi(pd.DataFrame({"x":[1], "y":[1]}), {"xmin":0})

def test_filter_by_roi_missing_columns():
    df = pd.DataFrame({"x":[1], "y":[1]})
    with pytest.raises(ValueError):
        filter_by_roi(df, {"xmin":0,"xmax":1,"ymin":0,"ymax":1}, x_col="z", y_col="w")

def test_filter_by_roi_success():
    df = pd.DataFrame({"x":[1,5,10], "y":[1,5,10]})
    roi = {"xmin":0, "xmax":6, "ymin":0, "ymax":6}
    out = filter_by_roi(df, roi)
    assert len(out) == 2


# ---------- get_trajectory_pairs ----------
def test_get_trajectory_pairs():
    df = pd.DataFrame({"track_id":[1,2,1,2,3], "frame":[0,0,1,1,1]})
    pairs = get_trajectory_pairs(df)
    assert (1,2) in pairs
    assert (1,3) in pairs
    assert (2,3) in pairs


# ---------- detect_conflicts ----------
def test_detect_conflicts_no_pet_col():
    df = pd.DataFrame({"x":[1,2]})
    with pytest.warns(RuntimeWarning):
        out = detect_conflicts(df)
    assert "is_conflict" in out.columns

def test_detect_conflicts_with_velocity():
    df = pd.DataFrame({"pet":[0.5, 2.0, 4.0], "velocity":[10, 5, 2]})
    out = detect_conflicts(df, pet_threshold=3.0, velocity_col="velocity")
    assert len(out) == 2
    assert "severity" in out.columns
    assert "pet_uncertainty_std" in out.columns

def test_detect_conflicts_no_velocity():
    df = pd.DataFrame({"pet":[0.5, 2.0, 4.0]})
    out = detect_conflicts(df, pet_threshold=3.0)
    assert len(out) == 2
    assert out["pet_uncertainty_std"].iloc[0] == pytest.approx(np.sqrt(0.2**2+0.3**2+0.1**2)/5.0)


# ---------- _row_to_trajectory ----------
def test_row_to_trajectory_none():
    traj = _row_to_trajectory(None, track_id=1)
    assert traj.track_id == 1
    assert len(traj.points) == 0

def test_row_to_trajectory_short_entry():
    traj = _row_to_trajectory([[0,1]], track_id=2)  # len < 3 skip
    assert len(traj.points) == 0

def test_row_to_trajectory_valid():
    traj = _row_to_trajectory([[0,1,2],[1,3,4]], track_id=3)
    assert len(traj.points) == 2


# ---------- dataframe_to_pet_events ----------
def test_dataframe_to_pet_events_missing_col():
    df = pd.DataFrame({"x":[1]})
    with pytest.raises(ValueError):
        dataframe_to_pet_events(df)

def test_dataframe_to_pet_events_success():
    df = pd.DataFrame({
        "pet":[0.5],
        "event_id":[1],
        "track_a":[10],
        "track_b":[20],
        "conflict_type":["crossing"],
        "frame":[5],
        "world_traj_i":[[[0,1,2]]],
        "world_traj_j":[[[0,3,4]]],
        "extra_col":[42],
    })
    events = dataframe_to_pet_events(df)
    assert len(events) == 1
    assert events[0].pet == 0.5
    assert events[0].frame == 5
    assert events[0].metadata["extra_col"] == 42


# ---------- PETConflictChecker methods ----------
def test_checker_init_invalid_threshold():
    with pytest.raises(ValueError):
        PETConflictChecker(pet_threshold=-1.0)

def test_checker_init_logging(tmp_path, monkeypatch):
    # Avoid actual file logging side effects
    checker = PETConflictChecker(enable_logging=True, log_dir=str(tmp_path))
    assert checker.logger is not None

def test_checker_detect_from_csv(tmp_path, capsys):
    csv = tmp_path / "pet.csv"
    csv.write_text("pet,velocity\n0.5,10\n4.0,2\n")
    checker = PETConflictChecker(enable_logging=False, enable_uncertainty=True)
    out = checker.detect_from_csv(str(csv), velocity_col="velocity")
    assert len(out) == 1
    assert out.iloc[0]["pet"] == 0.5

def test_checker_detect_from_csv_as_events(tmp_path):
    csv = tmp_path / "pet.csv"
    csv.write_text("""pet,event_id,track_a,track_b,conflict_type,frame,world_traj_i,world_traj_j
0.5,1,1,2,crossing,5,"[[0,1,2]]","[[0,3,4]]"
""")
    checker = PETConflictChecker(enable_logging=False, enable_uncertainty=False)
    events = checker.detect_from_csv_as_events(str(csv))
    assert len(events) == 1
    assert events[0].pet == 0.5

def test_checker_detect_from_trajectories_batch():
    checker = PETConflictChecker(enable_logging=False, enable_uncertainty=True)
    traj_a = pd.DataFrame({"track_id":[1,1], "frame":[0,1], "timestamp":[0.0, 0.1], "x":[0,1], "y":[0,1]})
    traj_b = pd.DataFrame({"track_id":[2,2], "frame":[0,1], "timestamp":[0.2, 0.3], "x":[2,3], "y":[2,3]})
    pairs = [(traj_a, traj_b)]
    results = checker.detect_from_trajectories_batch(pairs, fps=10)
    assert len(results) == 1
    assert results[0].pet == 0.1

def test_checker_estimate_velocity_basic():
    checker = PETConflictChecker(enable_logging=False)
    traj = pd.DataFrame({"x":[0,1,2], "y":[0,1,2], "timestamp":[0,1,2]})
    assert checker._estimate_velocity(traj) > 0

def test_checker_estimate_velocity_savgol_fallback():
    checker = PETConflictChecker(enable_logging=False)
    traj = pd.DataFrame({"x":[0,1], "y":[0,1], "timestamp":[0,1]})
    # Should fall back to _estimate_velocity because len < 3
    assert checker._estimate_velocity_savgol(traj) > 0

def test_checker_extract_trajectories():
    checker = PETConflictChecker(enable_logging=False)
    df = pd.DataFrame({"track_id":[1,1,2], "frame":[1,0,1], "x":[1,2,3], "y":[1,2,3]})
    trajs = checker.extract_trajectories(df)
    assert set(trajs.keys()) == {1,2}
    assert len(trajs[1]) == 2

def test_checker_process_video_stub():
    checker = PETConflictChecker(enable_logging=False)
    out = checker.process_video("dummy.mp4", "weights.pt")
    assert out.empty


# ---------- Additional coverage tests ----------

def test_fallback_import_block():
    """Cover fallback class definitions when core types cannot be imported."""
    import src.analysis.pet_conflict_checker as pcc
    original_core = sys.modules.get('src.core.types')
    fake_core = types.ModuleType('src.core.types')
    with patch.dict(sys.modules, {'src.core.types': fake_core}):
        importlib.reload(pcc)
        assert hasattr(pcc, 'WorldPoint')
        assert hasattr(pcc, 'Trajectory')
        assert hasattr(pcc, 'PETEvent')
    importlib.reload(pcc)
    if original_core is not None:
        assert sys.modules['src.core.types'] is original_core


def test_pet_uncertainty_properties():
    unc = PETUncertainty(2.0, 0.1, {})
    ci = unc.confidence_interval_95
    assert ci[0] == max(0.0, 2.0 - 1.96 * 0.1)
    assert ci[1] == 2.0 + 1.96 * 0.1
    rel = unc.relative_error_percent
    assert rel == pytest.approx((0.1 / 2.0) * 100.0)
    unc_zero = PETUncertainty(0.0, 0.1, {})
    assert unc_zero.relative_error_percent == np.inf


def test_conflict_result_to_dict():
    unc = PETUncertainty(1.0, 0.05, {})
    cr = ConflictResult(1, 2, 1.0, ConflictSeverity.MODERATE, unc, frame_start=0, frame_end=10, extra={'site': 'X'})
    d = cr.to_dict()
    assert d['id_a'] == 1
    assert 'pet_uncertainty_std' in d
    assert d['site'] == 'X'


def test_compute_pet_batch_exception():
    a = [np.array([1.0]), np.array([np.nan])]
    b = [np.array([2.0]), np.array([3.0])]
    with pytest.warns(RuntimeWarning):
        pets = compute_pet_batch(a, b)
    assert pets[0] == 1.0
    assert np.isinf(pets[1])


def test_detect_from_csv_with_logger(tmp_path):
    csv = tmp_path / "pet.csv"
    csv.write_text("pet,velocity\n0.5,10\n4.0,2\n")
    checker = PETConflictChecker(enable_logging=True, log_dir=str(tmp_path))
    out = checker.detect_from_csv(str(csv), velocity_col="velocity")
    # Clean up logger handlers to avoid side effects
    logging.getLogger("pet_conflict_checker").handlers.clear()
    assert len(out) == 1


def test_estimate_velocity_missing_columns():
    checker = PETConflictChecker(enable_logging=False)
    traj = pd.DataFrame({"frame":[0,1], "timestamp":[0,1]})
    assert checker._estimate_velocity(traj) == 5.0


def test_estimate_velocity_savgol_missing_cols():
    checker = PETConflictChecker(enable_logging=False)
    traj = pd.DataFrame({"x":[0,1], "y":[0,1]})  # missing timestamp
    assert np.isnan(checker._estimate_velocity_savgol(traj))


def test_estimate_velocity_savgol_short():
    checker = PETConflictChecker(enable_logging=False)
    traj = pd.DataFrame({"x":[0], "y":[0], "timestamp":[0]})
    # len < 3 -> fallback to _estimate_velocity
    assert checker._estimate_velocity_savgol(traj) == 5.0


def test_estimate_velocity_savgol_nonmonotonic():
    checker = PETConflictChecker(enable_logging=False)
    traj = pd.DataFrame({"x":[0,1,2], "y":[0,1,2], "timestamp":[1,1,2]})
    # non-monotonic -> fallback
    assert checker._estimate_velocity_savgol(traj) > 0


def test_estimate_velocity_savgol_nonuniform():
    checker = PETConflictChecker(enable_logging=False)
    traj = pd.DataFrame({"x":[0,1,2], "y":[0,1,2], "timestamp":[0,1,2.5]})
    # non-uniform -> fallback
    assert checker._estimate_velocity_savgol(traj) > 0


def test_estimate_velocity_savgol_valid():
    checker = PETConflictChecker(enable_logging=False)
    traj = pd.DataFrame({"x":[0,1,2,3,4], "y":[0,1,2,3,4], "timestamp":[0,1,2,3,4]})
    vel = checker._estimate_velocity_savgol(traj, window=5, polyorder=2)
    assert vel > 0


def test_checker_wrappers_get_pairs_and_filter():
    checker = PETConflictChecker(enable_logging=False)
    df = pd.DataFrame({"track_id":[1,2,1,2,3], "frame":[0,0,1,1,1]})
    pairs = checker.get_trajectory_pairs(df)
    assert (1,2) in pairs
    # filter_by_roi wrapper
    df_roi = pd.DataFrame({"x":[1,5,10], "y":[1,5,10]})
    roi = {"xmin":0, "xmax":6, "ymin":0, "ymax":6}
    out = checker.filter_by_roi(df_roi, roi)
    assert len(out) == 2


# ---------- Additional coverage for missing branches ----------

def test_compute_pet_nonmonotonic_both_warn():
    with pytest.warns(RuntimeWarning):
        compute_pet([1.0, 0.5], [2.0, 1.0])


def test_detect_from_trajectories_batch_logger_and_skip_and_exception():
    checker = PETConflictChecker(enable_logging=True, enable_uncertainty=True, pet_threshold=2.0)
    checker.logger = MagicMock()
    traj_a = pd.DataFrame({"track_id":[1,1], "frame":[0,1], "timestamp":[0.0, 0.1], "x":[0,1], "y":[0,1]})
    traj_b = pd.DataFrame({"track_id":[2,2], "frame":[0,1], "timestamp":[0.2, 0.3], "x":[2,3], "y":[2,3]})
    pairs = [(traj_a, traj_b), (traj_a, traj_b)]
    with patch('src.analysis.pet_conflict_checker.compute_pet', side_effect=[3.0, Exception("boom")]):
        results = checker.detect_from_trajectories_batch(pairs, fps=10)
    assert len(results) == 0
    assert checker.logger.info.called
    assert checker.logger.warning.called


def test_estimate_velocity_all_invalid_speeds():
    checker = PETConflictChecker(enable_logging=False)
    traj = pd.DataFrame({
        "x": [0.0, np.nan, 2.0],
        "y": [0.0, np.nan, 2.0],
        "timestamp": [0.0, 1.0, 2.0]
    })
    assert checker._estimate_velocity(traj) == 5.0


def test_estimate_velocity_savgol_window_adjust_and_fallback():
    checker = PETConflictChecker(enable_logging=False)
    # len=3, window=2 (even -> adjusted to 1), polyorder=2 triggers fallback
    traj = pd.DataFrame({
        "x": [0.0, 1.0, 2.0],
        "y": [0.0, 1.0, 2.0],
        "timestamp": [0.0, 1.0, 2.0]
    })
    assert checker._estimate_velocity_savgol(traj, window=2, polyorder=2) == checker._estimate_velocity(traj)


def test_estimate_velocity_savgol_return_nan_for_invalid_speeds():
    checker = PETConflictChecker(enable_logging=False)
    traj = pd.DataFrame({
        "x": [0.0, 1.0, 2.0, 3.0, 4.0],
        "y": [0.0, 1.0, 2.0, 3.0, 4.0],
        "timestamp": [0.0, 1.0, 2.0, 3.0, 4.0]
    })
    with patch('src.analysis.pet_conflict_checker.savgol_filter', side_effect=[np.array([np.nan]*5), np.array([np.nan]*5)]):
        assert np.isnan(checker._estimate_velocity_savgol(traj, window=5, polyorder=2))


def test_process_video_stub_with_logger():
    checker = PETConflictChecker(enable_logging=True)
    checker.logger = MagicMock()
    out = checker.process_video("dummy.mp4", "weights.pt")
    assert out.empty
    assert checker.logger.warning.called
