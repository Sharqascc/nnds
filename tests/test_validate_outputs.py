
import sys
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from scripts.validate_outputs import (
    load_csv,
    validate_detections,
    validate_tracking_stability,
    validate_trajectory_json,
    validate_pet,
    main,
    DETECTION_COLUMNS,
    PET_COLUMNS,
)


# ---------------- load_csv ----------------

def test_load_csv_missing_file():
    with pytest.raises(SystemExit):
        load_csv("/nonexistent/file.csv", [], "Test")

def test_load_csv_missing_columns(tmp_path):
    p = tmp_path / "test.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(SystemExit):
        load_csv(str(p), ["x", "y"], "Test")

def test_load_csv_empty(tmp_path):
    p = tmp_path / "test.csv"
    p.write_text("x,y\n")
    df, empty = load_csv(str(p), ["x", "y"], "Test")
    assert empty is True
    assert df.empty

def test_load_csv_valid(tmp_path):
    p = tmp_path / "test.csv"
    p.write_text("x,y\n1,2\n")
    df, empty = load_csv(str(p), ["x", "y"], "Test")
    assert empty is False
    assert len(df) == 1


# ---------------- validate_detections ----------------

def det_df():
    return pd.DataFrame({
        "x1": [0, 10, 20],
        "y1": [0, 10, 20],
        "x2": [10, 20, 10],
        "y2": [10, 20, 10],
        "conf": [0.5, 0.9, 1.5],
        "class_name": ["car", "bus", "unknown"],
    })

def test_validate_detections_problems():
    df = det_df()
    problems = validate_detections(df)
    assert len(problems) >= 3  # bad box, conf out of range, unknown class

def test_validate_detections_clean():
    df = pd.DataFrame({
        "x1": [0],
        "y1": [0],
        "x2": [10],
        "y2": [10],
        "conf": [0.8],
        "class_name": ["car"],
    })
    assert validate_detections(df) == []


# ---------------- validate_tracking_stability ----------------

def test_validate_tracking_stability_empty():
    assert validate_tracking_stability(pd.DataFrame()) == []

def test_validate_tracking_stability_problems():
    df = pd.DataFrame({
        "track_id": [1, 1, 1],
        "frame": [0, 20, 21],
        "cx": [0, 100, 0],
        "cy": [0, 100, 0],
    })
    problems = validate_tracking_stability(df, max_gap=5, max_jump=50)
    assert len(problems) >= 1

def test_validate_tracking_stability_clean():
    df = pd.DataFrame({
        "track_id": [1, 1],
        "frame": [0, 1],
        "cx": [0, 1],
        "cy": [0, 1],
    })
    assert validate_tracking_stability(df) == []


# ---------------- validate_trajectory_json ----------------

def test_validate_trajectory_json_non_string():
    assert validate_trajectory_json(None, "T") != []

def test_validate_trajectory_json_empty_string():
    assert validate_trajectory_json("", "T") != []

def test_validate_trajectory_json_invalid_json():
    assert validate_trajectory_json("not json", "T") != []

def test_validate_trajectory_json_few_points():
    assert validate_trajectory_json("[{\"frame\":1}]", "T") != []

def test_validate_trajectory_json_missing_keys():
    pts = [{"frame":1, "x_pixel":1, "y_pixel":1, "world_x":1, "world_y":1},
           {"frame":2, "x_pixel":2, "y_pixel":2}]  # missing world_x/world_y
    js = json.dumps(pts)
    problems = validate_trajectory_json(js, "T")
    assert any("missing keys" in p for p in problems)

def test_validate_trajectory_json_all_world_none():
    pts = [{"frame":1, "x_pixel":1, "y_pixel":1, "world_x":None, "world_y":None},
           {"frame":2, "x_pixel":2, "y_pixel":2, "world_x":None, "world_y":None}]
    js = json.dumps(pts)
    problems = validate_trajectory_json(js, "T")
    assert any("world coordinates are None" in p for p in problems)

def test_validate_trajectory_json_valid():
    pts = [{"frame":1, "x_pixel":1, "y_pixel":1, "world_x":1, "world_y":1},
           {"frame":2, "x_pixel":2, "y_pixel":2, "world_x":2, "world_y":2}]
    js = json.dumps(pts)
    assert validate_trajectory_json(js, "T") == []


# ---------------- validate_pet ----------------

def make_pet_df():
    return pd.DataFrame({
        "event_id": [1],
        "pet": [1.0],
        "frame": [10],
        "track_a": [1],
        "track_b": [2],
        "track_a_entry_frame": [5],
        "track_a_exit_frame": [6],
        "track_b_entry_frame": [7],
        "track_b_exit_frame": [8],
        "grid_cell": ["G_A_1"],
        "traj_a_json": ['[{"frame":1,"x_pixel":1,"y_pixel":1,"world_x":1,"world_y":1},{"frame":2,"x_pixel":2,"y_pixel":2,"world_x":2,"world_y":2}]'],
        "traj_b_json": ['[{"frame":1,"x_pixel":1,"y_pixel":1,"world_x":1,"world_y":1},{"frame":2,"x_pixel":2,"y_pixel":2,"world_x":2,"world_y":2}]'],
    })

def test_validate_pet_empty():
    assert validate_pet(pd.DataFrame(), pd.DataFrame()) == []

def test_validate_pet_clean():
    pet = make_pet_df()
    det = pd.DataFrame({"track_id": [1,2]})
    assert validate_pet(pet, det) == []

def test_validate_pet_negative():
    pet = make_pet_df()
    pet["pet"] = -1
    problems = validate_pet(pet, pd.DataFrame())
    assert any("PET <= 0" in p for p in problems)

def test_validate_pet_frame_out_of_range():
    pet = make_pet_df()
    pet["frame"] = 2000
    problems = validate_pet(pet, pd.DataFrame(), video_frames=100)
    assert any("Frame out of range" in p for p in problems)

def test_validate_pet_out_of_bounds():
    pet = make_pet_df()
    pet["grid_cell"] = "OUT_OF_BOUNDS"
    problems = validate_pet(pet, pd.DataFrame())
    assert any("OUT_OF_BOUNDS" in p for p in problems)

def test_validate_pet_entry_exit():
    pet = make_pet_df()
    pet["track_a_entry_frame"] = 10
    pet["track_a_exit_frame"] = 5
    problems = validate_pet(pet, pd.DataFrame())
    assert any("entry > exit" in p for p in problems)

def test_validate_pet_missing_track_id():
    pet = make_pet_df()
    det = pd.DataFrame({"track_id": [999]})
    problems = validate_pet(pet, det)
    assert any("track IDs not in" in p for p in problems)

def test_validate_pet_invalid_traj():
    pet = make_pet_df()
    pet["traj_a_json"] = "invalid"
    problems = validate_pet(pet, pd.DataFrame())
    assert any("Traj A" in p for p in problems)


# ---------------- main ----------------

def _complete_pet_df():
    valid_traj = json.dumps([
        {"frame":1,"x_pixel":1,"y_pixel":1,"world_x":1.0,"world_y":1.0},
        {"frame":2,"x_pixel":2,"y_pixel":2,"world_x":2.0,"world_y":2.0},
    ])
    return pd.DataFrame({
        "event_id": [1],
        "pet": [1.0],
        "pet_time_based": [1.0],
        "frame": [10],
        "track_a": [1],
        "track_b": [2],
        "orig_track_a": [1],
        "seg_a": [0],
        "orig_track_b": [2],
        "seg_b": [0],
        "conflict_type": ["crossing"],
        "grid_cell": ["G_A_1"],
        "track_a_entry_frame": [5],
        "track_a_exit_frame": [6],
        "track_a_exit_time_sec": [0.2],
        "track_b_entry_frame": [7],
        "track_b_entry_time_sec": [0.23],
        "track_b_exit_frame": [8],
        "world_traj_i": [valid_traj],
        "world_traj_j": [valid_traj],
        "traj_a_json": [valid_traj],
        "traj_b_json": [valid_traj],
        "video_source": ["giti"],
        "time_of_day_label": ["morning"],
        "gate_a_entry": ["G1"],
        "gate_b_entry": ["G2"],
    })

def test_main_all_checks_pass(tmp_path):
    # Create valid detections and PET CSVs
    det = pd.DataFrame({
        "frame": [0,1,0,1],
        "track_id": [1,1,2,2],
        "class_id": [2,2,2,2],
        "class_name": ["car","car","car","car"],
        "conf": [0.9,0.9,0.9,0.9],
        "x1": [0,0,10,10],
        "y1": [0,0,10,10],
        "x2": [10,10,20,20],
        "y2": [10,10,20,20],
        "cx": [5,5,15,15],
        "cy": [5,5,15,15],
        "source": ["uvh","uvh","uvh","uvh"],
    })
    pet = _complete_pet_df()
    det_path = tmp_path / "detections.csv"
    pet_path = tmp_path / "pet.csv"
    split_path = tmp_path / "split_detections.csv"
    det.to_csv(det_path, index=False)
    pet.to_csv(pet_path, index=False)
    det.to_csv(split_path, index=False)

    test_args = [
        'prog',
        '--detections', str(det_path),
        '--detections-split', str(split_path),
        '--pet', str(pet_path),
        '--max-gap', '10',
        '--max-jump', '50',
    ]
    with patch.object(sys, 'argv', test_args):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0


def test_main_problems_found(tmp_path):
    # Create detections with an invalid box and PET with a negative pet
    det = pd.DataFrame({
        "frame": [0],
        "track_id": [1],
        "class_id": [2],
        "class_name": ["car"],
        "conf": [0.9],
        "x1": [10],
        "y1": [0],
        "x2": [0],
        "y2": [10],
        "cx": [5],
        "cy": [5],
        "source": ["uvh"],
    })
    pet = make_pet_df()
    pet["pet"] = -1
    det_path = tmp_path / "detections.csv"
    pet_path = tmp_path / "pet.csv"
    det.to_csv(det_path, index=False)
    pet.to_csv(pet_path, index=False)

    test_args = [
        'prog',
        '--detections', str(det_path),
        '--pet', str(pet_path),
        '--video-frames', '100',
    ]
    with patch.object(sys, 'argv', test_args):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1


# ---------------- Additional branch coverage ----------------

def test_validate_tracking_stability_single_point():
    df = pd.DataFrame({"track_id": [1], "frame": [0], "cx": [0], "cy": [0]})
    assert validate_tracking_stability(df) == []

def _write_empty_csv_with_columns(path, columns):
    pd.DataFrame(columns=columns).to_csv(path, index=False)


def test_main_detections_empty(tmp_path):
    det_path = tmp_path / "detections.csv"
    pet_path = tmp_path / "pet.csv"
    _write_empty_csv_with_columns(det_path, DETECTION_COLUMNS)
    _write_empty_csv_with_columns(pet_path, PET_COLUMNS)
    test_args = ['prog', '--detections', str(det_path), '--pet', str(pet_path)]
    with patch.object(sys, 'argv', test_args):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0  # empty detections skip checks, pet empty also skip


def test_main_split_empty(tmp_path):
    det = pd.DataFrame({
        "frame": [0,1,0,1], "track_id": [1,1,2,2], "class_id": [2,2,2,2], "class_name": ["car","car","car","car"],
        "conf": [0.9,0.9,0.9,0.9], "x1": [0,0,10,10], "y1": [0,0,10,10], "x2": [10,10,20,20], "y2": [10,10,20,20],
        "cx": [5,5,15,15], "cy": [5,5,15,15], "source": ["uvh","uvh","uvh","uvh"]
    })
    pet = _complete_pet_df()
    det_path = tmp_path / "detections.csv"
    pet_path = tmp_path / "pet.csv"
    split_path = tmp_path / "split_detections.csv"
    det.to_csv(det_path, index=False)
    pet.to_csv(pet_path, index=False)
    _write_empty_csv_with_columns(split_path, DETECTION_COLUMNS)
    test_args = ['prog', '--detections', str(det_path), '--detections-split', str(split_path), '--pet', str(pet_path)]
    with patch.object(sys, 'argv', test_args):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0


def test_main_pet_empty(tmp_path):
    det_path = tmp_path / "detections.csv"
    pet_path = tmp_path / "pet.csv"
    _write_empty_csv_with_columns(det_path, DETECTION_COLUMNS)
    _write_empty_csv_with_columns(pet_path, PET_COLUMNS)
    test_args = ['prog', '--detections', str(det_path), '--pet', str(pet_path)]
    with patch.object(sys, 'argv', test_args):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0


def test_main_if_name_main_block(monkeypatch):
    import runpy
    with patch.object(sys, 'argv', ['prog', '--help']):
        with pytest.raises(SystemExit):
            runpy.run_module('scripts.validate_outputs', run_name='__main__')


def test_validate_pet_loop_break_after_six_rows():
    # Ensure idx > 5 triggers the break lines 152 and 154
    base = make_pet_df()
    pet = pd.concat([base] * 7, ignore_index=True)
    pet["event_id"] = range(7)
    problems = validate_pet(pet, pd.DataFrame())
    assert isinstance(problems, list)


def test_main_tracking_problems_exit_1(tmp_path):
    det = pd.DataFrame({
        "frame": [0, 1, 0, 1],
        "track_id": [1, 1, 2, 2],
        "class_id": [2, 2, 2, 2],
        "class_name": ["car", "car", "car", "car"],
        "conf": [0.9, 0.9, 0.9, 0.9],
        "x1": [0, 0, 10, 10],
        "y1": [0, 0, 10, 10],
        "x2": [10, 10, 20, 20],
        "y2": [10, 10, 20, 20],
        "cx": [5, 100, 15, 15],   # big jump for track 1
        "cy": [5, 100, 15, 15],
        "source": ["uvh", "uvh", "uvh", "uvh"],
    })
    pet = _complete_pet_df()
    det_path = tmp_path / "detections.csv"
    split_path = tmp_path / "split_detections.csv"
    pet_path = tmp_path / "pet.csv"
    det.to_csv(det_path, index=False)
    det.to_csv(split_path, index=False)
    pet.to_csv(pet_path, index=False)

    test_args = [
        'prog',
        '--detections', str(det_path),
        '--detections-split', str(split_path),
        '--pet', str(pet_path),
        '--max-gap', '10',
        '--max-jump', '50',
    ]
    with patch.object(sys, 'argv', test_args):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
