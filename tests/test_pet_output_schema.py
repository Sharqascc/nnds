
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEMO_CSV = ROOT / "docs/data_samples/petevents_bev_demo.csv"

REQUIRED_COLUMNS = [
    "event_id", "pet", "pet_time_based", "frame", "track_a", "track_b",
    "orig_track_a", "seg_a", "orig_track_b", "seg_b",
    "conflict_type", "grid_cell", "track_a_entry_frame", "track_a_exit_frame", "track_a_exit_time_sec",
    "track_b_entry_frame", "track_b_entry_time_sec", "track_b_exit_frame", "world_traj_i", "world_traj_j",
    "traj_a_json", "traj_b_json",
]

def test_demo_csv_has_required_columns():
    assert DEMO_CSV.exists(), f"Missing {DEMO_CSV}"
    df = pd.read_csv(DEMO_CSV)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    assert not missing, f"Missing columns: {missing}"

def test_demo_csv_has_no_missing_ids_or_tracks():
    df = pd.read_csv(DEMO_CSV)
    assert not df["event_id"].isna().any(), "event_id contains NaN"
    assert not df["frame"].isna().any(), "frame contains NaN"
    assert not df["track_a"].isna().any(), "track_a contains NaN"
    assert not df["track_b"].isna().any(), "track_b contains NaN"

def test_demo_csv_conflict_type_allowed():
    df = pd.read_csv(DEMO_CSV)
    allowed = {"image_intersection", "UNKNOWN"}
    invalid = set(df["conflict_type"].dropna().unique()) - allowed
    assert not invalid, f"Invalid conflict_type values: {invalid}"

def test_demo_csv_world_traj_format():
    df = pd.read_csv(DEMO_CSV)
    for col in ["world_traj_i", "world_traj_j"]:
        non_null = df[col].dropna()
        assert all(isinstance(v, str) and v.startswith("track_") for v in non_null), \
            f"Invalid {col} values"
