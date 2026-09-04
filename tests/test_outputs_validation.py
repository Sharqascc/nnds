
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

@pytest.mark.skipif(
    not (REPO / "outputs" / "combined_screened_simplified.csv").exists(),
    reason="Simplified combined PET outputs not generated",
)
def test_combined_simplified_site_labels_valid():
    df = pd.read_csv(REPO / "outputs" / "combined_screened_simplified.csv")
    assert "site" in df.columns
    assert "site.1" not in df.columns, "Duplicate site column present"
    assert df["site"].notna().all()
    assert set(df["site"].str.upper()) <= {"GITI", "MRC"}
    counts = df["site"].str.upper().value_counts().to_dict()
    assert len(counts) >= 1

@pytest.mark.skipif(
    not (REPO / "outputs" / "combined_screened_simplified.csv").exists(),
    reason="Simplified combined PET outputs not generated",
)
def test_combined_simplified_has_provenance_fields():
    df = pd.read_csv(REPO / "outputs" / "combined_screened_simplified.csv")
    for col in ["run_id", "pipeline_version", "git_commit", "config_hash", "fps",
                "pet_method", "conflict_zone_coordinate_system", "conflict_zone_half_size_m"]:
        assert col in df.columns, f"Missing provenance column: {col}"
        assert df[col].notna().all(), f"Column {col} has NaN values"
    assert (df["config_hash"] != "not_computed").all(), "config_hash not computed"

@pytest.mark.skipif(
    not (REPO / "outputs" / "combined_screened_simplified.csv").exists(),
    reason="Simplified combined PET outputs not generated",
)
def test_direction_fields_recompute_pet():
    df = pd.read_csv(REPO / "outputs" / "combined_screened_simplified.csv")
    fps = df["fps"].iloc[0]
    recomputed_pet_frames = df["second_entry_frame"] - df["first_exit_frame"]
    recomputed_pet = recomputed_pet_frames / fps
    # Tolerance for frame precision
    assert np.allclose(recomputed_pet, df["pet"], atol=0.001)
    assert np.allclose(recomputed_pet, df["pet_s"], atol=0.001)

@pytest.mark.skipif(
    not (REPO / "outputs" / "combined_screened_simplified.csv").exists(),
    reason="Simplified combined PET outputs not generated",
)
def test_gate_and_time_fields_clean():
    df = pd.read_csv(REPO / "outputs" / "combined_screened_simplified.csv")
    assert (df["time_of_day_label"] != "nan").all()
    assert (df["gate_a_entry"] != "unknown").all()
    assert (df["gate_b_entry"] != "unknown").all()
