
import pandas as pd
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
    assert df["site"].notna().all()
    assert set(df["site"].str.upper()) <= {"GITI", "MRC"}
    counts = df["site"].str.upper().value_counts().to_dict()
    assert len(counts) >= 1
