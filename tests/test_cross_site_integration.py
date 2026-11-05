"""Integration: GITI and MRC sites aggregate consistently.

No heavy assets. Runs in PR CI.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs"


def _skip_if_missing(paths):
    for p in paths:
        if not Path(p).exists():
            pytest.skip(f"artifact missing: {p}")


def test_combined_row_count_is_187():
    _skip_if_missing([OUT / "combined_screened_simplified.csv"])
    combined = pd.read_csv(OUT / "combined_screened_simplified.csv")
    assert len(combined) == 187


def test_sites_are_giti_and_mrc_only():
    _skip_if_missing([OUT / "combined_screened_simplified.csv"])
    combined = pd.read_csv(OUT / "combined_screened_simplified.csv")
    if "site" in combined.columns:
        assert set(combined.site.unique()).issubset({"GITI", "MRC"})


def test_giti_has_more_events_than_mrc():
    _skip_if_missing(
        [
            OUT / "giti_screened_with_gates.csv",
            OUT / "mrc_screened_with_gates.csv",
        ]
    )
    giti = pd.read_csv(OUT / "giti_screened_with_gates.csv")
    mrc = pd.read_csv(OUT / "mrc_screened_with_gates.csv")
    assert len(giti) > len(mrc)


def test_gate_labels_present_per_site():
    _skip_if_missing([OUT / "giti_screened_with_gates.csv"])
    giti = pd.read_csv(OUT / "giti_screened_with_gates.csv")
    # Gate columns must be populated
    assert giti.gate_a_entry.notna().any()
    assert giti.gate_b_entry.notna().any()
