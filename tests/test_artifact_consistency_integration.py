"""Integration: committed CSV and JSON artifacts agree.

No heavy assets. Runs in PR CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs"


def _skip_if_missing(paths):
    for p in paths:
        if not Path(p).exists():
            pytest.skip(f"artifact missing: {p}")


def test_screened_counts_match_summary_json():
    _skip_if_missing([
        OUT / "giti_screened_with_gates.csv",
        OUT / "mrc_screened_with_gates.csv",
        OUT / "final_screened_summary.json",
    ])
    giti = pd.read_csv(OUT / "giti_screened_with_gates.csv")
    mrc = pd.read_csv(OUT / "mrc_screened_with_gates.csv")
    summary = json.loads((OUT / "final_screened_summary.json").read_text())

    assert len(giti) == summary["GITI"]["screened_events"]
    assert len(mrc) == summary["MRC"]["screened_events"]


def test_raw_counts_match_summary_json():
    _skip_if_missing([
        OUT / "giti_raw.csv",
        OUT / "mrc_raw.csv",
        OUT / "final_screened_summary.json",
    ])
    giti = pd.read_csv(OUT / "giti_raw.csv")
    mrc = pd.read_csv(OUT / "mrc_raw.csv")
    summary = json.loads((OUT / "final_screened_summary.json").read_text())
    assert len(giti) == summary["GITI"]["raw_events"]
    assert len(mrc) == summary["MRC"]["raw_events"]


def test_screened_csvs_are_subsets_of_raw():
    _skip_if_missing([OUT / "giti_raw.csv", OUT / "giti_screened.csv"])
    raw = pd.read_csv(OUT / "giti_raw.csv")
    screened = pd.read_csv(OUT / "giti_screened.csv")
    # Every screened (track_a, track_b, pet) pair appears in raw
    raw_keys = set(zip(raw.track_a, raw.track_b, raw.pet.round(4), strict=True))
    screened_keys = set(zip(screened.track_a, screened.track_b, screened.pet.round(4), strict=True))
    assert screened_keys.issubset(raw_keys)


def test_combined_events_equal_sum_of_sites():
    _skip_if_missing([
        OUT / "giti_screened_with_gates.csv",
        OUT / "mrc_screened_with_gates.csv",
        OUT / "combined_screened_simplified.csv",
    ])
    giti = pd.read_csv(OUT / "giti_screened_with_gates.csv")
    mrc = pd.read_csv(OUT / "mrc_screened_with_gates.csv")
    combined = pd.read_csv(OUT / "combined_screened_simplified.csv")
    assert len(combined) == len(giti) + len(mrc)


def test_pet_values_are_positive_in_screened():
    _skip_if_missing([OUT / "giti_screened_with_gates.csv"])
    giti = pd.read_csv(OUT / "giti_screened_with_gates.csv")
    assert (giti.pet > 0).all()


def test_pet_screened_max_below_three_seconds():
    """Frozen artifacts use a 3.0 s PET threshold."""
    _skip_if_missing([OUT / "giti_screened_with_gates.csv"])
    giti = pd.read_csv(OUT / "giti_screened_with_gates.csv")
    assert giti.pet.max() <= 3.0 + 1e-6
