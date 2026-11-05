"""Runtime test for paper.analysis.tracking_diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper" / "results" / "tracking_diagnostics.json"


@pytest.fixture(scope="module")
def result() -> dict:
    from paper.analysis import tracking_diagnostics

    tracking_diagnostics.main()
    assert OUT.exists(), "tracking_diagnostics did not write output"
    return json.loads(OUT.read_text())


def test_top_level_keys(result):
    for k in (
        "n_unique_tracks",
        "track_lengths",
        "gap_rate",
        "fragment_pairs",
        "heading_instability",
        "speed_plausibility",
        "robustness_dropout_10pct",
        "robustness_pixel_noise_1px",
    ):
        assert k in result, f"missing {k}"


def test_track_lengths(result):
    tl = result["track_lengths"]
    assert tl["n_tracks"] > 0
    assert tl["min"] >= 1
    assert tl["max"] >= tl["median"]


def test_gap_rate_in_range(result):
    gr = result["gap_rate"]
    assert 0.0 <= gr["mean"] <= 1.0
    assert 0.0 <= gr["max"] <= 1.0


def test_robustness_fractions(result):
    for key in ("robustness_dropout_10pct", "robustness_pixel_noise_1px"):
        r = result[key]
        assert 0.0 <= r["fraction_robust"] <= 1.0
        assert r["n_events"] >= 0
