"""Runtime test for paper.analysis.tracker_health."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper" / "results" / "tracker_health.json"


@pytest.fixture(scope="module")
def result() -> dict:
    from paper.analysis import tracker_health

    tracker_health.main()
    assert OUT.exists(), "tracker_health did not write output"
    return json.loads(OUT.read_text())


def test_top_level_keys(result):
    assert set(result.keys()) == {
        "track_lengths",
        "event_participation",
        "jump_stats_by_verdict",
    }


def test_track_lengths_shape(result):
    tl = result["track_lengths"]
    for k in (
        "n_unique_tracks",
        "track_length_min",
        "track_length_median",
        "track_length_max",
    ):
        assert k in tl, f"missing {k}"
    assert tl["n_unique_tracks"] > 0
    assert tl["track_length_min"] >= 1
    assert tl["track_length_max"] >= tl["track_length_median"]


def test_event_participation_partition(result):
    ep = result["event_participation"]
    total = (
        ep["tracks_in_1_event"]
        + ep["tracks_in_2_3_events"]
        + ep["tracks_in_4_10_events"]
        + ep["tracks_in_gt10_events"]
    )
    assert total == ep["n_tracks_in_events"], "partitions must sum"
    assert len(ep["top_5_hub_tracks"]) <= 5


def test_jump_stats_by_verdict(result):
    jv = result["jump_stats_by_verdict"]
    assert "Y" in jv and "N" in jv
    for v in ("Y", "N"):
        assert jv[v]["n"] > 0
        assert "max_p95_median" in jv[v]
