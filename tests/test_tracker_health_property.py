"""Property tests for paper.analysis.tracker_health."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from paper.analysis import tracker_health as th


def test_track_lengths_shape_on_empty():
    raw = pd.DataFrame(columns=["track_a", "traj_a_json", "track_b", "traj_b_json"])
    out = th._track_lengths(raw)
    assert "n_unique_tracks" in out
    assert isinstance(out["n_unique_tracks"], int)


def test_track_lengths_counts_a_simple_track():
    traj = [
        {"frame": i, "x_pixel": 0.0, "y_pixel": 0.0, "world_x": 0.0, "world_y": 0.0}
        for i in range(10)
    ]
    raw = pd.DataFrame(
        [
            {
                "track_a": 1,
                "traj_a_json": json.dumps(traj),
                "track_b": None,
                "traj_b_json": None,
            }
        ]
    )
    out = th._track_lengths(raw)
    assert out["n_unique_tracks"] == 1
    assert out["track_length_max"] == 10


def test_event_participation_empty_screened():
    screened = pd.DataFrame(columns=["track_a", "track_b"])
    out = th._event_participation(screened)
    assert out["n_tracks_in_events"] == 0
    assert out["top_5_hub_tracks"] == []


def test_event_participation_counts():
    screened = pd.DataFrame(
        {
            "track_a": [1, 1, 2],
            "track_b": [2, 3, 4],
        }
    )
    out = th._event_participation(screened)
    assert out["n_tracks_in_events"] == 4
    assert out["events_per_track_max"] == 2
