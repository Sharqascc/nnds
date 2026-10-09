"""Property tests for paper.analysis.blind_windows."""

from __future__ import annotations

import pandas as pd

from paper.analysis import blind_windows as bw


def test_forbidden_frames_on_empty_df():
    df = pd.DataFrame(
        columns=[
            "track_a_entry_frame",
            "track_a_exit_frame",
            "track_b_entry_frame",
            "track_b_exit_frame",
        ]
    )
    assert bw._forbidden_frames(df) == set()


def test_sample_windows_cardinality_and_bounds():
    forbidden: set = set()
    starts = bw._sample_windows(forbidden)
    assert len(starts) == bw.N_WINDOWS
    for s in starts:
        assert isinstance(s, int)
        assert s >= 0
        assert s + bw.WINDOW_FRAMES <= bw.TOTAL_FRAMES


def test_sample_windows_avoids_forbidden_frames():
    forbidden = set(range(0, 200))
    starts = bw._sample_windows(forbidden)
    for s in starts:
        window = set(range(s, s + bw.WINDOW_FRAMES))
        assert not (window & forbidden)


def test_sample_windows_is_deterministic():
    a = bw._sample_windows(set())
    b = bw._sample_windows(set())
    assert a == b


def test_make_html_contains_one_video_per_row():
    rows = [
        {"clip_id": "blind_000", "start_frame": 0},
        {"clip_id": "blind_001", "start_frame": 100},
    ]
    html = bw._make_html(rows)
    assert html.count("<video") == 2
    assert "blind_000" in html
    assert "blind_001" in html
