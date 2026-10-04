"""Tracker health metrics for the GITI pipeline output.

Run:
    python -m paper.analysis.tracker_health

Reads:
    outputs/giti_raw.csv
    outputs/giti_screened_with_gates.csv
    data/reviews/ssm_review_114/to_label.csv
    data/reviews/ssm_review_114/label_to_giti_mapping.csv
    data/reviews/ssm_review_114/ev_jumpstats.csv

Writes:
    paper/results/tracker_health.json
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "outputs" / "giti_raw.csv"
SCREENED = REPO / "outputs" / "giti_screened_with_gates.csv"
REVIEWS = REPO / "data" / "reviews" / "ssm_review_114"
LABELS = REVIEWS / "to_label.csv"
MAPPING = REVIEWS / "label_to_giti_mapping.csv"
JUMPSTATS = REVIEWS / "ev_jumpstats.csv"
OUT = REPO / "paper" / "results" / "tracker_health.json"


def _track_lengths(raw: pd.DataFrame) -> dict:
    lengths: dict[int, int] = {}
    for _, r in raw.iterrows():
        for tc, jc in (("track_a", "traj_a_json"), ("track_b", "traj_b_json")):
            t, js = r.get(tc), r.get(jc)
            if pd.isna(t) or pd.isna(js):
                continue
            try:
                pts = json.loads(js)
            except json.JSONDecodeError:
                continue
            tid = int(t)
            n = len(pts)
            if n > lengths.get(tid, 0):
                lengths[tid] = n
    vals = np.array(list(lengths.values()))
    return {
        "n_unique_tracks": len(lengths),
        "track_length_min": int(vals.min()),
        "track_length_p25": float(np.percentile(vals, 25)),
        "track_length_median": float(np.median(vals)),
        "track_length_p75": float(np.percentile(vals, 75)),
        "track_length_max": int(vals.max()),
        "track_length_mean": float(vals.mean()),
        "track_length_std": float(vals.std()),
        "short_tracks_le_10_frames": int((vals <= 10).sum()),
        "short_tracks_le_30_frames": int((vals <= 30).sum()),
    }


def _event_participation(screened: pd.DataFrame) -> dict:
    counts: Counter[int] = Counter()
    for _, r in screened.iterrows():
        counts[int(r.track_a)] += 1
        counts[int(r.track_b)] += 1
    vals = np.array(list(counts.values()))
    top = Counter(counts).most_common(5)
    return {
        "n_tracks_in_events": len(counts),
        "events_per_track_max": int(vals.max()),
        "events_per_track_mean": float(vals.mean()),
        "tracks_in_1_event": int((vals == 1).sum()),
        "tracks_in_2_3_events": int(((vals >= 2) & (vals <= 3)).sum()),
        "tracks_in_4_10_events": int(((vals >= 4) & (vals <= 10)).sum()),
        "tracks_in_gt10_events": int((vals > 10).sum()),
        "top_5_hub_tracks": [
            {"track_id": int(t), "n_events": int(n)} for t, n in top
        ],
    }


def _jump_stats_by_verdict(
    labels: pd.DataFrame, mapping: pd.DataFrame, jump: pd.DataFrame
) -> dict:
    m = labels.merge(mapping, left_on="idx", right_on="label_idx")
    m = m.merge(jump, left_on="label_idx", right_on="event_idx")
    out: dict = {}
    for v in ("Y", "N"):
        sub = m[m.verdict == v]
        if len(sub) == 0:
            continue
        out[v] = {
            "n": len(sub),
            "max_p95_median": float(sub.max_p95.median()),
            "max_p99_median": float(sub.max_p99.median()),
            "max_frac_median": float(sub.max_frac.median()),
            "max_jump_median": float(sub.max_jump.median()),
        }
    return out


def main() -> Path:
    raw = pd.read_csv(RAW)
    screened = pd.read_csv(SCREENED)
    labels = pd.read_csv(LABELS)
    mapping = pd.read_csv(MAPPING)
    jump = pd.read_csv(JUMPSTATS)

    result = {
        "track_lengths": _track_lengths(raw),
        "event_participation": _event_participation(screened),
        "jump_stats_by_verdict": _jump_stats_by_verdict(labels, mapping, jump),
    }
    OUT.write_text(json.dumps(result, indent=2))
    return OUT


if __name__ == "__main__":
    p = main()
    print(f"wrote {p}  ({p.stat().st_size} bytes)")
