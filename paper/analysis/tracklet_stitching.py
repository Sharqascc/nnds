"""Tracklet stitching — link fragmented tracks by spatio-temporal continuity.

Run:  python -m paper.analysis.tracklet_stitching

Reads:  outputs/giti_raw.csv
        outputs/giti_screened_with_gates.csv
        data/reviews/ssm_review_114/to_label.csv
        data/reviews/ssm_review_114/label_to_giti_mapping.csv
Writes: paper/results/tracklet_stitching.json

Pre-specified merge thresholds (from tracking_diagnostics.py, not tuned):
    time gap <= 15 frames   (~0.5 s at 30 fps)
    spatial gap <= 1.5 m
    speed delta <= 3 m/s
    heading delta <= 30 deg
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "outputs" / "giti_raw.csv"
SCREENED = REPO / "outputs" / "giti_screened_with_gates.csv"
LABELS = REPO / "data" / "reviews" / "ssm_review_114" / "to_label.csv"
MAPPING = REPO / "data" / "reviews" / "ssm_review_114" / "label_to_giti_mapping.csv"
OUT = REPO / "paper" / "results" / "tracklet_stitching.json"

FPS = 30.0
MAX_GAP_FRAMES = 15
MAX_GAP_M = 1.5
MAX_SPEED_DELTA_MPS = 3.0
MAX_HEADING_DELTA_DEG = 30.0
SHORT_TRACK_FRAMES = 30
END_WINDOW = 5


def _load_tracks(raw: pd.DataFrame) -> dict:
    tracks: dict = {}
    for _, r in raw.iterrows():
        for tc, jc in (("track_a", "traj_a_json"), ("track_b", "traj_b_json")):
            t = r.get(tc)
            js = r.get(jc)
            if pd.isna(t) or pd.isna(js):
                continue
            try:
                pts = json.loads(js)
            except json.JSONDecodeError:
                continue
            tid = int(t)
            d = tracks.setdefault(tid, {})
            for p in pts:
                d[int(p["frame"])] = (
                    float(p["x_pixel"]),
                    float(p["y_pixel"]),
                    float(p["world_x"]),
                    float(p["world_y"]),
                )
    return tracks


def _endpoint(td: dict, frames: list, at_end: bool) -> dict:
    sel = frames[-END_WINDOW:] if at_end else frames[:END_WINDOW]
    if len(sel) < 2:
        return {"xy": None, "speed": 0.0, "heading": 0.0}
    xs = np.array([td[f][2] for f in sel])
    ys = np.array([td[f][3] for f in sel])
    ds = np.hypot(np.diff(xs), np.diff(ys))
    dfs = np.diff(sel)
    valid = dfs > 0
    if not valid.any():
        return {"xy": (float(xs[-1]), float(ys[-1])), "speed": 0.0, "heading": 0.0}
    speeds = ds[valid] / dfs[valid] * FPS
    speed = float(np.median(speeds))
    dx = xs[-1] - xs[0]
    dy = ys[-1] - ys[0]
    heading = float(np.degrees(np.arctan2(dy, dx)))
    xy = (float(xs[-1]), float(ys[-1]))
    return {"xy": xy, "speed": speed, "heading": heading}


def _track_summary(td: dict) -> dict:
    frames = sorted(td)
    if len(frames) < 2:
        return None
    start = _endpoint(td, frames, at_end=False)
    end = _endpoint(td, frames, at_end=True)
    return {
        "start_frame": frames[0],
        "end_frame": frames[-1],
        "length": len(frames),
        "start_xy": start["xy"],
        "end_xy": end["xy"],
        "start_speed": start["speed"],
        "end_speed": end["speed"],
        "start_heading": start["heading"],
        "end_heading": end["heading"],
    }


def _angle_diff_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return float(min(d, 360.0 - d))


def _merge_score(a: dict, b: dict) -> dict:
    gap_frames = b["start_frame"] - a["end_frame"]
    if gap_frames < 0 or gap_frames > MAX_GAP_FRAMES:
        return None
    ax, ay = a["end_xy"]
    bx, by = b["start_xy"]
    space_gap = float(np.hypot(ax - bx, ay - by))
    if space_gap > MAX_GAP_M:
        return None
    speed_delta = abs(a["end_speed"] - b["start_speed"])
    if speed_delta > MAX_SPEED_DELTA_MPS:
        return None
    heading_delta = _angle_diff_deg(a["end_heading"], b["start_heading"])
    if heading_delta > MAX_HEADING_DELTA_DEG:
        return None
    score = (
        1.0
        - gap_frames / MAX_GAP_FRAMES
        + 1.0
        - space_gap / MAX_GAP_M
        + 1.0
        - speed_delta / MAX_SPEED_DELTA_MPS
        + 1.0
        - heading_delta / MAX_HEADING_DELTA_DEG
    ) / 4.0
    return {
        "gap_frames": int(gap_frames),
        "space_gap_m": space_gap,
        "speed_delta_mps": speed_delta,
        "heading_delta_deg": heading_delta,
        "score": score,
    }


def _find_candidates(summaries: dict) -> list:
    cands = []
    for a_id, a in summaries.items():
        if a is None:
            continue
        for b_id, b in summaries.items():
            if b is None or a_id == b_id:
                continue
            m = _merge_score(a, b)
            if m is None:
                continue
            cands.append({"a": a_id, "b": b_id, **m})
    cands.sort(key=lambda x: -x["score"])
    return cands


def _greedy_chains(cands: list) -> list:
    used_end: set = set()
    used_start: set = set()
    chains = []
    for c in cands:
        if c["a"] in used_end or c["b"] in used_start:
            continue
        if c["a"] == c["b"]:
            continue
        used_end.add(c["a"])
        used_start.add(c["b"])
        chains.append(c)
    merged = {}
    for c in chains:
        target = merged.get(c["a"], c["a"])
        merged[c["b"]] = target
    return chains, merged


def main() -> Path:
    raw = pd.read_csv(RAW)
    tracks = _load_tracks(raw)
    summaries = {tid: _track_summary(td) for tid, td in tracks.items()}
    valid_summaries = {tid: s for tid, s in summaries.items() if s is not None}

    short_before = [tid for tid, s in valid_summaries.items() if s["length"] <= SHORT_TRACK_FRAMES]

    cands = _find_candidates(valid_summaries)
    chains, merge_map = _greedy_chains(cands)

    merged_lengths: dict = {}
    for tid, s in valid_summaries.items():
        root = merge_map.get(tid, tid)
        merged_lengths[root] = merged_lengths.get(root, 0) + s["length"]
    short_after = [root for root, L in merged_lengths.items() if L <= SHORT_TRACK_FRAMES]

    screened = pd.read_csv(SCREENED)
    labels = pd.read_csv(LABELS)
    mapping = pd.read_csv(MAPPING)
    valid_idx = set(mapping.label_idx)

    def event_hits_short(track_id: int, short_set: set) -> bool:
        if track_id in short_set:
            return True
        root = merge_map.get(track_id)
        return root in short_set if root is not None else False

    ev_rows = []
    for _, r in labels.iterrows():
        if r.idx not in valid_idx:
            continue
        match = screened[
            screened.index == int(mapping.loc[mapping.label_idx == r.idx, "giti_idx"].iloc[0])
        ]
        if len(match) == 0:
            continue
        ta = int(match.iloc[0]["track_a"])
        tb = int(match.iloc[0]["track_b"])
        ev_rows.append(
            {
                "idx": int(r.idx),
                "verdict": r.verdict,
                "before_short": event_hits_short(ta, set(short_before))
                or event_hits_short(tb, set(short_before)),
                "after_short": event_hits_short(ta, set(short_after))
                or event_hits_short(tb, set(short_after)),
            }
        )
    ev = pd.DataFrame(ev_rows)

    def stats(mask: pd.Series) -> dict:
        sub = ev[mask]
        y = int((sub["verdict"] == "Y").sum())
        n = int((sub["verdict"] == "N").sum())
        total = y + n
        return {"n": total, "Y": y, "N": n, "precision": (y / total) if total else 0.0}

    before_short = stats(ev["before_short"])
    after_short = stats(ev["after_short"])

    out = {
        "n_tracks_before": len(valid_summaries),
        "n_short_before": len(short_before),
        "n_merge_candidates": len(cands),
        "n_chains": len(chains),
        "n_short_after": len(short_after),
        "short_reduction": len(short_before) - len(short_after),
        "merge_chains": chains[:50],
        "events_with_short_track_before": before_short,
        "events_with_short_track_after": after_short,
        "thresholds": {
            "max_gap_frames": MAX_GAP_FRAMES,
            "max_gap_m": MAX_GAP_M,
            "max_speed_delta_mps": MAX_SPEED_DELTA_MPS,
            "max_heading_delta_deg": MAX_HEADING_DELTA_DEG,
        },
    }
    OUT.write_text(json.dumps(out, indent=2))
    return OUT


if __name__ == "__main__":
    p = main()
    print(f"wrote {p}  ({p.stat().st_size} bytes)")
