"""Loose screening: find candidate conflicts the strict pipeline missed.

Run:  python -m paper.analysis.loose_candidates

Reads:  outputs/giti_raw.csv, outputs/giti_screened_with_gates.csv
Writes: paper/results/loose_candidates.json
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "outputs" / "giti_raw.csv"
SCREENED = REPO / "outputs" / "giti_screened_with_gates.csv"
OUT = REPO / "paper" / "results" / "loose_candidates.json"

FPS = 30.0
MIN_PATH_DIST_M = 3.0
MAX_TIME_GAP_S = 5.0
MIN_TRACK_FRAMES = 10


def _load_all_tracks():
    raw = pd.read_csv(RAW)
    tracks = {}
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
                d[int(p["frame"])] = (float(p["world_x"]), float(p["world_y"]))
    return tracks


def _known_pairs():
    pairs = set()
    for p in (RAW, SCREENED):
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            a = int(r.track_a)
            b = int(r.track_b)
            pairs.add((min(a, b), max(a, b)))
    return pairs


def _arr(td):
    frames = sorted(td)
    xy = np.array([td[f] for f in frames], dtype=float)
    return np.array(frames), xy


def _speed_mps(xy, frames):
    if len(frames) < 2:
        return 0.0
    d = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    dt = np.diff(frames) / FPS
    m = dt > 0
    if not m.any():
        return 0.0
    return float(np.mean(d[m] / dt[m]))


def _heading_deg(xy):
    if len(xy) < 2:
        return 0.0
    dx = xy[-1, 0] - xy[0, 0]
    dy = xy[-1, 1] - xy[0, 1]
    return float(np.degrees(np.arctan2(dy, dx)))


def _pair_features(fa, xya, fb, xyb):
    diff = xya[:, None, :] - xyb[None, :, :]
    dists = np.hypot(diff[:, :, 0], diff[:, :, 1])
    idx = np.unravel_index(np.argmin(dists), dists.shape)
    min_dist = float(dists[idx])
    fa_closest = int(fa[idx[0]])
    fb_closest = int(fb[idx[1]])
    time_gap_s = abs(fa_closest - fb_closest) / FPS
    speed_a = _speed_mps(xya, fa)
    speed_b = _speed_mps(xyb, fb)
    head_a = _heading_deg(xya)
    head_b = _heading_deg(xyb)
    hdiff = abs(head_a - head_b) % 360.0
    hdiff = min(hdiff, 360.0 - hdiff)
    return {
        "min_cross_time_dist_m": min_dist,
        "time_gap_at_min_s": time_gap_s,
        "mean_speed_a_mps": speed_a,
        "mean_speed_b_mps": speed_b,
        "heading_delta_deg": float(hdiff),
        "track_a_len": int(len(fa)),
        "track_b_len": int(len(fb)),
    }


def main():
    tracks = _load_all_tracks()
    known = _known_pairs()
    valid_ids = sorted(tid for tid, td in tracks.items() if len(td) >= MIN_TRACK_FRAMES)
    print(f"tracks total: {len(tracks)}, valid: {len(valid_ids)}")
    cands = []
    n_excluded = 0
    for a_id, b_id in combinations(valid_ids, 2):
        key = (min(a_id, b_id), max(a_id, b_id))
        if key in known:
            n_excluded += 1
            continue
        fa, xya = _arr(tracks[a_id])
        fb, xyb = _arr(tracks[b_id])
        f = _pair_features(fa, xya, fb, xyb)
        if f["min_cross_time_dist_m"] > MIN_PATH_DIST_M:
            continue
        if f["time_gap_at_min_s"] > MAX_TIME_GAP_S:
            continue
        score = (
            0.6 * (1.0 - f["min_cross_time_dist_m"] / MIN_PATH_DIST_M)
            + 0.4 * (1.0 - f["time_gap_at_min_s"] / MAX_TIME_GAP_S)
        )
        f["combined_score"] = score
        f["track_a"] = a_id
        f["track_b"] = b_id
        cands.append(f)
    cands.sort(key=lambda x: -x["combined_score"])
    result = {
        "n_tracks_total": len(tracks),
        "n_tracks_valid": len(valid_ids),
        "n_pairs_considered": len(valid_ids) * (len(valid_ids) - 1) // 2,
        "n_known_pairs_excluded": n_excluded,
        "n_candidates": len(cands),
        "filters": {
            "min_path_dist_m": MIN_PATH_DIST_M,
            "max_time_gap_s": MAX_TIME_GAP_S,
            "min_track_frames": MIN_TRACK_FRAMES,
        },
        "candidates": cands,
    }
    OUT.write_text(json.dumps(result, indent=2))
    return OUT


if __name__ == "__main__":
    p = main()
    print(f"wrote {p}  ({p.stat().st_size} bytes)")
