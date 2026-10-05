"""Tracker quality v2 — heading fix, gap buckets, continuous Q.

Run:  python -m paper.analysis.tracker_quality_v2

Reads:  outputs/giti_raw.csv
        data/reviews/ssm_review_114/to_label.csv
        data/reviews/ssm_review_114/label_to_giti_mapping.csv
Writes: paper/results/tracker_quality_v2.json

Fixes over v1:
- heading computed only when speed > MIN_SPEED_MPS (0.3 m/s)
- gaps bucketed: 1-2 / 3-5 / 6-15 / >15 frames
- continuous per-track quality Q in [0, 1]
- AUC of Q against 103-event TP/FP labels
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "outputs" / "giti_raw.csv"
SCREENED = REPO / "outputs" / "giti_screened_with_gates.csv"
LABELS = REPO / "data" / "reviews" / "ssm_review_114" / "to_label.csv"
MAPPING = REPO / "data" / "reviews" / "ssm_review_114" / "label_to_giti_mapping.csv"
OUT = REPO / "paper" / "results" / "tracker_quality_v2.json"

FPS = 30.0
MIN_SPEED_MPS = 0.3
SHORT_TRACK_FRAMES = 30
HEADING_JITTER_DEG = 45.0


def _load_tracks(raw: pd.DataFrame) -> dict:
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
                d[int(p["frame"])] = (
                    float(p["x_pixel"]),
                    float(p["y_pixel"]),
                    float(p["world_x"]),
                    float(p["world_y"]),
                )
    return tracks


def _metrics(td: dict) -> dict:
    frames = sorted(td)
    n = len(frames)
    if n < 2:
        return {"length": n, "gap_rate": 0.0, "gap_bucket": "n/a",
                "speed_mean": 0.0, "speed_max": 0.0,
                "heading_jitter_rate": 0.0, "net_disp": 0.0}
    span = frames[-1] - frames[0] + 1
    gaps = [frames[i] - frames[i - 1] - 1 for i in range(1, len(frames))]
    max_gap = max(gaps) if gaps else 0
    if max_gap <= 2:
        bucket = "0-2"
    elif max_gap <= 5:
        bucket = "3-5"
    elif max_gap <= 15:
        bucket = "6-15"
    else:
        bucket = ">15"
    gap_rate = (span - n) / span if span else 0.0

    speeds = []
    headings = []
    for i in range(1, len(frames)):
        f0, f1 = frames[i - 1], frames[i]
        df = f1 - f0
        if df <= 0:
            continue
        dx = td[f1][2] - td[f0][2]
        dy = td[f1][3] - td[f0][3]
        d = float(np.hypot(dx, dy))
        v = d / df * FPS
        speeds.append(v)
        if v >= MIN_SPEED_MPS:
            headings.append(float(np.degrees(np.arctan2(dy, dx))))
    if len(headings) >= 2:
        arr = np.array(headings)
        dth = np.abs(np.diff(arr))
        dth = np.minimum(dth, 360.0 - dth)
        jitter_rate = float((dth > HEADING_JITTER_DEG).mean())
    else:
        jitter_rate = 0.0

    net_disp = float(np.hypot(td[frames[-1]][2] - td[frames[0]][2],
                              td[frames[-1]][3] - td[frames[0]][3]))
    speeds_arr = np.array(speeds) if speeds else np.array([0.0])
    return {"length": n, "gap_rate": float(gap_rate), "gap_bucket": bucket,
            "speed_mean": float(speeds_arr.mean()),
            "speed_max": float(speeds_arr.max()),
            "heading_jitter_rate": jitter_rate,
            "net_disp": net_disp}


def _quality(m: dict) -> float:
    # Q in [0,1]: high = clean, low = unreliable
    q_len = min(1.0, m["length"] / 60.0)
    q_gap = 1.0 - min(1.0, m["gap_rate"])
    q_jit = 1.0 - min(1.0, m["heading_jitter_rate"])
    return float(0.4 * q_len + 0.4 * q_gap + 0.2 * q_jit)


def main() -> Path:
    raw = pd.read_csv(RAW)
    tracks = _load_tracks(raw)
    per_track = {}
    for tid, td in tracks.items():
        m = _metrics(td)
        m["quality"] = _quality(m)
        m["track_id"] = tid
        per_track[tid] = m

    labels = pd.read_csv(LABELS)
    mapping = pd.read_csv(MAPPING)
    screened = pd.read_csv(SCREENED)
    giti_to_label = {int(r.giti_idx): int(r.label_idx) for _, r in mapping.iterrows()}
    idx_to_verdict = dict(zip(labels.idx, labels.verdict, strict=True))

    event_q = []
    y = []
    for gi, row in screened.iterrows():
        li = giti_to_label.get(int(gi))
        if li is None:
            continue
        verdict = idx_to_verdict.get(li)
        if verdict not in ("Y", "N"):
            continue
        ta = int(row.track_a)
        tb = int(row.track_b)
        qa = per_track.get(ta, {}).get("quality", 0.0)
        qb = per_track.get(tb, {}).get("quality", 0.0)
        event_q.append(min(qa, qb))  # weaker track dominates
        y.append(1 if verdict == "Y" else 0)

    y_arr = np.array(y)
    q_arr = np.array(event_q)
    try:
        auc = float(roc_auc_score(y_arr, q_arr))
    except ValueError:
        auc = 0.5

    gap_buckets = {}
    for m in per_track.values():
        b = m["gap_bucket"]
        gap_buckets[b] = gap_buckets.get(b, 0) + 1

    jitter_vals = [m["heading_jitter_rate"] for m in per_track.values()]
    short_tracks = [t for t, m in per_track.items() if m["length"] <= SHORT_TRACK_FRAMES]

    out = {
        "n_tracks": len(per_track),
        "n_events_scored": len(y),
        "n_positive_events": int(y_arr.sum()),
        "quality_auc_vs_verdict": auc,
        "quality_mean_Y": float(np.array(event_q)[y_arr == 1].mean()) if (y_arr == 1).any() else 0.0,
        "quality_mean_N": float(np.array(event_q)[y_arr == 0].mean()) if (y_arr == 0).any() else 0.0,
        "heading_jitter_median": float(np.median(jitter_vals)),
        "heading_jitter_p90": float(np.percentile(jitter_vals, 90)),
        "gap_buckets": gap_buckets,
        "short_track_count": len(short_tracks),
        "min_speed_for_heading_mps": MIN_SPEED_MPS,
        "per_track": list(per_track.values()),
    }
    OUT.write_text(json.dumps(out, indent=2))
    return OUT


if __name__ == "__main__":
    p = main()
    print(f"wrote {p}  ({p.stat().st_size} bytes)")
