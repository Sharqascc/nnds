"""Per-track quality flags and correlation with PET verdict.

Run:  python -m paper.analysis.track_quality_detailed

Reads:  outputs/giti_raw.csv
        data/reviews/ssm_review_114/to_label.csv
        data/reviews/ssm_review_114/label_to_giti_mapping.csv
Writes: paper/results/track_quality_detailed.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "outputs" / "giti_raw.csv"
LABELS = REPO / "data" / "reviews" / "ssm_review_114" / "to_label.csv"
MAPPING = REPO / "data" / "reviews" / "ssm_review_114" / "label_to_giti_mapping.csv"
OUT = REPO / "paper" / "results" / "track_quality_detailed.json"

FPS = 30.0
SHORT_FRAMES = 30
GAPPY_RATE = 0.20
JUMPY_P95_DEG = 45.0
FAST_MPS = 20.0
HUGE_JUMP_M = 5.0


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


def _track_metrics(td: dict) -> dict:
    frames = sorted(td)
    n = len(frames)
    span = frames[-1] - frames[0] + 1
    gap_rate = (span - n) / span if span else 0.0
    max_gap = 0
    for i in range(1, len(frames)):
        d = frames[i] - frames[i - 1] - 1
        if d > max_gap:
            max_gap = d
    speeds = []
    jumps = []
    headings = []
    for i in range(1, len(frames)):
        f0 = frames[i - 1]
        f1 = frames[i]
        df = f1 - f0
        if df <= 0:
            continue
        dx = td[f1][2] - td[f0][2]
        dy = td[f1][3] - td[f0][3]
        dist = float(np.hypot(dx, dy))
        speeds.append(dist / df * FPS)
        jumps.append(dist)
        headings.append(float(np.degrees(np.arctan2(dy, dx))))
    speeds_arr = np.array(speeds) if speeds else np.array([0.0])
    jumps_arr = np.array(jumps) if jumps else np.array([0.0])
    if len(headings) >= 2:
        dth = np.abs(np.diff(np.array(headings)))
        dth = np.minimum(dth, 360.0 - dth)
        heading_p95 = float(np.percentile(dth, 95))
    else:
        heading_p95 = 0.0
    return {
        "length": n,
        "span": span,
        "gap_rate": float(gap_rate),
        "max_gap_frames": int(max_gap),
        "mean_speed_mps": float(speeds_arr.mean()),
        "max_speed_mps": float(speeds_arr.max()),
        "max_jump_m": float(jumps_arr.max()),
        "heading_p95_deg": float(heading_p95),
    }


def _flags(m: dict) -> list:
    flags = []
    if m["length"] < SHORT_FRAMES:
        flags.append("short")
    if m["gap_rate"] > GAPPY_RATE:
        flags.append("gappy")
    if m["heading_p95_deg"] > JUMPY_P95_DEG:
        flags.append("jumpy")
    if m["max_speed_mps"] > FAST_MPS:
        flags.append("fast")
    if m["max_jump_m"] > HUGE_JUMP_M:
        flags.append("huge_jump")
    return flags


def main() -> Path:
    raw = pd.read_csv(RAW)
    tracks = _load_tracks(raw)
    per_track = {}
    for tid, td in tracks.items():
        m = _track_metrics(td)
        m["track_id"] = int(tid)
        m["flags"] = _flags(m)
        m["n_flags"] = len(m["flags"])
        per_track[int(tid)] = m

    labels = pd.read_csv(LABELS)
    mapping = pd.read_csv(MAPPING)
    valid_idx = set(mapping.label_idx)
    rows = []
    for _, r in labels.iterrows():
        if r.idx not in valid_idx:
            continue
        ta = int(r.track_a)
        tb = int(r.track_b)
        ma = per_track.get(ta, {"flags": []})
        mb = per_track.get(tb, {"flags": []})
        any_flags = set(ma.get("flags", [])) | set(mb.get("flags", []))
        rows.append(
            {
                "idx": int(r.idx),
                "verdict": r.verdict,
                "track_a": ta,
                "track_b": tb,
                "a_flags": ",".join(ma.get("flags", [])),
                "b_flags": ",".join(mb.get("flags", [])),
                "any_flags": ",".join(sorted(any_flags)),
                "n_any_flags": len(any_flags),
            }
        )
    ev = pd.DataFrame(rows)

    ct = pd.crosstab(ev["n_any_flags"] > 0, ev["verdict"])
    ct_dict = {str(k): v for k, v in ct.to_dict().items()}

    all_flags = set()
    for f in ev["any_flags"]:
        for x in f.split(","):
            if x:
                all_flags.add(x)

    flag_stats = {}
    for f in sorted(all_flags):
        mask = ev["any_flags"].str.contains(rf"\b{f}\b", regex=True)
        sub = ev[mask]
        y = int((sub["verdict"] == "Y").sum())
        n = int((sub["verdict"] == "N").sum())
        flag_stats[f] = {
            "n_total": len(sub),
            "Y": y,
            "N": n,
            "precision": y / len(sub) if len(sub) else 0.0,
        }

    clean = ev[ev["n_any_flags"] == 0]
    clean_stats: dict = {
        "n": len(clean),
        "Y": int((clean["verdict"] == "Y").sum()),
        "N": int((clean["verdict"] == "N").sum()),
    }
    clean_stats["precision"] = clean_stats["Y"] / clean_stats["n"] if clean_stats["n"] else 0.0

    out = {
        "n_tracks": len(per_track),
        "per_track": list(per_track.values()),
        "flag_stats": flag_stats,
        "clean_events": clean_stats,
        "crosstab_any_flag_vs_verdict": ct_dict,
    }
    OUT.write_text(json.dumps(out, indent=2))
    return OUT


if __name__ == "__main__":
    p = main()
    print(f"wrote {p}  ({p.stat().st_size} bytes)")
