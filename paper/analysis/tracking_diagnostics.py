"""Tracker diagnostics and PET robustness test.

Run:  python -m paper.analysis.tracking_diagnostics

Reads:  outputs/giti_raw.csv
Writes: paper/results/tracking_diagnostics.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "outputs" / "giti_raw.csv"
OUT = REPO / "paper" / "results" / "tracking_diagnostics.json"

FPS = 30.0
FRAG_MAX_GAP = 15
FRAG_MAX_DIST_M = 1.5
HEADING_THRESH_DEG = 45.0
MAX_PLAUSIBLE_SPEED = 30.0
SEED = 42

# BEV pixel -> world parameters (mirrors configs/bev_config.json)
A_X = 0.01586042861296038
C_X = 730897.8930766084
A_Y = 0.027072759088983777
C_Y = 221994.9929775622
X_MIN = 730900.97
Y_MAX = 222014.35


def _load_tracks(raw: pd.DataFrame) -> dict:
    tracks: dict = {}
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
            d = tracks.setdefault(tid, {})
            for p in pts:
                d[int(p["frame"])] = (
                    float(p["x_pixel"]),
                    float(p["y_pixel"]),
                    float(p["world_x"]),
                    float(p["world_y"]),
                )
    return tracks


def _track_lengths(tracks: dict) -> dict:
    lengths = np.array([len(t) for t in tracks.values()])
    return {
        "n_tracks": len(lengths),
        "min": int(lengths.min()),
        "p25": float(np.percentile(lengths, 25)),
        "median": float(np.median(lengths)),
        "p75": float(np.percentile(lengths, 75)),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "short_le_10": int((lengths <= 10).sum()),
        "short_le_30": int((lengths <= 30).sum()),
    }


def _gap_rate(tracks: dict) -> dict:
    rates = []
    for td in tracks.values():
        if len(td) < 2:
            continue
        frames = sorted(td)
        span = frames[-1] - frames[0] + 1
        if span <= 0:
            continue
        rates.append((span - len(frames)) / span)
    arr = np.array(rates) if rates else np.array([0.0])
    return {
        "n_tracks": len(rates),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
        "tracks_with_any_gap": int((arr > 0).sum()),
    }


def _fragment_pairs(tracks: dict) -> dict:
    ends = []
    starts = []
    for tid, td in tracks.items():
        if len(td) < 3:
            continue
        frames = sorted(td)
        ends.append((tid, frames[-1], td[frames[-1]][2], td[frames[-1]][3]))
        starts.append((tid, frames[0], td[frames[0]][2], td[frames[0]][3]))
    pairs = 0
    for ea in ends:
        for sb in starts:
            if ea[0] == sb[0]:
                continue
            gap = sb[1] - ea[1]
            if gap < 0 or gap > FRAG_MAX_GAP:
                continue
            dist = float(np.hypot(ea[2] - sb[2], ea[3] - sb[3]))
            if dist <= FRAG_MAX_DIST_M:
                pairs += 1
    return {
        "n_fragment_pairs": pairs,
        "n_tracks_considered": len(ends),
        "max_gap_frames": FRAG_MAX_GAP,
        "max_distance_m": FRAG_MAX_DIST_M,
    }


def _heading_instability(tracks: dict) -> dict:
    total_pairs = 0
    violations = 0
    for td in tracks.values():
        frames = sorted(td)
        if len(frames) < 3:
            continue
        xs = np.array([td[f][2] for f in frames])
        ys = np.array([td[f][3] for f in frames])
        dx = np.diff(xs)
        dy = np.diff(ys)
        theta = np.degrees(np.arctan2(dy, dx))
        dtheta = np.abs(np.diff(theta))
        dtheta = np.minimum(dtheta, 360.0 - dtheta)
        total_pairs += len(dtheta)
        violations += int((dtheta > HEADING_THRESH_DEG).sum())
    rate = (violations / total_pairs) if total_pairs else 0.0
    return {
        "frame_pairs": total_pairs,
        "violations": violations,
        "violation_rate": float(rate),
        "threshold_deg": HEADING_THRESH_DEG,
    }


def _speed_plausibility(tracks: dict) -> dict:
    speeds = []
    for td in tracks.values():
        frames = sorted(td)
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
    arr = np.array(speeds) if speeds else np.array([0.0])
    return {
        "n_samples": len(speeds),
        "median_mps": float(np.median(arr)),
        "p95_mps": float(np.percentile(arr, 95)),
        "max_mps": float(arr.max()),
        "implausible_count": int((arr > MAX_PLAUSIBLE_SPEED).sum()),
        "max_plausible_mps": MAX_PLAUSIBLE_SPEED,
    }


def _to_cell_px(px: float, py: float) -> tuple[int, int]:
    wx = A_X * px + C_X - X_MIN
    wy = Y_MAX - (A_Y * py + C_Y)
    col = int(wx / 0.4)
    row = int(wy / 0.4)
    return col, row


def _perturbation_test(
    raw: pd.DataFrame,
    tracks: dict,
    perturb: str,
    n_trials: int,
) -> dict:
    rng = np.random.default_rng(SEED)
    n_robust = 0
    n_total = 0
    for _, r in raw.iterrows():
        ta = int(r.track_a)
        tb = int(r.track_b)
        tA = tracks.get(ta, {})
        tB = tracks.get(tb, {})
        if not tA or not tB:
            continue
        n_total += 1
        robust_trials = 0
        for _ in range(n_trials):
            fa = sorted(tA)
            fb = sorted(tB)
            if perturb == "drop":
                na = max(1, int(0.9 * len(fa)))
                nb = max(1, int(0.9 * len(fb)))
                fa = [fa[i] for i in sorted(rng.choice(len(fa), na, replace=False))]
                fb = [fb[i] for i in sorted(rng.choice(len(fb), nb, replace=False))]
            cells_A = set()
            cells_B = set()
            for f in fa:
                px = tA[f][0]
                py = tA[f][1]
                if perturb == "noise":
                    px += float(rng.normal(0, 1.0))
                    py += float(rng.normal(0, 1.0))
                cells_A.add(_to_cell_px(px, py))
            for f in fb:
                px = tB[f][0]
                py = tB[f][1]
                if perturb == "noise":
                    px += float(rng.normal(0, 1.0))
                    py += float(rng.normal(0, 1.0))
                cells_B.add(_to_cell_px(px, py))
            if cells_A & cells_B:
                robust_trials += 1
        if robust_trials > 0:
            n_robust += 1
    frac = (n_robust / n_total) if n_total else 0.0
    return {
        "perturbation": perturb,
        "n_trials_per_event": n_trials,
        "n_events": n_total,
        "n_robust": n_robust,
        "fraction_robust": float(frac),
    }


def main() -> Path:
    raw = pd.read_csv(RAW)
    tracks = _load_tracks(raw)
    result: dict[str, Any] = {
        "n_unique_tracks": len(tracks),
        "track_lengths": _track_lengths(tracks),
        "gap_rate": _gap_rate(tracks),
        "fragment_pairs": _fragment_pairs(tracks),
        "heading_instability": _heading_instability(tracks),
        "speed_plausibility": _speed_plausibility(tracks),
        "robustness_dropout_10pct": _perturbation_test(raw, tracks, "drop", 10),
        "robustness_pixel_noise_1px": _perturbation_test(raw, tracks, "noise", 10),
    }
    OUT.write_text(json.dumps(result, indent=2))
    return OUT


if __name__ == "__main__":
    p = main()
    print(f"wrote {p}  ({p.stat().st_size} bytes)")
