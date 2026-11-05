"""Programmatic FP taxonomy — features, AUC, Fisher, clustering.

Run:  python -m paper.analysis.pair_analysis

Reads:  outputs/giti_screened_with_gates.csv
        data/reviews/ssm_review_114/to_label.csv
        data/reviews/ssm_review_114/label_to_giti_mapping.csv
Writes: paper/results/pair_analysis.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
SCREENED = REPO / "outputs" / "giti_screened_with_gates.csv"
LABELS = REPO / "data" / "reviews" / "ssm_review_114" / "to_label.csv"
MAPPING = REPO / "data" / "reviews" / "ssm_review_114" / "label_to_giti_mapping.csv"
OUT = REPO / "paper" / "results" / "pair_analysis.json"

FPS = 30.0
STATIONARY_MPS = 0.5


def _track_points(js: str) -> dict:
    out: dict = {}
    try:
        pts = json.loads(js)
    except (json.JSONDecodeError, TypeError):
        return out
    for p in pts:
        out[int(p["frame"])] = (float(p["world_x"]), float(p["world_y"]))
    return out


def _speed_stats(td: dict) -> tuple:
    frames = sorted(td)
    if len(frames) < 2:
        return 0.0, 0.0
    speeds = []
    for i in range(1, len(frames)):
        f0, f1 = frames[i - 1], frames[i]
        df = f1 - f0
        if df <= 0:
            continue
        d = float(np.hypot(td[f1][0] - td[f0][0], td[f1][1] - td[f0][1]))
        speeds.append(d / df * FPS)
    if not speeds:
        return 0.0, 0.0
    return float(np.mean(speeds)), float(np.max(speeds))


def _heading(td: dict) -> float:
    frames = sorted(td)
    if len(frames) < 2:
        return 0.0
    dx = td[frames[-1]][0] - td[frames[0]][0]
    dy = td[frames[-1]][1] - td[frames[0]][1]
    return float(np.degrees(np.arctan2(dy, dx)))


def _gap_rate(td: dict) -> float:
    frames = sorted(td)
    if len(frames) < 2:
        return 0.0
    span = frames[-1] - frames[0] + 1
    return (span - len(frames)) / span if span > 0 else 0.0


def _net_disp(td: dict) -> float:
    frames = sorted(td)
    if len(frames) < 2:
        return 0.0
    return float(
        np.hypot(td[frames[-1]][0] - td[frames[0]][0], td[frames[-1]][1] - td[frames[0]][1])
    )


def _event_features(ta: dict, tb: dict) -> dict:
    common = sorted(set(ta) & set(tb))
    n_common = len(common)
    if n_common > 0:
        dists = np.array(
            [float(np.hypot(ta[f][0] - tb[f][0], ta[f][1] - tb[f][1])) for f in common]
        )
        min_common = float(dists.min())
        frac_under_2 = float((dists < 2.0).mean())
        frac_under_5 = float((dists < 5.0).mean())
    else:
        min_common = float("inf")
        frac_under_2 = 0.0
        frac_under_5 = 0.0

    if ta and tb:
        cross = []
        for _, (xa, ya) in ta.items():
            for _, (xb, yb) in tb.items():
                cross.append(float(np.hypot(xa - xb, ya - yb)))
        min_cross = float(min(cross)) if cross else float("inf")
    else:
        min_cross = float("inf")

    spd_a_mean, spd_a_max = _speed_stats(ta)
    spd_b_mean, spd_b_max = _speed_stats(tb)
    head_a = _heading(ta)
    head_b = _heading(tb)
    hdiff = abs(head_a - head_b) % 360.0
    hdiff = min(hdiff, 360.0 - hdiff)
    overlap = n_common / min(len(ta), len(tb)) if ta and tb else 0.0

    return {
        "n_common_frames": n_common,
        "overlap_ratio": overlap,
        "min_concurrent_dist_m": min_common,
        "min_cross_time_dist_m": min_cross,
        "frac_concurrent_under_2m": frac_under_2,
        "frac_concurrent_under_5m": frac_under_5,
        "track_a_len": len(ta),
        "track_b_len": len(tb),
        "mean_speed_a_mps": spd_a_mean,
        "mean_speed_b_mps": spd_b_mean,
        "max_speed_a_mps": spd_a_max,
        "max_speed_b_mps": spd_b_max,
        "heading_a_deg": head_a,
        "heading_b_deg": head_b,
        "heading_delta_deg": hdiff,
        "cos_heading_delta": float(np.cos(np.radians(hdiff))),
        "gap_rate_a": _gap_rate(ta),
        "gap_rate_b": _gap_rate(tb),
        "net_disp_a_m": _net_disp(ta),
        "net_disp_b_m": _net_disp(tb),
        "both_stationary": int(spd_a_mean < STATIONARY_MPS and spd_b_mean < STATIONARY_MPS),
        "one_stationary": int((spd_a_mean < STATIONARY_MPS) != (spd_b_mean < STATIONARY_MPS)),
    }


def _fisher_ratio(x: np.ndarray, y: np.ndarray) -> float:
    pos = x[y == 1]
    neg = x[y == 0]
    if len(pos) < 2 or len(neg) < 2:
        return 0.0
    num = (pos.mean() - neg.mean()) ** 2
    den = pos.var() + neg.var() + 1e-9
    return float(np.log1p(num / den))


def main() -> Path:
    screened = pd.read_csv(SCREENED)
    labels = pd.read_csv(LABELS)
    mapping = pd.read_csv(MAPPING)

    idx_to_verdict = dict(zip(labels.idx, labels.verdict, strict=True))
    giti_to_label = {}
    for _, row in mapping.iterrows():
        giti_to_label[int(row.giti_idx)] = int(row.label_idx)

    features = []
    verdicts = []
    for gi, row in screened.iterrows():
        li = giti_to_label.get(int(gi))
        if li is None:
            continue
        verdict = idx_to_verdict.get(li)
        if verdict not in ("Y", "N"):
            continue
        ta = _track_points(row.traj_a_json)
        tb = _track_points(row.traj_b_json)
        if not ta or not tb:
            continue
        feats = _event_features(ta, tb)
        feats["pet_s"] = float(row.pet)
        feats["event_idx"] = li
        features.append(feats)
        verdicts.append(verdict)

    df = pd.DataFrame(features)
    y = np.array([1 if v == "Y" else 0 for v in verdicts])

    numeric_cols = [c for c in df.columns if c != "event_idx" and df[c].dtype.kind in "fiu"]

    auc_rows = []
    fisher_rows = []
    for c in numeric_cols:
        vals = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        try:
            a = float(roc_auc_score(y, vals))
        except ValueError:
            a = 0.5
        auc_rows.append(
            {
                "feature": c,
                "auc": a,
                "separation": max(a, 1.0 - a),
            }
        )
        fisher_rows.append(
            {
                "feature": c,
                "fisher_log": _fisher_ratio(vals, y),
            }
        )

    auc_rows.sort(key=lambda x: -x["separation"])
    fisher_rows.sort(key=lambda x: -x["fisher_log"])

    mask_fp = y == 0
    fp_features = df.loc[mask_fp, numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    fp_idx = df.loc[mask_fp, "event_idx"].tolist()

    clusters = []
    if len(fp_features) >= 8:
        scaler = StandardScaler()
        X = scaler.fit_transform(fp_features.values)
        for k in (3, 4, 5):
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels_k = km.fit_predict(X)
            profile = []
            for ci in range(k):
                idxs = labels_k == ci
                n = int(idxs.sum())
                if n == 0:
                    continue
                means = fp_features.values[idxs].mean(axis=0)
                sd = fp_features.values.std(axis=0) + 1e-9
                z = (means - fp_features.values.mean(axis=0)) / sd
                top = np.argsort(-np.abs(z))[:5]
                profile.append(
                    {
                        "cluster": ci,
                        "size": n,
                        "top_features": [
                            {
                                "feature": numeric_cols[int(t)],
                                "mean_in_cluster": float(means[int(t)]),
                                "z_vs_all_fp": float(z[int(t)]),
                            }
                            for t in top
                        ],
                        "member_event_idx": [int(e) for e in np.array(fp_idx)[idxs][:30]],
                    }
                )
            clusters.append(
                {
                    "k": k,
                    "inertia": float(km.inertia_),
                    "profiles": profile,
                }
            )

    out = {
        "n_events": len(df),
        "n_Y": int((y == 1).sum()),
        "n_N": int((y == 0).sum()),
        "baseline_precision": float((y == 1).sum() / len(y)),
        "mean_Y": df[y == 1][numeric_cols].mean().to_dict(),
        "mean_N": df[y == 0][numeric_cols].mean().to_dict(),
        "auc_ranking": auc_rows,
        "fisher_ranking": fisher_rows,
        "fp_clusters": clusters,
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    return OUT


if __name__ == "__main__":
    p = main()
    print(f"wrote {p}  ({p.stat().st_size} bytes)")
