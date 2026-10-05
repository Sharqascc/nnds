"""5-fold cross-validation of a pre-specified PET filter rule.

Rule (fixed a priori, not tuned):
    reject if min_concurrent_dist_m > 5.0 m
    reject if pet_s > 2.0 s
    else: keep

Run:  python -m paper.analysis.filter_cv

Reads:  outputs/giti_screened_with_gates.csv
        data/reviews/ssm_review_114/to_label.csv
        data/reviews/ssm_review_114/label_to_giti_mapping.csv
Writes: paper/results/filter_cv.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[2]
SCREENED = REPO / "outputs" / "giti_screened_with_gates.csv"
LABELS = REPO / "data" / "reviews" / "ssm_review_114" / "to_label.csv"
MAPPING = REPO / "data" / "reviews" / "ssm_review_114" / "label_to_giti_mapping.csv"
OUT = REPO / "paper" / "results" / "filter_cv.json"

FPS = 30.0
MAX_CONCURRENT_DIST_M = 5.0
MAX_PET_S = 2.0
N_FOLDS = 5
N_BOOT = 2000
SEED = 42


def _track_points(js: str) -> dict:
    out = {}
    try:
        pts = json.loads(js)
    except (json.JSONDecodeError, TypeError):
        return out
    for p in pts:
        out[int(p["frame"])] = (float(p["world_x"]), float(p["world_y"]))
    return out


def _min_concurrent_dist(ta: dict, tb: dict) -> float:
    common = set(ta) & set(tb)
    if not common:
        return float("inf")
    return float(min(
        np.hypot(ta[f][0] - tb[f][0], ta[f][1] - tb[f][1])
        for f in common
    ))


def _load_events() -> pd.DataFrame:
    screened = pd.read_csv(SCREENED)
    labels = pd.read_csv(LABELS)
    mapping = pd.read_csv(MAPPING)

    idx_to_verdict = dict(zip(labels.idx, labels.verdict, strict=True))
    giti_to_label = {}
    for _, row in mapping.iterrows():
        giti_to_label[int(row.giti_idx)] = int(row.label_idx)

    rows = []
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
        rows.append({
            "event_idx": li,
            "y_true": 1 if verdict == "Y" else 0,
            "min_concurrent_dist_m": _min_concurrent_dist(ta, tb),
            "pet_s": float(row.pet),
        })
    return pd.DataFrame(rows)


def _apply_rule(df: pd.DataFrame) -> np.ndarray:
    keep = (df["min_concurrent_dist_m"] <= MAX_CONCURRENT_DIST_M) & \
           (df["pet_s"] <= MAX_PET_S)
    return keep.values.astype(int)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1}


def _bootstrap_ci(values: list, n_boot: int, rng: np.random.Generator) -> dict:
    arr = np.array(values)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(sample.mean()))
    means = np.array(means)
    return {"mean": float(arr.mean()),
            "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5))}


def main() -> Path:
    df = _load_events()
    y_true = df["y_true"].values
    y_pred = _apply_rule(df)

    baseline = _metrics(y_true, np.ones_like(y_true))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_prec, fold_rec, fold_f1 = [], [], []
    fold_results = []
    for i, (tr_idx, te_idx) in enumerate(skf.split(df, y_true)):
        te = df.iloc[te_idx]
        yt = te["y_true"].values
        yp = _apply_rule(te)
        m = _metrics(yt, yp)
        fold_prec.append(m["precision"])
        fold_rec.append(m["recall"])
        fold_f1.append(m["f1"])
        fold_results.append({"fold": i, **m})

    rng = np.random.default_rng(SEED)
    ci_prec = _bootstrap_ci(fold_prec, N_BOOT, rng)
    ci_rec = _bootstrap_ci(fold_rec, N_BOOT, rng)
    ci_f1 = _bootstrap_ci(fold_f1, N_BOOT, rng)

    filtered = _metrics(y_true, y_pred)

    out = {
        "rule": {
            "max_concurrent_dist_m": MAX_CONCURRENT_DIST_M,
            "max_pet_s": MAX_PET_S,
            "note": "pre-specified, not tuned on this data",
        },
        "n_events": int(len(df)),
        "n_positive": int((y_true == 1).sum()),
        "baseline_all_kept": baseline,
        "filtered_all_events": filtered,
        "cv": {
            "n_folds": N_FOLDS,
            "fold_results": fold_results,
            "precision_ci95": ci_prec,
            "recall_ci95": ci_rec,
            "f1_ci95": ci_f1,
        },
    }
    OUT.write_text(json.dumps(out, indent=2))
    return OUT


if __name__ == "__main__":
    p = main()
    print(f"wrote {p}  ({p.stat().st_size} bytes)")
