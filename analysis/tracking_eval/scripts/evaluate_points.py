
"""Point-based CLEAR-MOT + IDF1 evaluator.

Fixes two issues in the naive version:
  1. ID-switch direction: counts when a GT trajectory's matched pred_id
     changes over time, not the reverse.
  2. IDF1 requires global assignment: build per-pair overlap matrix,
     then solve a second Hungarian to find primary GT<->pred pairs.

Input CSVs: frame, track_id, world_x, world_y
Match rule: per-frame Hungarian on Euclidean distance, cutoff.
"""
import argparse, json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def load(path, xcol="world_x", ycol="world_y"):
    df = pd.read_csv(path)
    for c in ("frame", "track_id", xcol, ycol):
        if c not in df.columns:
            raise ValueError(f"{path}: missing {c}")
    out = df[["frame", "track_id", xcol, ycol]].dropna().copy()
    out.columns = ["frame", "track_id", "x", "y"]
    out["frame"] = out["frame"].astype(int)
    out["track_id"] = out["track_id"].astype(int)
    return out


def per_frame_match(gt, pred, threshold):
    frames = sorted(set(gt.frame) | set(pred.frame))
    matches = []   # (frame, gt_id, pred_id, dist)
    gt_present = {f: [] for f in frames}
    pred_present = {f: [] for f in frames}
    for f in frames:
        g = gt[gt.frame == f]
        p = pred[pred.frame == f]
        gt_present[f] = list(g.track_id)
        pred_present[f] = list(p.track_id)
        if len(g) == 0 or len(p) == 0:
            continue
        G = g[["x","y"]].values.astype(float)
        P = p[["x","y"]].values.astype(float)
        D = np.linalg.norm(G[:,None,:] - P[None,:,:], axis=2)
        BIG = 1e9
        D_cost = np.where(D <= threshold, D, BIG)
        row, col = linear_sum_assignment(D_cost)
        for r, c in zip(row, col):
            if D[r, c] <= threshold:
                matches.append((f, int(g.iloc[r].track_id),
                                  int(p.iloc[c].track_id), float(D[r,c])))
    return matches, gt_present, pred_present


def evaluate(gt, pred, threshold):
    matches, gt_present, pred_present = per_frame_match(gt, pred, threshold)
    frames = sorted(set(gt_present) | set(pred_present))

    # per-frame matching counts
    total_gt_frames = sum(len(gt_present[f]) for f in frames)
    total_pred_frames = sum(len(pred_present[f]) for f in frames)
    TP = len(matches)
    FN = total_gt_frames - TP
    FP = total_pred_frames - TP

    # ---- ID switches (gt-centric) ----
    # for each GT track: walk its frames, track which pred_id is assigned.
    # IDSW when the assigned pred_id changes between consecutive matched frames
    # (missing frames do not reset — reappearing with a new id IS a switch).
    # Frag = a gap in matched frames without pred_id change.
    by_gt = {}   # gt_id -> list of (frame, pred_id or None)
    for f in frames:
        for gid in gt_present[f]:
            by_gt.setdefault(gid, []).append([f, None])
    match_lookup = {}
    for f, gid, pid, d in matches:
        match_lookup[(f, gid)] = pid
    for gid, lst in by_gt.items():
        for entry in lst:
            entry[1] = match_lookup.get((entry[0], gid))

    id_switches = 0
    fragmentation = 0
    for gid, timeline in by_gt.items():
        last_pid = None
        in_gap = False
        for f, pid in timeline:
            if pid is None:
                in_gap = True
                continue
            # any gap in matches = one fragmentation
            if in_gap:
                fragmentation += 1
            # any change of assigned pred_id = one ID switch
            if last_pid is not None and pid != last_pid:
                id_switches += 1
            last_pid = pid
            in_gap = False

    # ---- IDF1 (global assignment) ----
    # overlap matrix: rows = gt_id, cols = pred_id, value = match count
    gt_ids = sorted(by_gt.keys())
    pred_ids = sorted(set(pred.track_id))
    g2i = {g: i for i, g in enumerate(gt_ids)}
    p2i = {p: j for j, p in enumerate(pred_ids)}
    O = np.zeros((len(gt_ids), len(pred_ids)), dtype=int)
    for f, gid, pid, _ in matches:
        O[g2i[gid], p2i[pid]] += 1
    # maximize overlap -> minimize -O
    if O.size:
        gi, pi = linear_sum_assignment(-O)
        IDTP = int(O[gi, pi].sum())
    else:
        IDTP = 0
    # leftover pred matches (matched to GT but not primary) = IDFP
    IDFN = total_gt_frames - IDTP
    IDFP = total_pred_frames - IDTP
    denom = 2 * IDTP + IDFP + IDFN
    idf1 = (2 * IDTP) / denom if denom else 0.0

    # MOTA = 1 - (FN + FP + IDSW) / total_gt_frames
    mota = 1.0 - (FN + FP + id_switches) / total_gt_frames if total_gt_frames else 0.0

    return {
        "threshold_m": float(threshold),
        "GT_frames": int(total_gt_frames),
        "PRED_frames": int(total_pred_frames),
        "TP": int(TP),
        "FP": int(FP),
        "FN": int(FN),
        "ID_switches": int(id_switches),
        "fragmentations": int(fragmentation),
        "IDTP": int(IDTP),
        "IDFP": int(IDFP),
        "IDFN": int(IDFN),
        "IDF1": float(idf1),
        "MOTA": float(mota),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--threshold", type=float, default=1.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gt = load(a.gt)
    pred = load(a.pred)
    print(f"GT:   {len(gt)} pts  {gt.track_id.nunique()} tracks")
    print(f"Pred: {len(pred)} pts  {pred.track_id.nunique()} tracks")
    r = evaluate(gt, pred, a.threshold)
    print(json.dumps(r, indent=2))
    with open(a.out, "w") as f:
        json.dump(r, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
