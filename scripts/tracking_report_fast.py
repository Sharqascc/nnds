#!/usr/bin/env python3
"""Fast tracking report for large datasets."""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-gap", type=int, default=10)
    parser.add_argument("--max-distance", type=float, default=50.0)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df["frame"] = pd.to_numeric(df["frame"])
    df["track_id"] = pd.to_numeric(df["track_id"])
    df["cx"] = pd.to_numeric(df["cx"])
    df["cy"] = pd.to_numeric(df["cy"])

    # Precompute per-track summary once
    track_info = {}
    for tid, grp in tqdm(df.groupby("track_id"), desc="Summarising tracks", unit="track"):
        grp = grp.sort_values("frame")
        frames = grp["frame"].values
        cx = grp["cx"].values
        cy = grp["cy"].values
        jumps = np.sqrt(np.diff(cx)**2 + np.diff(cy)**2) if len(grp) > 1 else np.array([])
        max_gap = int(np.diff(frames).max()) if len(grp) > 1 else 0
        cls_counts = grp["class_name"].value_counts()
        main_cls = cls_counts.index[0] if len(cls_counts) else "unknown"
        track_info[tid] = {
            "start": int(frames.min()),
            "end": int(frames.max()),
            "det": len(grp),
            "cls": main_cls,
            "max_gap": max_gap,
            "max_jump": float(jumps.max()) if len(jumps) else 0.0,
            "avg_jump": float(jumps.mean()) if len(jumps) else 0.0,
            "first": grp.iloc[0],
            "last": grp.iloc[-1],
        }

    # Per-track metrics table
    lines = ["="*80, "FAST TRACKING REPORT", f"Input: {args.csv}", "="*80]
    lines.append(f"Unique tracks: {len(track_info)}")
    lines.append("")
    lines.append("PER-TRACK METRICS (top 20 by detections)")
    lines.append("-"*80)
    lines.append(f"{'track':<8}{'class':<12}{'start':<8}{'end':<8}{'det':<6}{'max_gap':<9}{'max_jump':<10}{'avg_jump':<10}")
    top = sorted(track_info.items(), key=lambda x: x[1]["det"], reverse=True)[:20]
    for tid, s in top:
        lines.append(f"{tid:<8}{s['cls']:<12}{s['start']:<8}{s['end']:<8}{s['det']:<6}{s['max_gap']:<9}{s['max_jump']:<10.2f}{s['avg_jump']:<10.2f}")
    lines.append("")

    # ID switch candidates
    lines.append("CANDIDATE ID SWITCHES")
    lines.append("-"*80)
    switches = []
    tids = list(track_info.keys())
    # Precompute start/end arrays
    starts = np.array([track_info[t]["start"] for t in tids])
    ends = np.array([track_info[t]["end"] for t in tids])
    classes = [track_info[t]["cls"] for t in tids]

    for i in tqdm(range(len(tids)), desc="Checking ID switches", unit="track"):
        tid_a = tids[i]
        s_a = track_info[tid_a]
        for j in range(len(tids)):
            if i == j:
                continue
            tid_b = tids[j]
            s_b = track_info[tid_b]
            gap = s_b["start"] - s_a["end"]
            if 0 <= gap <= args.max_gap and s_a["cls"] == s_b["cls"]:
                dist = np.sqrt((s_a["last"]["cx"]-s_b["first"]["cx"])**2 + (s_a["last"]["cy"]-s_b["first"]["cy"])**2)
                if dist < args.max_distance:
                    switches.append((tid_a, tid_b, gap, dist, s_a["cls"]))

    if switches:
        for sw in switches:
            lines.append(f"Track {sw[0]} -> Track {sw[1]}: gap={sw[2]} frames, dist={sw[3]:.1f}px, class={sw[4]}")
        lines.append(f"Total candidates: {len(switches)}")
    else:
        lines.append("No candidates found.")

    lines.append("")
    lines.append("SUMMARY")
    lines.append("-"*80)
    lines.append(f"Total tracks: {len(track_info)}")
    lines.append(f"ID switch candidates: {len(switches)}")
    suspicious = sum(1 for s in track_info.values() if s['max_gap'] > args.max_gap or s['max_jump'] > 80)
    lines.append(f"Suspicious tracks: {suspicious}")
    lines.append("="*80)

    report = "\n".join(lines)
    print(report)
    if args.output:
        Path(args.output).write_text(report)
        print(f"\n✅ Saved to {args.output}")

if __name__ == "__main__":
    main()
