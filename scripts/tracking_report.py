#!/usr/bin/env python3
"""
Automatic tracking evaluation & report.

Reads a detections CSV with columns:
  frame, track_id, class_name, cx, cy, x1, y1, x2, y2, source

Produces a structured text report with:
  - overall statistics
  - per-track metrics (duration, detections, jumps, gaps)
  - overlapping track pairs
  - candidate ID switches

Usage:
  python scripts/tracking_report.py --csv outputs/petevents_bev_300_custom_detections.csv
  python scripts/tracking_report.py --csv outputs/... --output outputs/tracking_report.txt
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Detection CSV path")
    parser.add_argument("--output", default=None, help="Output report path (optional)")
    parser.add_argument("--max-gap", type=int, default=10, help="Max frame gap for ID switch candidate")
    parser.add_argument("--max-distance", type=float, default=50.0, help="Max center distance for ID switch candidate")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"frame", "track_id", "class_name", "cx", "cy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df["frame"] = pd.to_numeric(df["frame"])
    df["track_id"] = pd.to_numeric(df["track_id"])
    df["cx"] = pd.to_numeric(df["cx"])
    df["cy"] = pd.to_numeric(df["cy"])

    lines = []
    lines.append("=" * 80)
    lines.append("TRACKING EVALUATION REPORT")
    lines.append(f"Input: {csv_path}")
    lines.append("=" * 80)
    lines.append(f"Total frames: {df['frame'].nunique()}")
    lines.append(f"Total detections: {len(df)}")
    lines.append(f"Unique tracks: {df['track_id'].nunique()}")
    lines.append("")

    track_stats = {}
    for tid, grp in df.groupby("track_id"):
        grp = grp.sort_values("frame")
        frames = grp["frame"].values
        cx = grp["cx"].values
        cy = grp["cy"].values

        if len(grp) >= 2:
            dx = np.diff(cx)
            dy = np.diff(cy)
            jumps = np.sqrt(dx*dx + dy*dy)
            max_jump = float(jumps.max())
            avg_jump = float(jumps.mean())
            gaps = np.diff(frames)
            max_gap = int(gaps.max()) if len(gaps) > 0 else 0
        else:
            max_jump = 0.0
            avg_jump = 0.0
            max_gap = 0

        cls_counts = grp["class_name"].value_counts().to_dict()
        main_cls = max(cls_counts, key=cls_counts.get) if cls_counts else "unknown"

        track_stats[tid] = {
            "start": int(frames.min()),
            "end": int(frames.max()),
            "num_det": len(grp),
            "main_class": main_cls,
            "max_gap": max_gap,
            "max_jump": max_jump,
            "avg_jump": avg_jump,
        }

    lines.append("PER-TRACK METRICS (top 30 by detections)")
    lines.append("-" * 80)
    lines.append(f"{'track':<8}{'class':<12}{'start':<8}{'end':<8}{'det':<6}{'max_gap':<9}{'max_jump':<10}{'avg_jump':<10}")
    lines.append("-" * 80)
    sorted_tracks = sorted(track_stats.items(), key=lambda x: x[1]["num_det"], reverse=True)
    for tid, s in sorted_tracks[:30]:
        lines.append(f"{tid:<8}{s['main_class']:<12}{s['start']:<8}{s['end']:<8}{s['num_det']:<6}{s['max_gap']:<9}{s['max_jump']:<10.2f}{s['avg_jump']:<10.2f}")
    lines.append("")

    lines.append("OVERLAPPING TRACK PAIRS (spatial + temporal)")
    lines.append("-" * 80)
    overlap_count = 0
    track_ids = list(track_stats.keys())
    for i in tqdm(range(len(track_ids)), desc="Checking overlaps", unit="track"):
        for j in range(i+1, len(track_ids)):
            tid_a = track_ids[i]
            tid_b = track_ids[j]
            grp_a = df[df["track_id"] == tid_a]
            grp_b = df[df["track_id"] == tid_b]
            frames_a = set(grp_a["frame"].unique())
            frames_b = set(grp_b["frame"].unique())
            common = frames_a & frames_b
            if not common:
                continue
            min_dist = float('inf')
            for f in common:
                pts_a = grp_a[grp_a["frame"] == f][["cx", "cy"]].values
                pts_b = grp_b[grp_b["frame"] == f][["cx", "cy"]].values
                if len(pts_a) == 0 or len(pts_b) == 0:
                    continue
                for pa in pts_a:
                    for pb in pts_b:
                        d = np.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
                        if d < min_dist:
                            min_dist = d
            if min_dist < 50.0:
                lines.append(f"Tracks {tid_a} ({track_stats[tid_a]['main_class']}) and {tid_b} ({track_stats[tid_b]['main_class']}): min center dist {min_dist:.1f}px, common frames {len(common)}")
                overlap_count += 1
    if overlap_count == 0:
        lines.append("No overlapping track pairs detected.")
    lines.append("")

    lines.append("CANDIDATE ID SWITCHES")
    lines.append("-" * 80)
    switches = []
    for tid_a, s_a in tqdm(track_stats.items(), desc="Checking ID switches", unit="track"):
        for tid_b, s_b in track_stats.items():
            if tid_a == tid_b:
                continue
            gap = s_b["start"] - s_a["end"]
            if 0 <= gap <= args.max_gap and s_a["main_class"] == s_b["main_class"]:
                last_a = df[df["track_id"] == tid_a].sort_values("frame").iloc[-1]
                first_b = df[df["track_id"] == tid_b].sort_values("frame").iloc[0]
                dist = np.sqrt((last_a["cx"]-first_b["cx"])**2 + (last_a["cy"]-first_b["cy"])**2)
                if dist < args.max_distance:
                    switches.append((tid_a, tid_b, gap, dist, s_a["main_class"]))
    if switches:
        for sw in switches:
            lines.append(f"Track {sw[0]} -> Track {sw[1]}: gap={sw[2]} frames, dist={sw[3]:.1f}px, class={sw[4]}")
        lines.append(f"Total candidates: {len(switches)}")
    else:
        lines.append("No candidate ID switches detected.")
    lines.append("")

    lines.append("SUMMARY")
    lines.append("-" * 80)
    lines.append(f"Total tracks: {len(track_stats)}")
    lines.append(f"Overlapping pairs: {overlap_count}")
    lines.append(f"ID switch candidates: {len(switches)}")
    suspicious = sum(1 for s in track_stats.values() if s['max_gap'] > args.max_gap or s['max_jump'] > 80)
    lines.append(f"Suspicious tracks (max_gap > {args.max_gap} or max_jump > 80): {suspicious}")
    lines.append("=" * 80)

    report = "\n".join(lines)
    print(report)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(report)
        print("\n✅ Report saved to", out_path)


if __name__ == "__main__":
    main()
