#!/usr/bin/env python3
"""
Deterministic per-track tracking assessment and summary.

For each track in a detections CSV, computes:
  - dominant class and class consistency
  - frame gap statistics (max, median, count > threshold)
  - spatial jump statistics (max, median, 95th percentile, count > threshold)
  - displacement rate (pixels/frame) instead of raw speed
  - quality flag: CLEAN, SPLIT_REQUIRED, CLASS_UNSTABLE, LOW_CONFIDENCE

Outputs:
  1. Detailed log (--log-output, default: outputs/tracking_full_assessment.log)
  2. Summary CSV (--summary-csv, default: outputs/tracking_summary.csv)

Usage:
  python scripts/tracking_assessment.py --detections outputs/tracking_e2e_300_detections.csv
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True)
    parser.add_argument("--log-output", default="outputs/tracking_full_assessment.log")
    parser.add_argument("--summary-csv", default="outputs/tracking_summary.csv")
    parser.add_argument("--max-gap", type=int, default=10)
    parser.add_argument("--max-jump", type=float, default=50.0)
    args = parser.parse_args()

    det_path = Path(args.detections)
    if not det_path.exists():
        print("Detections file not found:", det_path)
        return

    det = pd.read_csv(det_path)
    if det.empty:
        print("No detections.")
        return

    log_lines = []
    log_lines.append("Tracking full assessment log generated at " + str(pd.Timestamp.now().isoformat()))
    log_lines.append("Input: " + str(det_path))
    log_lines.append("Total detections: " + str(len(det)))
    log_lines.append("Unique tracks: " + str(det['track_id'].nunique()))
    log_lines.append("=" * 80)

    summary_rows = []

    for track_id, group in det.groupby('track_id'):
        group = group.sort_values('frame')
        frames = group['frame'].values
        xs = group['cx'].values
        ys = group['cy'].values
        confs = group['conf'].values
        classes = group['class_name'].values

        n_det = len(group)
        first_frame = int(frames[0])
        last_frame = int(frames[-1])
        mean_conf = float(np.mean(confs))

        # Class consistency
        class_counts = pd.Series(classes).value_counts()
        dominant_class = class_counts.index[0]
        dominant_ratio = float(class_counts.iloc[0] / n_det)
        class_switches = 0
        for i in range(1, n_det):
            if classes[i] != classes[i-1]:
                class_switches += 1

        # Gaps and jumps
        gaps = np.diff(frames)
        dx = np.diff(xs)
        dy = np.diff(ys)
        jumps = np.sqrt(dx**2 + dy**2)

        max_gap = int(gaps.max()) if len(gaps) else 0
        median_gap = float(np.median(gaps)) if len(gaps) else 0
        gaps_over = int((gaps > args.max_gap).sum()) if len(gaps) else 0

        max_jump = float(jumps.max()) if len(jumps) else 0.0
        median_jump = float(np.median(jumps)) if len(jumps) else 0.0
        p95_jump = float(np.percentile(jumps, 95)) if len(jumps) else 0.0
        jumps_over = int((jumps > args.max_jump).sum()) if len(jumps) else 0

        # Displacement rate = jump / gap (pixels per frame)
        with np.errstate(divide='ignore', invalid='ignore'):
            rates = jumps / np.maximum(gaps, 1)
        max_rate = float(np.nanmax(rates)) if len(rates) else 0.0
        median_rate = float(np.nanmedian(rates)) if len(rates) else 0.0

        # Quality flag
        flags = []
        if class_switches > 0 or dominant_ratio < 0.9:
            flags.append("CLASS_UNSTABLE")
        if max_gap > args.max_gap:
            flags.append("SPLIT_REQUIRED")
        if mean_conf < 0.5:
            flags.append("LOW_CONFIDENCE")
        if not flags:
            flags.append("CLEAN")
        quality = ",".join(flags)

        # Log lines
        log_lines.append("")
        log_lines.append("Track ID: " + str(track_id))
        log_lines.append("  Detections: " + str(n_det))
        log_lines.append("  Frame range: " + str(first_frame) + " -> " + str(last_frame))
        log_lines.append("  Mean confidence: " + f"{mean_conf:.3f}")
        log_lines.append("  Dominant class: " + dominant_class)
        log_lines.append("  Class consistency: " + f"{dominant_ratio*100:.1f}%")
        log_lines.append("  Class switches: " + str(class_switches))
        log_lines.append("  Max frame gap: " + str(max_gap) + " | gaps > " + str(args.max_gap) + ": " + str(gaps_over))
        log_lines.append("  Max spatial jump: " + f"{max_jump:.1f} px | jumps > " + str(args.max_jump) + ": " + str(jumps_over))
        log_lines.append("  Displacement rate (px/frame): median=" + f"{median_rate:.2f}, max=" + f"{max_rate:.2f}")
        log_lines.append("  Quality: " + quality)

        summary_rows.append({
            "track_id": track_id,
            "detections": n_det,
            "first_frame": first_frame,
            "last_frame": last_frame,
            "mean_conf": round(mean_conf,3),
            "dominant_class": dominant_class,
            "class_consistency_pct": round(dominant_ratio*100,1),
            "class_switches": class_switches,
            "max_gap": max_gap,
            "median_gap": median_gap,
            "gaps_over_threshold": gaps_over,
            "max_jump_px": round(max_jump,1),
            "median_jump_px": round(median_jump,1),
            "p95_jump_px": round(p95_jump,1),
            "jumps_over_threshold": jumps_over,
            "median_disp_rate_px_per_frame": round(median_rate,2),
            "max_disp_rate_px_per_frame": round(max_rate,2),
            "quality": quality,
        })

    # Write log
    log_path = Path(args.log_output)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log_lines))
    print("Full tracking log written to", log_path)

    # Write summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_path = Path(args.summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print("Summary CSV written to", summary_path)

    # Print summary stats
    print("\n=== SUMMARY ===")
    print("Total tracks:", len(summary_df))
    print("Quality distribution:")
    print(summary_df['quality'].value_counts())
    print("\nTracks with class switches:", (summary_df['class_switches']>0).sum())
    print("Tracks with gaps > threshold:", (summary_df['gaps_over_threshold']>0).sum())
    print("Tracks with jumps > threshold:", (summary_df['jumps_over_threshold']>0).sum())


if __name__ == "__main__":
    main()
