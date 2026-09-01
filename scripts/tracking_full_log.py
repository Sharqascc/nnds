#!/usr/bin/env python3
"""
Comprehensive tracking assessment via logs (no visual output).

Reads a detections CSV and generates a detailed per-track log.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True)
    parser.add_argument("--output", default="outputs/tracking_full_assessment.log")
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
    log_lines.append('Tracking full assessment log generated at ' + str(pd.Timestamp.now().isoformat()))
    log_lines.append('Input: ' + str(det_path))
    log_lines.append('Total detections: ' + str(len(det)))
    log_lines.append('Unique tracks: ' + str(det['track_id'].nunique()))
    log_lines.append('=' * 80)

    for track_id, group in det.groupby('track_id'):
        group = group.sort_values('frame')
        frames = group['frame'].values
        xs = group['cx'].values
        ys = group['cy'].values
        confs = group['conf'].values
        classes = group['class_name'].values

        log_lines.append('')
        log_lines.append('Track ID: ' + str(track_id))
        log_lines.append('  Detections: ' + str(len(group)))
        log_lines.append('  Frame range: ' + str(frames[0]) + ' -> ' + str(frames[-1]))
        log_lines.append('  Mean confidence: ' + f'{confs.mean():.3f}')
        class_counts = pd.Series(classes).value_counts().to_dict()
        log_lines.append('  Class counts: ' + str(class_counts))

        if len(group) > 1:
            gaps = np.diff(frames)
            dx = np.diff(xs)
            dy = np.diff(ys)
            distances = np.sqrt(dx**2 + dy**2)

            log_lines.append('  Max frame gap: ' + str(gaps.max() if len(gaps) else 0))
            log_lines.append('  Max spatial jump (pixels): ' + f'{distances.max() if len(distances) else 0:.1f}')

            for i, (gap, dist) in enumerate(zip(gaps, distances)):
                if gap > args.max_gap:
                    log_lines.append('    ⚠️ Gap > ' + str(args.max_gap) + ' frames at frame ' + str(frames[i]) + ' -> ' + str(frames[i+1]) + ' (gap=' + str(gap) + ')')
                if dist > args.max_jump:
                    log_lines.append('    ⚠️ Jump > ' + str(args.max_jump) + ' px at frame ' + str(frames[i]) + ' -> ' + str(frames[i+1]) + ' (distance=' + f'{dist:.1f}' + ')')

            speeds = distances / np.maximum(gaps, 1)
            log_lines.append('  Speed (px/frame): min=' + f'{speeds.min():.2f}' + ', mean=' + f'{speeds.mean():.2f}' + ', max=' + f'{speeds.max():.2f}')
        else:
            log_lines.append('  Single detection; no gap/jump/speed analysis possible.')

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    newline = chr(10)  # newline character
    out_path.write_text(newline.join(log_lines))
    print('Full tracking log written to ' + str(out_path))
    print('Total tracks logged: ' + str(det['track_id'].nunique()))

if __name__ == "__main__":
    main()