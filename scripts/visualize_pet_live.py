#!/usr/bin/env python3
"""
Live visualizer for PET events.

Instead of drawing the full trajectory as a static overlay, this draws
the current track point (and a short trail) on each frame to show
whether the tracker follows the same object over time.

Usage:
    python scripts/visualize_pet_live.py --csv outputs/petevents_bev_300_stricter_split.csv --event-id 0 --output outputs/event_0_live.mp4
"""

import argparse
import json

import cv2
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--event-id", type=int, required=True)
    parser.add_argument("--video", default="data/sample_data/traffic_video.mp4")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--trail-length",
        type=int,
        default=5,
        help="Number of past points to draw for each track",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    row = df.iloc[args.event_id]

    traj_a = json.loads(row["traj_a_json"])
    traj_b = json.loads(row["traj_b_json"])

    # Convert to dict by frame for fast lookup
    def to_frame_dict(points):
        d = {}
        for p in points:
            d[int(p["frame"])] = (p["x_pixel"], p["y_pixel"])
        return d

    dict_a = to_frame_dict(traj_a)
    dict_b = to_frame_dict(traj_b)

    all_frames = sorted(set([p["frame"] for p in traj_a] + [p["frame"] for p in traj_b]))
    start_frame = min(all_frames)
    end_frame = max(all_frames)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    trail_a = []
    trail_b = []

    print(f"Live visualization: Event {args.event_id}, frames {start_frame}-{end_frame}")

    for frame_idx in range(start_frame, end_frame + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Update trails with current point if available
        if frame_idx in dict_a:
            trail_a.append((frame_idx, dict_a[frame_idx]))
        if frame_idx in dict_b:
            trail_b.append((frame_idx, dict_b[frame_idx]))

        # Keep only last N points
        trail_a = trail_a[-args.trail_length :]
        trail_b = trail_b[-args.trail_length :]

        # Draw trails
        for f, (x, y) in trail_a:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)  # red
        for f, (x, y) in trail_b:
            cv2.circle(frame, (int(x), int(y)), 4, (255, 0, 0), -1)  # blue

        # Draw bigger current marker
        if frame_idx in dict_a:
            x, y = dict_a[frame_idx]
            cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 255), 2)
        if frame_idx in dict_b:
            x, y = dict_b[frame_idx]
            cv2.circle(frame, (int(x), int(y)), 8, (255, 0, 0), 2)

        # Legend
        cv2.putText(
            frame,
            f"Frame {frame_idx}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Red: track {row['track_a']}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Blue: track {row['track_b']}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

        if writer:
            writer.write(frame)

    cap.release()
    if writer:
        writer.release()
        print(f"✅ Saved live video to {args.output}")
    else:
        print("No output file specified")


if __name__ == "__main__":
    main()
