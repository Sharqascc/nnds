#!/usr/bin/env python3
"""
Visualize a PET event from a detailed PET CSV.

Draws the two tracks on the actual video frames, frame-by-frame,
with start/end markers and direction arrows.

Usage:
    python scripts/visualize_pet.py --csv outputs/petevents_bev_full.csv --event-id 0
    python scripts/visualize_pet.py --csv outputs/petevents_bev_full.csv --event-id 0 --output outputs/event_0.mp4
"""

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="outputs/petevents_bev_full.csv")
    parser.add_argument("--event-id", type=int, required=True)
    parser.add_argument("--video", default="data/sample_data/traffic_video.mp4")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def draw_tracks(frame, traj_points, color, label):
    """Draw a track with points, connecting lines, start/end markers, and direction arrows."""
    if not traj_points:
        return

    # Sort by frame
    traj_points = sorted(traj_points, key=lambda p: p["frame"])

    # Draw lines connecting consecutive points
    for i in range(1, len(traj_points)):
        p1 = (int(traj_points[i - 1]["x_pixel"]), int(traj_points[i - 1]["y_pixel"]))
        p2 = (int(traj_points[i]["x_pixel"]), int(traj_points[i]["y_pixel"]))
        cv2.line(frame, p1, p2, color, 2)

    # Draw points
    for p in traj_points:
        cv2.circle(frame, (int(p["x_pixel"]), int(p["y_pixel"])), 3, color, -1)

    # Start marker (green square)
    first = traj_points[0]
    cv2.rectangle(
        frame,
        (int(first["x_pixel"]) - 5, int(first["y_pixel"]) - 5),
        (int(first["x_pixel"]) + 5, int(first["y_pixel"]) + 5),
        (0, 255, 0),
        2,
    )

    # End marker (blue square)
    last = traj_points[-1]
    cv2.rectangle(
        frame,
        (int(last["x_pixel"]) - 5, int(last["y_pixel"]) - 5),
        (int(last["x_pixel"]) + 5, int(last["y_pixel"]) + 5),
        (255, 0, 0),
        2,
    )

    # Direction arrow from first to second point
    if len(traj_points) >= 2:
        p1 = (int(first["x_pixel"]), int(first["y_pixel"]))
        p2 = (int(traj_points[1]["x_pixel"]), int(traj_points[1]["y_pixel"]))
        cv2.arrowedLine(frame, p1, p2, color, 2, tipLength=0.3)


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if args.event_id >= len(df) or args.event_id < 0:
        raise ValueError(f"event-id {args.event_id} out of range (0-{len(df) - 1})")

    row = df.iloc[args.event_id]
    traj_a = json.loads(row["traj_a_json"])
    traj_b = json.loads(row["traj_b_json"])

    # Determine frame window
    all_frames = [p["frame"] for p in traj_a] + [p["frame"] for p in traj_b]
    start_frame = min(all_frames)
    end_frame = max(all_frames)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    # Prepare video writer if output specified
    writer = None
    if args.output:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"Visualizing Event {args.event_id}: frames {start_frame}-{end_frame}")
    print(f"Track A: {row['track_a']} | Track B: {row['track_b']}")

    for frame_idx in range(start_frame, end_frame + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Filter points for this frame and previous frames (to draw entire track so far)
        traj_a_so_far = [p for p in traj_a if p["frame"] <= frame_idx]
        traj_b_so_far = [p for p in traj_b if p["frame"] <= frame_idx]

        draw_tracks(frame, traj_a_so_far, (0, 0, 255), f"Track {row['track_a']}")  # Red
        draw_tracks(frame, traj_b_so_far, (255, 0, 0), f"Track {row['track_b']}")  # Blue

        # Add legend
        cv2.putText(
            frame,
            f"Event {args.event_id}  Track {row['track_a']} (red) vs Track {row['track_b']} (blue)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        if writer:
            writer.write(frame)

    cap.release()
    if writer:
        writer.release()
        print(f"✅ Saved video to {args.output}")
    else:
        print("No output file specified; frames not saved.")


if __name__ == "__main__":
    main()
