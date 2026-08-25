#!/usr/bin/env python3
"""
Debug tracking by drawing all detection boxes with track IDs on the video.

Usage:
    python scripts/debug_tracking_video.py \
        --csv outputs/petevents_bev_300_stricter_split_detections.csv \
        --video data/sample_data/traffic_video.mp4 \
        --start 0 --end 150 \
        --output outputs/tracking_debug_0_150.mp4
"""

import argparse

import cv2
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--video", default="data/sample_data/traffic_video.mp4")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=150)
    parser.add_argument("--output", default="outputs/tracking_debug.mp4")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.2,
        help="Only show boxes with confidence above this",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.csv)
    df = df[(df["frame"] >= args.start) & (df["frame"] <= args.end)]
    if df.empty:
        print("No detections in range")
        return

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"Creating debug video: frames {args.start}-{args.end}, detections={len(df)}")

    for frame_idx in range(args.start, args.end + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        frame_dets = df[df["frame"] == frame_idx]
        for _, det in frame_dets.iterrows():
            if det.get("conf", 1.0) < args.conf:
                continue

            x1, y1, x2, y2 = map(
                int, [det.get("x1"), det.get("y1"), det.get("x2"), det.get("y2")]
            )
            track_id = int(det.get("track_id", -1))
            cls = det.get("class_name", "?")
            conf = det.get("conf", 0.0)

            color = (0, 255, 0) if cls in ("pedestrian", "person") else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{track_id} {cls} {conf:.2f}"
            cv2.putText(
                frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        cv2.putText(
            frame,
            f"Frame {frame_idx}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"✅ Saved {args.output}")


if __name__ == "__main__":
    main()
