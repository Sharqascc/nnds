#!/usr/bin/env python3
"""
Extract representative frames (before, closest, after) for each PET event.

Usage:
    python scripts/extract_event_frames.py --pet-csv outputs/e2e_validation_pet.csv --video data/sample_data/traffic_video.mp4 --output-dir outputs/event_frames
"""
import argparse
from pathlib import Path

import cv2
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pet-csv", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", default="outputs/event_frames")
    args = parser.parse_args()

    pet = pd.read_csv(args.pet_csv)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Could not open video")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in pet.iterrows():
        frame_conflict = int(row['frame'])
        event_id = row['event_id']
        event_dir = out_dir / f"event_{event_id:03d}"
        event_dir.mkdir(exist_ok=True)
        for offset, label in [(-5, 'before'), (0, 'closest'), (5, 'after')]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_conflict + offset))
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(str(event_dir / f"{label}.jpg"), frame)

    cap.release()
    print(f"✅ Saved event frames to {out_dir}")

if __name__ == "__main__":
    main()
