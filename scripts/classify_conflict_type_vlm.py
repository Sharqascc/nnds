#!/usr/bin/env python3
"""
Optional VLM-based conflict type classifier.

For each PET event in a PET CSV, extracts the conflict frame from the video,
asks a lightweight VLM to classify the conflict type, and outputs a new CSV.

Usage:
    python scripts/classify_conflict_type_vlm.py --pet-csv outputs/petevents.csv --video data/sample_data/traffic_video.mp4 --max-events 10
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import cv2
import tempfile
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vlm.analyzer import VLLMAnalyzer


CONFLICT_TYPES = ["rear-end", "head-on", "crossing", "side-swipe", "other", "unknown"]

def classify_frame(vlm, img_path):
    prompt = "What type of traffic conflict is shown? Answer with one of: rear-end, head-on, crossing, side-swipe, other, unknown."
    ans = vlm.analyze_image(img_path, prompt).strip().lower()
    for t in CONFLICT_TYPES[:-1]:
        if t in ans:
            return t
    return "unknown"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pet-csv", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="outputs/conflict_types_vlm.csv")
    parser.add_argument("--max-events", type=int, default=None)
    args = parser.parse_args()

    pet_path = Path(args.pet_csv)
    if not pet_path.exists():
        print(f"PET CSV not found: {pet_path}")
        return

    pet = pd.read_csv(pet_path)
    if pet.empty:
        print("PET CSV is empty; no events to classify.")
        return

    if args.max_events is not None:
        pet = pet.head(args.max_events)

    print(f"Loading VLM for conflict classification on {len(pet)} events...")
    vlm = VLLMAnalyzer()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Could not open video")
        return

    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, row in pet.iterrows():
            frame_num = int(row.get('frame', -1))
            if frame_num < 0:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                results.append({'event_id': row.get('event_id', idx),
                                'frame': frame_num,
                                'conflict_type_original': row.get('conflict_type', 'image_intersection'),
                                'conflict_type_vlm': 'unknown'})
                continue
            img_path = os.path.join(tmpdir, f"event_{idx}.jpg")
            cv2.imwrite(img_path, frame)
            label = classify_frame(vlm, img_path)
            results.append({'event_id': row.get('event_id', idx),
                            'frame': frame_num,
                            'track_a': row.get('track_a', None),
                            'track_b': row.get('track_b', None),
                            'conflict_type_original': row.get('conflict_type', 'image_intersection'),
                            'conflict_type_vlm': label})
    cap.release()

    if results:
        out_df = pd.DataFrame(results)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"✅ Saved conflict type classifications to {out_path}")
        print(out_df.to_string(index=False))
    else:
        print("No results produced.")

if __name__ == "__main__":
    main()
