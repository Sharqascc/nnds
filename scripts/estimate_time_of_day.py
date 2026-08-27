#!/usr/bin/env python3
"""
EXPERIMENTAL / FUTURE WORK.
Estimate time of day (morning/evening/unknown) from a video using VLM.

Samples a few frames, sends them to the VLM with a prompt about lighting/shadows,
and returns the majority label.

Usage:
    python scripts/estimate_time_of_day.py --video data/sample_data/traffic_video.mp4 [--max-frames 5]
"""
import argparse
import tempfile
import cv2
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--max-frames", type=int, default=5)
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print("unknown")
        return

    # Try importing VLM; if missing, fallback unknown
    try:
        from src.vlm.analyzer import VLLMAnalyzer
        vlm = VLLMAnalyzer()
    except Exception as e:
        print("unknown")
        return

    # Sample frames evenly
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        print("unknown")
        return
    sample_indices = [int(total * (i + 0.5) / args.max_frames) for i in range(args.max_frames)]
    labels = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            img_path = os.path.join(tmpdir, f"frame_{idx}.jpg")
            cv2.imwrite(img_path, frame)
            prompt = (
                "Based on the lighting, shadows, and sun position in this traffic camera image, "
                "is it likely morning (9-11 AM) or evening (4:30-6:30 PM)? "
                "Answer only with 'morning', 'evening', or 'unknown'."
            )
            try:
                answer = vlm.analyze_image(img_path, prompt).strip().lower()
                if "morning" in answer:
                    labels.append("morning")
                elif "evening" in answer:
                    labels.append("evening")
                else:
                    labels.append("unknown")
            except Exception:
                labels.append("unknown")
    cap.release()

    # Majority vote
    if labels:
        from collections import Counter
        label, _ = Counter(labels).most_common(1)[0]
        print(label)
    else:
        print("unknown")

if __name__ == "__main__":
    main()
