#!/usr/bin/env python3
"""
Generate annotated verification videos for PET events.

Usage:
    python scripts/generate_pet_verification_video.py \
        --pet-csv outputs/giti_merged_for_visualization.csv \
        --video data/sample_data/anonymized_traffic_video_50f.mp4 \
        --output-dir outputs/verification_videos
"""
import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.visualization.pet_verification_visualizer import PETVerificationVisualizer


def generate_videos(pet_csv, video_path, output_dir, event_ids=None, fps=10):
    """Generate annotated videos for specified event IDs (or all if None)."""
    visualizer = PETVerificationVisualizer(pet_csv, video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if event_ids is None:
        event_ids = visualizer.df['event_id'].unique().tolist()

    paths = []
    for event_id in event_ids:
        out_path = output_dir / f"event_{event_id:04d}.mp4"
        visualizer.generate_video(event_id, str(out_path), fps=fps)
        paths.append(out_path)
        print(f"✅ Event {event_id}: {out_path}")
    return paths


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PET verification videos")
    parser.add_argument("--pet-csv", required=True, help="Path to merged PET CSV with trajectories")
    parser.add_argument("--video", required=True, help="Path to source video")
    parser.add_argument("--output-dir", default="outputs/verification_videos", help="Output directory")
    parser.add_argument("--event-id", type=int, action="append", help="Event ID (can be repeated)")
    parser.add_argument("--fps", type=float, default=10.0, help="Output video FPS")
    return parser.parse_args()


def main():
    args = parse_args()
    generate_videos(
        pet_csv=args.pet_csv,
        video_path=args.video,
        output_dir=args.output_dir,
        event_ids=args.event_id,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
