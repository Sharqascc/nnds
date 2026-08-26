#!/usr/bin/env python3
"""
Compute standard MOT metrics (MOTA, IDF1, HOTA) if ground truth tracks are available.

Requires:
    pip install motmetrics
    pip install trackeval

Usage:
    python scripts/evaluate_mot_metrics.py --tracked outputs/tracked.csv --ground-truth tests/fixtures/gt_tracks.csv
"""
import argparse
import sys

try:
    import motmetrics as mm
    print("motmetrics available")
except ImportError:
    print("motmetrics not installed. Please run: pip install motmetrics")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracked", required=True)
    parser.add_argument("--ground-truth", required=True)
    args = parser.parse_args()

    # Placeholder: actual implementation requires format conversion.
    # This script structure shows where to integrate TrackEval/MOTChallenge evaluation.
    print("MOT metrics evaluation requires TrackEval or motmetrics with proper format.")
    print("Please refer to https://github.com/JonathonLuiten/TrackEval for setup.")
    print("We recommend using TrackEval for MOTA, IDF1, HOTA on your dataset.")

if __name__ == "__main__":
    main()
