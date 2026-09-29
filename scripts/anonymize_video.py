#!/usr/bin/env python3
"""
Anonymize a video by applying heavy Gaussian blur to the entire frame.
This is a simple privacy protection measure for demonstration videos.

Usage:
    python scripts/anonymize_video.py --input data/sample_data/traffic_video.mp4 --output data/sample_data/anonymized_traffic_video.mp4
"""

import argparse

import cv2


def anonymize_frame(frame, kernel_size=51):
    return cv2.GaussianBlur(frame, (kernel_size | 1, kernel_size | 1), 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kernel", type=int, default=51, help="Gaussian kernel size (odd number)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        blurred = anonymize_frame(frame, args.kernel)
        out.write(blurred)
        frame_count += 1

    cap.release()
    out.release()
    print(f"✅ Anonymized video saved to {args.output} ({frame_count} frames)")


if __name__ == "__main__":
    main()
