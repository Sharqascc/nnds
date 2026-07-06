#!/usr/bin/env bash
set -e

echo "[bootstrap] Pulling latest code..."
git pull --ff-only

echo "[bootstrap] Installing runtime deps..."
pip install -U ultralytics

echo "[bootstrap] Recreating local data dirs..."
mkdir -p videos videos_trim outputs checkpoints

echo "[bootstrap] Reminder:"
echo " - Re-upload videos/traffic_video.mp4"
echo " - Re-upload or re-download sam3.pt into ./sam3.pt"
echo "Then you can run:"
echo "  python traffic_analyzer.py --video videos_trim/traffic_video_30f.mp4 \\"
echo "    --sam3-weights sam3.pt --out-csv outputs/petevents_bev_30f.csv --pet-threshold 2.0"
