import subprocess
import sys
from pathlib import Path

import pandas as pd

repo = Path(__file__).resolve().parents[1]


def test_detection_metrics_script(tmp_path):
    # Create simple GT and detections
    gt = pd.DataFrame(
        {
            "frame": [0, 0],
            "x1": [10, 30],
            "y1": [20, 40],
            "x2": [50, 70],
            "y2": [80, 100],
            "class_name": ["car", "pedestrian"],
        }
    )
    det = pd.DataFrame(
        {
            "frame": [0, 0],
            "x1": [12, 32],
            "y1": [22, 42],
            "x2": [52, 72],
            "y2": [82, 102],
            "class_name": ["car", "pedestrian"],
            "conf": [0.9, 0.8],
        }
    )
    gt_path = tmp_path / "gt.csv"
    det_path = tmp_path / "det.csv"
    gt.to_csv(gt_path, index=False)
    det.to_csv(det_path, index=False)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_detection_metrics.py",
            "--detections",
            str(det_path),
            "--ground-truth",
            str(gt_path),
        ],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "mAP@50" in result.stdout


def test_tracking_metrics_script(tmp_path):
    gt = pd.DataFrame(
        {
            "frame": [0, 1],
            "track_id": [1, 1],
            "x": [10, 11],
            "y": [20, 21],
            "w": [30, 30],
            "h": [40, 40],
        }
    )
    trk = pd.DataFrame(
        {
            "frame": [0, 1],
            "track_id": [1, 1],
            "x": [10, 11],
            "y": [20, 21],
            "w": [30, 30],
            "h": [40, 40],
        }
    )
    gt_path = tmp_path / "gt_tracks.csv"
    trk_path = tmp_path / "trk_tracks.csv"
    gt.to_csv(gt_path, index=False)
    trk.to_csv(trk_path, index=False)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_tracking_metrics.py",
            "--tracked",
            str(trk_path),
            "--ground-truth",
            str(gt_path),
        ],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "ID switches" in result.stdout
