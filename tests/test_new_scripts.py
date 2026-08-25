import subprocess, sys, json, os
from pathlib import Path
import pandas as pd

repo = Path(__file__).resolve().parents[1]

def test_experiment_logger_generates_json(tmp_path):
    # Create minimal CSV files
    det = tmp_path / "det.csv"
    pet = tmp_path / "pet.csv"
    pd.DataFrame({
        'frame': [0], 'track_id': [1], 'class_name': ['car'], 'conf': [0.9],
        'x1': [10], 'y1': [20], 'x2': [50], 'y2': [70], 'cx': [30], 'cy': [45], 'source': ['uvh26']
    }).to_csv(det, index=False)
    pd.DataFrame({
        'event_id': [0], 'pet': [1.5], 'frame': [10], 'track_a': [1], 'track_b': [2]
    }).to_csv(pet, index=False)

    log_path = tmp_path / "log.json"
    result = subprocess.run(
        [sys.executable, "scripts/experiment_logger.py",
         "--detections", str(det), "--pet", str(pet), "--output", str(log_path)],
        cwd=repo, capture_output=True, text=True
    )
    assert result.returncode == 0
    assert log_path.exists()
    log = json.loads(log_path.read_text())
    assert log["detections"]["rows"] == 1
    assert log["pet"]["rows"] == 1

def test_anonymize_video_script_exists():
    assert (repo / "scripts" / "anonymize_video.py").exists()
