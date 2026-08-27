
from pathlib import Path
import subprocess, sys

def test_estimator_script_exists():
    assert Path("scripts/estimate_time_of_day.py").exists()

def test_estimator_returns_label_without_vlm():
    # Without VLM dependencies, it should print 'unknown' and not crash.
    result = subprocess.run(
        [sys.executable, "scripts/estimate_time_of_day.py", "--video", "data/sample_data/traffic_video.mp4"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert result.stdout.strip() in {"morning", "evening", "unknown"}
