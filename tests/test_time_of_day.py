import subprocess
import sys
from pathlib import Path


def test_estimator_script_exists():
    assert Path("scripts/estimate_time_of_day.py").exists()


def test_estimator_returns_label_without_vlm():
    # Without VLM dependencies, it should print a valid label on the last line.
    result = subprocess.run(
        [
            sys.executable,
            "scripts/estimate_time_of_day.py",
            "--video",
            "data/sample_data/traffic_video.mp4",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    output_lines = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    assert output_lines, "No output"
    label = output_lines[-1]
    assert label in {"morning", "evening", "unknown"}
