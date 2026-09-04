import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def test_validate_outputs_help():
    result = subprocess.run(
        [sys.executable, 'scripts/validate_outputs.py', '--help'],
        capture_output=True, text=True, cwd=str(REPO), timeout=30
    )
    assert result.returncode == 0
    assert '--detections' in result.stdout
    assert '--pet' in result.stdout

def test_generate_pet_verification_video_help():
    result = subprocess.run(
        [sys.executable, 'scripts/generate_pet_verification_video.py', '--help'],
        capture_output=True, text=True, cwd=str(REPO), timeout=30
    )
    assert result.returncode == 0
    assert '--pet-csv' in result.stdout
    assert '--video' in result.stdout
