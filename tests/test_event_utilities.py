
from pathlib import Path


def test_event_utility_scripts_exist():
    assert Path("scripts/generate_event_descriptions.py").exists()
    assert Path("scripts/extract_event_frames.py").exists()
    assert Path("scripts/generate_safety_report_groq.py").exists()
