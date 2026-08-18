
from pathlib import Path

def test_demo_script_exists():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "traffic_analyzer_demo.py"
    assert script.exists(), "scripts/traffic_analyzer_demo.py not found"
