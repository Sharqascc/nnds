
from pathlib import Path

def test_demo_script_exists():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_pipeline.py"
    assert script.exists(), "scripts/run_pipeline.py not found"
