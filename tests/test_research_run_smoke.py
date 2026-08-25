import subprocess
import sys
from pathlib import Path


def test_cli_help():
    """Smoke test: run the pipeline with --help and check it works."""
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "run_pipeline.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


def test_pipeline_module_exists():
    """Smoke test: ensure the main pipeline module exists."""
    repo_root = Path(__file__).resolve().parents[1]
    pipeline_file = repo_root / "src" / "pipeline" / "traffic_analyzer.py"
    assert pipeline_file.exists()
