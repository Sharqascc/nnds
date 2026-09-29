import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_demo_end_to_end(tmp_path):
    out_dir = tmp_path / "e2e"
    result = subprocess.run(
        [sys.executable, "scripts/demo_end_to_end.py", "--out-dir", str(out_dir)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Gold-standard rows: 15" in result.stdout

    # All four JSONs exist and are well-formed
    for name in ("detection", "tracking", "trajectory", "ssm"):
        p = out_dir / f"{name}.json"
        assert p.exists(), f"missing {p}"
        data = json.loads(p.read_text())
        assert isinstance(data, dict)
        assert data, f"{p} is empty"

    # Report exists and has 15 data rows + header + separator
    report = out_dir / "validation_report.md"
    assert report.exists()
    lines = report.read_text().splitlines()
    data_rows = [
        ln
        for ln in lines
        if ln.startswith("| ") and "---" not in ln and not ln.startswith("| Metric")
    ]
    assert len(data_rows) == 15
