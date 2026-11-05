"""Entry-point import hygiene.

Guards against accidental heavy imports at module scope on the CLI entry
points. If someone adds `import torch` (or ultralytics/matplotlib/sklearn)
to src/pipeline/traffic_analyzer.py or scripts/run_pipeline.py, these tests
fail in the fast CI lane before any other test has a chance to import them.

Uses subprocess so we observe a fresh interpreter: sys.modules inside pytest
is already polluted by other tests that legitimately import torch.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

HEAVY = ("torch", "ultralytics", "matplotlib", "sklearn")


def _run_probe(code: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + existing if existing else "")
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO),
    )


def test_import_traffic_analyzer_does_not_load_heavy_modules():
    code = (
        "import sys; "
        "import src.pipeline.traffic_analyzer; "
        "heavy = [m for m in " + repr(HEAVY) + " if m in sys.modules]; "
        "print('HEAVY=' + ','.join(heavy))"
    )
    r = _run_probe(code)
    assert r.returncode == 0, f"probe failed: {r.stderr}"
    heavy = r.stdout.strip().split("HEAVY=", 1)[1]
    assert heavy == "", f"traffic_analyzer pulled in heavy modules at import: {heavy}"


def test_import_run_pipeline_is_side_effect_free():
    code = (
        "import scripts.run_pipeline as rp; "
        "assert hasattr(rp, 'main'), 'main() missing'; "
        "assert callable(rp.main), 'main not callable'; "
        "print('OK')"
    )
    r = _run_probe(code)
    assert r.returncode == 0, f"probe failed: {r.stderr}"
    assert "OK" in r.stdout


def test_run_pipeline_module_does_not_load_heavy_modules():
    code = (
        "import sys; "
        "import scripts.run_pipeline; "
        "heavy = [m for m in " + repr(HEAVY) + " if m in sys.modules]; "
        "print('HEAVY=' + ','.join(heavy))"
    )
    r = _run_probe(code)
    assert r.returncode == 0, f"probe failed: {r.stderr}"
    heavy = r.stdout.strip().split("HEAVY=", 1)[1]
    assert heavy == "", f"run_pipeline pulled in heavy modules at import: {heavy}"
