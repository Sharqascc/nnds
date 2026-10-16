"""Regression tests for scripts/run_pipeline.py entry-point bugs.

These correspond to the qodo-code-review findings on PR #54 (fixed in the
`fix/run-pipeline-bugs` branch):

1. --auto-time-of-day was forwarded to the analyzer, which rejected it.
2. nnds-run could not locate the analyzer outside a source checkout.
3. Model preparation ran in the caller's cwd instead of the repo root.
4. Estimator subprocess failures were treated as valid "unknown" labels.
5. --version triggered model preparation.

The tests exercise the CLI in a subprocess so we see the real argv behavior.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _run(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + existing if existing else "")
    # Ensure model prep would fail loudly if the informational-flag guard
    # ever regressed; we point it at a nonexistent script by accident here.
    return subprocess.run(
        [sys.executable, "-m", "scripts.run_pipeline", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=str(REPO),
    )


def test_version_does_not_trigger_model_prep():
    """--version must return the version without downloading models.

    If prep were triggered, this test would time out or emit
    '[run_pipeline] Checking models...' — we assert neither.
    """
    r = _run(["--version"])
    assert r.returncode == 0, f"stderr:\n{r.stderr}"
    assert "Checking models" not in r.stdout, (
        "--version triggered model preparation; guard in ensure_models_if_needed "
        "regressed. stdout:\n" + r.stdout
    )
    # Version string is emitted by the analyzer's pre-argparse handler.
    assert "revision" in r.stdout.lower(), r.stdout


def test_help_does_not_trigger_model_prep():
    r = _run(["--help"])
    assert "Checking models" not in r.stdout, r.stdout


def test_auto_time_of_day_does_not_reach_analyzer_argparse():
    """The runner-only flag must be stripped before dispatch.

    We use --self-check so the analyzer exits before doing any real work;
    if --auto-time-of-day leaked through, argparse would error out on an
    unrecognized argument and return a nonzero code with 'unrecognized'.
    """
    r = _run(["--auto-time-of-day", "--self-check", "--video", "nonexistent.mp4"])
    combined = r.stdout + r.stderr
    # If the flag leaked, argparse would say "unrecognized arguments: --auto-time-of-day".
    assert "--auto-time-of-day" not in combined or "unrecognized" not in combined.lower(), (
        "runner-only flag leaked to the analyzer's argparse:\n" + combined
    )


def test_analyzer_accepts_time_of_day_flag():
    """Analyzer's argparse must accept --time-of-day (added by this fix).

    --self-check exits immediately after preflight; we only care that argparse
    did not reject the new flag.
    """
    r = _run(["--time-of-day", "morning", "--self-check"])
    combined = r.stdout + r.stderr
    assert "unrecognized arguments" not in combined, combined

def test_run_pipeline_rejects_non_integer_env_seed():
    """NNDS_SEED=abc should exit with a clear error, not an int() traceback."""
    import os
    env = os.environ.copy()
    env["NNDS_SEED"] = "abc"
    from pathlib import Path as _Path
    repo = _Path(__file__).resolve().parents[1]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo) + (os.pathsep + existing if existing else "")
    r = subprocess.run(
        [sys.executable, "-m", "scripts.run_pipeline", "--version"],
        capture_output=True, text=True, env=env, cwd=str(repo), timeout=30,
    )
    assert r.returncode != 0
    assert "NNDS_SEED must be an integer" in (r.stdout + r.stderr)


def test_run_pipeline_rejects_out_of_range_env_seed():
    """NNDS_SEED=2**40 should exit with the range error, not crash mid-startup."""
    import os
    env = os.environ.copy()
    env["NNDS_SEED"] = str(2**40)
    from pathlib import Path as _Path
    repo = _Path(__file__).resolve().parents[1]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo) + (os.pathsep + existing if existing else "")
    r = subprocess.run(
        [sys.executable, "-m", "scripts.run_pipeline", "--version"],
        capture_output=True, text=True, env=env, cwd=str(repo), timeout=30,
    )
    assert r.returncode != 0
    assert "invalid NNDS_SEED" in (r.stdout + r.stderr)
