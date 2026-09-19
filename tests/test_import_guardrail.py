"""Tests for scripts/check_imports.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHECKER = REPO / "scripts" / "check_imports.py"


def _run(paths: list[Path] | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(CHECKER)]
    if paths:
        cmd += [str(p) for p in paths]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)


def test_checker_exists_and_is_executable():
    assert CHECKER.exists()
    assert CHECKER.stat().st_mode & 0o111


def test_checker_passes_on_current_tree():
    r = _run()
    assert r.returncode == 0, f"guardrail fails on the tree:\n{r.stderr}"
    assert "clean" in r.stdout


def test_canonical_pipeline_forbidden_from_importing_diffusion():
    f = REPO / "src" / "analysis" / "_test_violation.py"
    try:
        f.write_text("from src.diffusion.complete_ddpm import LinearNoiseScheduler\n")
        r = _run([f])
        assert r.returncode == 1, f"expected violation:\n{r.stdout}\n{r.stderr}"
        assert "src.diffusion" in r.stderr
    finally:
        f.unlink(missing_ok=True)


def test_canonical_pipeline_forbidden_from_importing_vlm():
    f = REPO / "src" / "analysis" / "_test_violation.py"
    try:
        f.write_text("from src.vlm.analyzer import VLLMAnalyzer\n")
        r = _run([f])
        assert r.returncode == 1
        assert "src.vlm" in r.stderr
    finally:
        f.unlink(missing_ok=True)


def test_dynamic_import_also_caught():
    f = REPO / "src" / "analysis" / "_test_violation.py"
    try:
        f.write_text("import importlib\nm = importlib.import_module('src.vlm.analyzer')\n")
        r = _run([f])
        assert r.returncode == 1, f"dynamic import not caught:\n{r.stdout}\n{r.stderr}"
        assert "src.vlm" in r.stderr
    finally:
        f.unlink(missing_ok=True)


def test_leaf_layer_forbidden_from_importing_pipeline():
    f = REPO / "src" / "utils" / "_test_violation.py"
    try:
        f.write_text("from src.pipeline.traffic_analyzer import main\n")
        r = _run([f])
        assert r.returncode == 1, f"expected violation:\n{r.stdout}\n{r.stderr}"
        assert "src.pipeline" in r.stderr
    finally:
        f.unlink(missing_ok=True)


def test_leaf_layer_core_is_also_checked():
    """Regression: shallow files under a leaf layer must be inspected.

    The initial guardrail used fnmatch, where **/ means one-or-more path
    segments. src/core/**/*.py silently failed to match
    src/core/__init__.py and similar top-level files, so the rule was
    unenforced for the majority of the tree.
    """
    f = REPO / "src" / "core" / "_test_violation.py"
    try:
        f.write_text("from src.pipeline.traffic_analyzer import main\n")
        r = _run([f])
        assert r.returncode == 1, f"shallow core file not checked:\n{r.stdout}\n{r.stderr}"
        assert "src.pipeline" in r.stderr
    finally:
        f.unlink(missing_ok=True)


def test_shallow_bev_file_is_checked():
    """Same regression, expressed for src/bev (top-level files)."""
    f = REPO / "src" / "bev" / "_test_violation.py"
    try:
        f.write_text("from src.diffusion.complete_ddpm import LinearNoiseScheduler\n")
        r = _run([f])
        assert r.returncode == 1, f"shallow bev file not checked:\n{r.stdout}\n{r.stderr}"
        assert "src.diffusion" in r.stderr
    finally:
        f.unlink(missing_ok=True)


def test_nested_bev_file_is_checked():
    """Nested files under src/bev/ are also inspected."""
    f = REPO / "src" / "bev" / "calibration" / "_test_violation.py"
    try:
        f.write_text("from src.vlm.analyzer import VLLMAnalyzer\n")
        r = _run([f])
        assert r.returncode == 1
        assert "src.vlm" in r.stderr
    finally:
        f.unlink(missing_ok=True)


def test_allowlisted_file_is_not_flagged():
    f = REPO / "src" / "analysis" / "safety_eval_diffusion.py"
    assert f.exists(), "allowlisted file missing; update RULES if renamed"
    r = _run([f])
    assert r.returncode == 0, f"allowlisted file was flagged:\n{r.stderr}"


def test_unrelated_layer_passes():
    f = REPO / "src" / "diffusion" / "_test_ok.py"
    try:
        f.write_text(
            "from src.vlm.analyzer import VLLMAnalyzer\n"
            "from src.pipeline.traffic_analyzer import main\n"
        )
        r = _run([f])
        assert r.returncode == 0, f"unexpected violation:\n{r.stderr}"
    finally:
        f.unlink(missing_ok=True)
