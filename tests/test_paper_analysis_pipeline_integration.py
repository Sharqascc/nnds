"""Integration: every paper/analysis module runs on frozen data.

No heavy assets. Runs in PR CI. Each subprocess runs the module as it
would be invoked by the Paper Analysis workflow.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _skip_if_missing(paths):
    for p in paths:
        if not Path(p).exists():
            pytest.skip(f"artifact missing: {p}")


def _run_module(name: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", f"paper.analysis.{name}"],
        cwd=REPO, capture_output=True, text=True, timeout=180,
    )


def test_lock_denominator_runs():
    _skip_if_missing([
        REPO / "data/reviews/ssm_review_114/to_label.csv",
        REPO / "data/reviews/ssm_review_114/label_to_giti_mapping.csv",
    ])
    r = _run_module("lock_denominator")
    assert r.returncode == 0, r.stderr
    assert (REPO / "paper/results/pet_audit.json").exists()


def test_statistical_tests_runs():
    _skip_if_missing([
        REPO / "outputs/giti_screened_with_gates.csv",
        REPO / "outputs/mrc_screened_with_gates.csv",
    ])
    r = _run_module("statistical_tests")
    assert r.returncode == 0, r.stderr
    assert (REPO / "paper/results/stat_tests.json").exists()


def test_extract_fp_list_runs():
    _skip_if_missing([
        REPO / "data/reviews/ssm_review_114/to_label.csv",
        REPO / "data/reviews/ssm_review_114/label_to_giti_mapping.csv",
    ])
    r = _run_module("extract_fp_list")
    assert r.returncode == 0, r.stderr


def test_tracker_health_runs():
    _skip_if_missing([REPO / "outputs/giti_raw.csv"])
    r = _run_module("tracker_health")
    assert r.returncode == 0, r.stderr


def test_tracking_diagnostics_runs():
    _skip_if_missing([REPO / "outputs/giti_raw.csv"])
    r = _run_module("tracking_diagnostics")
    assert r.returncode == 0, r.stderr


def test_pair_analysis_runs():
    _skip_if_missing([
        REPO / "outputs/giti_screened_with_gates.csv",
        REPO / "data/reviews/ssm_review_114/to_label.csv",
    ])
    r = _run_module("pair_analysis")
    assert r.returncode == 0, r.stderr


def test_filter_cv_runs():
    _skip_if_missing([
        REPO / "outputs/giti_screened_with_gates.csv",
        REPO / "data/reviews/ssm_review_114/to_label.csv",
    ])
    r = _run_module("filter_cv")
    assert r.returncode == 0, r.stderr
