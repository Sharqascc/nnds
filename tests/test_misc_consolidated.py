"""Consolidated small/miscellaneous tests."""
from pathlib import Path

import numpy as np

from src.core.validation import compute_error_metrics

repo_root = Path(__file__).resolve().parents[1]


def test_pet_conflict_placeholder():
    assert True


def test_repo_smoke():
    root = Path(".")
    assert (root / "src").exists()
    assert (root / "tests").exists()


def test_event_utility_scripts_exist():
    assert Path("scripts/generate_event_descriptions.py").exists()
    assert Path("scripts/extract_event_frames.py").exists()
    assert Path("scripts/generate_safety_report_groq.py").exists()


def test_bev_error_metrics():
    m = compute_error_metrics([0.1, 0.2, 0.3])
    assert m.num_samples == 3
    assert np.isclose(m.mean_error, 0.2)
    assert np.isclose(m.max_error, 0.3)


def test_sensitivity_analysis_script_exists():
    assert (repo_root / "scripts" / "sensitivity_pet_fragmentation.py").exists()


def test_mot_metrics_placeholder_removed():
    assert not (repo_root / "scripts" / "evaluate_mot_metrics.py").exists()
