from pathlib import Path

repo = Path(__file__).resolve().parents[1]


def test_sensitivity_analysis_script_exists():
    assert (repo / "scripts" / "sensitivity_pet_fragmentation.py").exists()


def test_mot_metrics_placeholder_removed():
    # The placeholder was removed in favor of explicit limitation documentation.
    assert not (repo / "scripts" / "evaluate_mot_metrics.py").exists()
