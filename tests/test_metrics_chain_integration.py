"""Integration: detection -> tracking -> trajectory metrics chain.

Runs in PR CI.
"""

from __future__ import annotations

import numpy as np


def test_detection_metrics_public_api():
    from src.analysis import detection_metrics

    names = [n for n in dir(detection_metrics) if not n.startswith("_")]
    assert any("iou" in n.lower() or "match" in n.lower() or "map" in n.lower() for n in names), (
        names
    )


def test_tracking_metrics_public_api():
    from src.analysis import tracking_metrics

    names = [n for n in dir(tracking_metrics) if not n.startswith("_")]
    assert any("mota" in n.lower() or "motp" in n.lower() or "track" in n.lower() for n in names), (
        names
    )


def test_traj_error_public_api():
    """traj_error exposes velocity/acceleration error metric helpers.

    Regression guard: the module's public names are `*_metrics` and
    primitive trajectory functions, not `*error*` / `*ade*` / `*fde*`.
    Previous substring check never matched and had been failing on the
    integration branch since before PR #50.
    """
    from src.analysis import traj_error

    names = [n for n in dir(traj_error) if not n.startswith("_")]
    expected = {"velocity_metrics", "acceleration_metrics", "trajectory_metrics"}
    missing = expected - set(names)
    assert not missing, f"traj_error missing public helpers: {sorted(missing)}. Actual: {names}"


def test_metrics_modules_are_importable_together():
    from src.analysis import detection_metrics, tracking_metrics, traj_error

    assert detection_metrics is not None
    assert tracking_metrics is not None
    assert traj_error is not None


def test_traj_error_module_has_numeric_dtype_expectations():
    """traj_error should reference numpy numeric types (no I/O side effects)."""
    from src.analysis import traj_error

    text = Path(traj_error.__file__).read_text()
    assert "np" in text or "numpy" in text


from pathlib import Path
