"""Bit-for-bit determinism. Guards against unseeded RNG and races."""

from __future__ import annotations

import pytest


def _run_tracker(seed: int, n_frames: int = 20):
    from src.pipeline.custom_tracker import CustomTracker, Detection
    from src.utils.seed import set_seed

    set_seed(seed)
    import numpy as np

    rng = np.random.default_rng(seed)
    tracker = CustomTracker(max_age=10, min_hits=1, iou_threshold=0.2)
    events = []
    for f in range(n_frames):
        cx = 100 + f * 3 + float(rng.normal(0, 1.0))
        cy = 100 + float(rng.normal(0, 1.0))
        det = Detection(
            frame=f,
            x1=cx - 20,
            y1=cy - 20,
            x2=cx + 20,
            y2=cy + 20,
            cx=cx,
            cy=cy,
            cls_id=0,
            cls_name="car",
            conf=0.9,
            source="test",
        )
        events.append(tuple(sorted(tracker.update([det], frame=f).items())))
    return events


def test_tracker_bitwise_deterministic_under_seed():
    assert _run_tracker(seed=42) == _run_tracker(seed=42)


def test_ssm_metrics_deterministic():
    from src.analysis.ssm_error import SsmEvent, pet_value_metrics

    pred = [SsmEvent(1, 2, pet=1.5), SsmEvent(3, 4, pet=2.0)]
    gt = [SsmEvent(1, 2, pet=1.4), SsmEvent(3, 4, pet=2.2)]
    assert pet_value_metrics(pred, gt) == pet_value_metrics(pred, gt)


def test_agreement_metrics_deterministic():
    import numpy as np

    from src.analysis.ssm_agreement import agreement_metrics
    from src.analysis.ssm_error import SsmEvent

    rng = np.random.default_rng(7)
    pred = [SsmEvent(i, i + 1, pet=float(rng.uniform(0.5, 3))) for i in range(1, 20)]
    gt = [SsmEvent(i, i + 1, pet=float(rng.uniform(0.5, 3))) for i in range(1, 20)]
    assert agreement_metrics(pred, gt, "pet") == agreement_metrics(pred, gt, "pet")


# Note: there is intentionally no "different seed -> different output" test.
# The tracker is a Kalman filter and is fully deterministic given the same
# detections; seed variation changes the detections but not the tracker's
# identity assignment for small jitter. The seed-invariance test above
# confirms the tracker is reproducible; the pipeline-wide nondeterminism
# (device selection, cudnn) is tracked separately in
# docs/GITI_EVAL_STATUS.md.
