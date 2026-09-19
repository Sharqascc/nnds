"""Metamorphic tests for SSM metrics.

Input transformations with known output transformations. Do not need
ground truth; suited to this pipeline because real GT is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.metamorphic


@pytest.mark.parametrize("shift", [0.5, 1.0, 10.0, 100.0])
def test_pet_invariant_under_common_time_shift(shift):
    from src.analysis.grid_trajectory.pet_grid import (
        Interval,
        WorldSample,
        compute_pet,
    )

    ws = [WorldSample(t=0.0, x=0.0, y=0.0)]
    a = Interval(obj_id=1, cell_id="G", t_enter=0.0 + shift, t_exit=1.0 + shift, world_samples=ws)
    b = Interval(obj_id=2, cell_id="G", t_enter=2.0 + shift, t_exit=3.0 + shift, world_samples=ws)
    ev = compute_pet([a, b], pet_threshold=10.0)
    assert len(ev) == 1
    assert ev[0].pet == pytest.approx(1.0)


@pytest.mark.parametrize("dx,dy", [(10, 0), (0, 10), (-5, 3), (100, -200)])
def test_pet_invariant_under_common_spatial_translation(dx, dy):
    from src.analysis.grid_trajectory.pet_grid import (
        Interval,
        WorldSample,
        compute_pet,
    )

    def shift(ws):
        return [WorldSample(t=s.t, x=s.x + dx, y=s.y + dy) for s in ws]

    ws = [WorldSample(t=0.0, x=0.0, y=0.0)]
    a = Interval(obj_id=1, cell_id="G", t_enter=0.0, t_exit=1.0, world_samples=shift(ws))
    b = Interval(obj_id=2, cell_id="G", t_enter=2.0, t_exit=3.0, world_samples=shift(ws))
    ev = compute_pet([a, b], pet_threshold=10.0)
    assert len(ev) == 1
    assert ev[0].pet == pytest.approx(1.0)


@pytest.mark.parametrize("s", [0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 1000.0])
def test_iou_scale_invariant(s):
    from src.analysis.detection_metrics import iou

    b1 = (0.0, 0.0, 10.0, 10.0)
    b2 = (5.0, 5.0, 15.0, 15.0)
    sb1 = tuple(v * s for v in b1)
    sb2 = tuple(v * s for v in b2)
    assert iou(b1, b2) == pytest.approx(iou(sb1, sb2))


def test_map_is_one_when_predictions_match_ground_truth():
    from src.analysis.detection_metrics import (
        Detection,
        GroundTruth,
        map_at_iou_range,
    )

    dets, gts = [], []
    for f in range(5):
        for k in range(3):
            box = (float(k * 20), 0.0, float(k * 20 + 15), 15.0)
            dets.append(Detection(frame=f, box=box, cls="car", conf=0.9))
            gts.append(GroundTruth(frame=f, box=box, cls="car"))
    m = map_at_iou_range(dets, gts)
    assert m["mAP50"] == pytest.approx(1.0)
    assert m["mAP50:95"] == pytest.approx(1.0)


def test_tracker_produces_single_id_under_center_jitter():
    from src.pipeline.custom_tracker import CustomTracker, Detection

    rng = np.random.default_rng(0)
    tracker = CustomTracker(max_age=10, min_hits=1, iou_threshold=0.2)
    assigned: set[int] = set()
    for f in range(20):
        jx, jy = float(rng.normal(0, 1.5)), float(rng.normal(0, 1.5))
        cx, cy = 100 + f * 3 + jx, 100 + jy
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
        for tid in tracker.update([det], frame=f).values():
            assigned.add(tid)
    assert len(assigned) == 1, f"jitter created {len(assigned)} tracks: {assigned}"
