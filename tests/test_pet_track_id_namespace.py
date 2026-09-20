"""Regression: pet.csv and pet_detections.csv share a track-ID namespace.

Historically, _split_tracks_by_gaps produced composite keys of the form
`orig * 1000 + seg` and those composite values were written to `track_a` /
`track_b` in the PET CSV. pet_detections.csv uses raw tracker IDs (small
integers). The two files had no common key, so downstream consumers had
to decode `orig = composite // 1000` and match on that.

After the fix, `track_a` / `track_b` in the PET CSV are the raw tracker
IDs. The split segment index is preserved separately in `seg_a` / `seg_b`,
and `orig_track_a` / `orig_track_b` are retained for backward
compatibility (they now equal `track_a` / `track_b`).
"""

from __future__ import annotations

import pathlib

PIPE = pathlib.Path(__file__).resolve().parents[1] / (
    "src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py"
)


def test_pet_emits_raw_track_ids_not_composite():
    """The PET event dict must write `orig_a` (decoded raw ID) into
    `track_a`, not the composite `first_id`."""
    text = PIPE.read_text()
    assert '"track_a": int(orig_a),' in text, (
        "track_a must be int(orig_a); the composite first_id went into "
        "track_a previously and created a namespace collision with "
        "pet_detections.csv"
    )
    assert '"track_b": int(orig_b),' in text, "track_b must be int(orig_b)"
    assert '"track_a": int(first_id),' not in text, "composite first_id still written to track_a"
    assert '"track_b": int(second_id),' not in text, "composite second_id still written to track_b"


def test_splitter_still_returns_composite_internally():
    """The splitter itself must keep returning composite keys; only the
    PET emission layer unwraps them. Decode contract is preserved."""
    from src.analysis.grid_trajectory.uvh_coco_fused_grid_pet import (
        TrackPoint,
        _split_tracks_by_gaps,
    )

    pts = [
        TrackPoint(frame=i, x=100.0 + i, y=200.0, cls_id=2, cls_name="car", conf=0.9)
        for i in range(10)
    ]
    pts += [
        TrackPoint(frame=i, x=500.0 + i, y=200.0, cls_id=2, cls_name="car", conf=0.9)
        for i in range(20, 25)
    ]
    out = _split_tracks_by_gaps(
        {44: pts}, max_frame_gap=5, max_spatial_jump=30.0, prediction_tolerance=80.0
    )
    assert sorted(out.keys()) == [44000, 44001], (
        f"splitter key contract changed: {sorted(out.keys())}"
    )
    for k in out:
        assert k // 1000 == 44, "orig id must be recoverable via k // 1000"
        assert k % 1000 in (0, 1), "seg index must be k % 1000"
