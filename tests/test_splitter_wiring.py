import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.grid_trajectory.uvh_coco_fused_grid_pet import (
    TrackPoint,
    _split_tracks_by_gaps,
)


def test_splitter_threshold_changes_output():
    # Create a track with a large gap and jump
    pts = [
        TrackPoint(frame=0, x=0, y=0, cls_id=0, cls_name='car', conf=0.9),
        TrackPoint(frame=1, x=1, y=1, cls_id=0, cls_name='car', conf=0.9),
        TrackPoint(frame=20, x=100, y=100, cls_id=0, cls_name='car', conf=0.9),
        TrackPoint(frame=21, x=101, y=101, cls_id=0, cls_name='car', conf=0.9),
    ]
    tracks = {1: pts}

    # Strict thresholds -> should split
    strict = _split_tracks_by_gaps(tracks, max_frame_gap=5, max_spatial_jump=30.0)
    # Loose thresholds -> should not split
    loose = _split_tracks_by_gaps(tracks, max_frame_gap=100, max_spatial_jump=200.0)

    assert len(strict) == 2, f"Expected 2 split tracks with strict thresholds, got {len(strict)}"
    assert len(loose) == 1, f"Expected 1 track with loose thresholds, got {len(loose)}"
