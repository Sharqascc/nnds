"""Regression: TrackPoint y must be the box bottom edge, not the box center.

The pipeline stores trajectory points at the box bottom-center so BEV
projection, grid-cell assignment, and PET all refer to the ground-contact
point rather than the visual center of the detection box.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PIPELINE = REPO / "src" / "analysis" / "grid_trajectory" / "uvh_coco_fused_grid_pet.py"


def test_trackpoint_uses_y2_not_cy():
    text = PIPELINE.read_text()
    # Locate the TrackPoint construction inside the frame loop
    assert "y=det.y2," in text, "TrackPoint must use det.y2 (box bottom), not det.cy"
    # Make sure the old form does not appear anywhere in the file
    assert "y=det.cy," not in text, "found stale y=det.cy; anchor must be det.y2"


def test_trackpoint_x_still_center():
    text = PIPELINE.read_text()
    assert "x=det.cx," in text, "TrackPoint x must remain det.cx (bottom-center anchor)"
