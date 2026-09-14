
import pytest
from src.analysis.grid_trajectory.uvh_coco_fused_grid_pet import _track_missing_ratio


@pytest.mark.parametrize(
    "min_f, max_f, n, expected",
    [
        (0, 99, 100, 0.0),
        (0, 99, 90, 0.10),
        (0, 99, 89, 0.11),
        (0, 99, 50, 0.50),
        (5, 5, 1, 0.0),
        (5, 5, 0, 1.0),
        (10, 5, 3, 1.0),
    ],
)
def test_track_missing_ratio(min_f, max_f, n, expected):
    meta = {"min_frame": min_f, "max_frame": max_f}
    assert _track_missing_ratio(meta, n) == pytest.approx(expected)
