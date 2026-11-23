"""Check the documented positive-FPS constructor contract."""

import pytest

from src.analysis.grid_trajectory.pet_grid import TrajectoryLogger


@pytest.mark.parametrize(
    ("fps", "accepted"),
    [
        pytest.param(-30, False, id="negative-integer"),
        pytest.param(-1.0, False, id="negative-float"),
        pytest.param(-0.001, False, id="negative-fraction"),
        pytest.param(0, False, id="zero-integer"),
        pytest.param(-0.0, False, id="negative-zero"),
        pytest.param(0.001, True, id="positive-fraction"),
        pytest.param(1, True, id="positive-integer"),
        pytest.param(29.97, True, id="positive-float"),
    ],
)
def test_fps_acceptance_table(fps, accepted):
    if not accepted:
        with pytest.raises(ValueError, match="fps must be positive"):
            TrajectoryLogger(fps=fps)
        return

    logger = TrajectoryLogger(fps=fps)
    assert logger.fps == float(fps)
    assert logger.get_stats()["fps"] == float(fps)
