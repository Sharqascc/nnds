import hypothesis.strategies as st
import numpy as np
from hypothesis import given

from src.analysis.grid_trajectory.pet_grid import classify_pet
from src.bev.calibration.monte_carlo_calibration_benchmark import mae_world

# mae_world is defined in monte_carlo_calibration_benchmark but not in pet_grid; use correct module.
# We'll import from src.bev.calibration.monte_carlo_calibration_benchmark


@given(st.lists(st.floats(min_value=0, max_value=10, allow_nan=False), min_size=1, max_size=20))
def test_mae_world_nonnegative(points):
    pred = np.array([[0.0, 0.0] for _ in points])
    gt = np.array([[1.0, 0.0] for _ in points])
    mae = mae_world(pred, gt)
    assert mae >= 0.0


@given(st.floats(min_value=-5, max_value=10, allow_nan=False))
def test_classify_pet_valid(pet):
    label = classify_pet(pet)
    assert label in {"critical", "moderate", "safe"}
