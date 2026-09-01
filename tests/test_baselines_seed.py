import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from baselines.constant_velocity import constant_velocity_predict
from baselines.kalman_filter import SimpleKalmanFilter
from src.utils.seed import set_seed


def test_constant_velocity_predict():
    past = [(0, 0, 0), (1, 1, 2)]
    future = constant_velocity_predict(past, num_future=3)
    assert len(future) == 3
    # Velocity = (1,2) per frame
    assert future[0] == (2, 2, 4)
    assert future[1] == (3, 3, 6)
    assert future[2] == (4, 4, 8)

def test_kalman_filter_runs():
    kf = SimpleKalmanFilter(dt=1.0)
    kf.update([1, 1])
    pred = kf.predict()
    assert pred.shape == (2,)

def test_set_seed_reproducible():
    set_seed(42)
    a = np.random.rand()
    set_seed(42)
    b = np.random.rand()
    assert a == b
