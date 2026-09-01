import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from baselines.constant_acceleration import constant_acceleration_predict
from baselines.social_force import SocialForceModel


def test_constant_acceleration_predict():
    past = [(0,0,0), (1,1,1), (2,4,4)]  # v=(1,1), a=(2,2)
    future = constant_acceleration_predict(past, 3)
    assert len(future) == 3
    # At t=1: x = 4 + 1*1 + 0.5*2*1 = 4+1+1=6
    assert abs(future[0][1] - 8) < 1e-6
    assert abs(future[0][2] - 8) < 1e-6

def test_social_force_runs():
    model = SocialForceModel()
    pos = np.array([[0,0],[2,0]])
    vel = np.array([[0.5,0],[-0.5,0]])
    dest = np.array([10,0])
    steps = []
    for p in model.predict(pos, vel, dest, 3):
        steps.append(p)
    assert len(steps) == 3
    assert steps[-1].shape == (2,2)
