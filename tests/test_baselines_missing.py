import contextlib
import runpy
from io import StringIO

import numpy as np
import pytest

from baselines.constant_acceleration import constant_acceleration_predict
from baselines.constant_velocity import constant_velocity_predict
from baselines.social_force import SocialForceModel


# ---------- constant_velocity ----------
def test_cv_too_few_points():
    with pytest.raises(ValueError):
        constant_velocity_predict([(0, 0, 0)])


def test_cv_non_increasing_frames():
    with pytest.raises(ValueError):
        constant_velocity_predict([(0, 0, 0), (0, 1, 1)])


def test_cv_valid_prediction():
    future = constant_velocity_predict([(0, 0, 0), (1, 1, 2)], num_future=3)
    assert len(future) == 3
    # last frame =1, vx=1, vy=2
    assert future[0] == (2, 2, 4)
    assert future[1] == (3, 3, 6)
    assert future[2] == (4, 4, 8)


def test_cv_main():
    # Run module as __main__ to cover __main__ block
    with contextlib.redirect_stdout(StringIO()):
        runpy.run_module("baselines.constant_velocity", run_name="__main__")


# ---------- constant_acceleration ----------
def test_ca_too_few_points():
    with pytest.raises(ValueError):
        constant_acceleration_predict([(0, 0, 0), (1, 1, 1)])


def test_ca_valid_prediction():
    past = [(0, 0, 0), (1, 1, 1), (2, 4, 4)]
    future = constant_acceleration_predict(past, num_future=2)
    # dt=1, vx=(4-1)/1=3, vy=3
    # ax = (3 - (1-0)/(1))/1 = 2, ay=2
    # t=1: x=4+3+0.5*2=8, y=4+3+0.5*2=8 => (3,8,8)
    # t=2: x=4+6+4=14, y=4+6+4=14 => (4,14,14)
    assert future[0] == (3, 8, 8)
    assert future[1] == (4, 14, 14)


def test_ca_main():
    with contextlib.redirect_stdout(StringIO()):
        runpy.run_module("baselines.constant_acceleration", run_name="__main__")


# ---------- social_force ----------
def test_social_force_basic_prediction():
    model = SocialForceModel()
    pos = np.array([[0, 0], [2, 0]])
    vel = np.array([[0.5, 0], [-0.5, 0]])
    dest = np.array([10, 0])
    gen = model.predict(pos, vel, dest, num_steps=2)
    positions = list(gen)
    assert len(positions) == 2
    assert positions[0].shape == (2, 2)


def test_social_force_close_agents_repulsion():
    model = SocialForceModel()
    pos = np.array([[0, 0], [1, 0]])  # distance 1 < 2, triggers repulsion
    vel = np.array([[0.5, 0], [-0.5, 0]])
    dest = np.array([10, 0])
    gen = model.predict(pos, vel, dest, num_steps=1)
    next_pos = next(gen)
    # Just ensure it runs without error and shape correct
    assert next_pos.shape == (2, 2)


def test_social_force_main():
    with contextlib.redirect_stdout(StringIO()):
        runpy.run_module("baselines.social_force", run_name="__main__")
