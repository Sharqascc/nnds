"""Analytical tests for time-to-collision safety calculations."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.safety_eval_diffusion import compute_ttc_seq


def test_approaching_head_on_returns_positive_ttc():
    vehicle_a = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    vehicle_b = np.array([[10.0, 0.0], [9.0, 0.0], [8.0, 0.0]])

    ttc = compute_ttc_seq(vehicle_a, vehicle_b, dt=1.0, d_thresh=1.0)

    assert ttc == pytest.approx([5.0, 4.0])


def test_ttc_decreases_as_vehicles_approach():
    vehicle_a = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0]])
    vehicle_b = np.array([[20.0, 0.0], [18.0, 0.0], [16.0, 0.0], [14.0, 0.0]])

    ttc = compute_ttc_seq(vehicle_a, vehicle_b, dt=1.0, d_thresh=1.0)

    assert ttc == pytest.approx([5.0, 4.0, 3.0])


def test_separating_vehicles_return_none():
    vehicle_a = np.array([[0.0, 0.0], [1.0, 0.0]])
    vehicle_b = np.array([[10.0, 0.0], [11.0, 0.0]])

    ttc = compute_ttc_seq(vehicle_a, vehicle_b, dt=1.0, d_thresh=1.0)

    assert ttc == [None]


def test_close_but_separating_vehicles_return_none():
    vehicle_a = np.array([[0.0, 0.0], [1.0, 0.0]])
    vehicle_b = np.array([[0.5, 0.0], [1.5, 0.0]])

    ttc = compute_ttc_seq(vehicle_a, vehicle_b, dt=1.0, d_thresh=1.0)

    assert ttc == [None]


def test_parallel_same_velocity_returns_none():
    vehicle_a = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    vehicle_b = np.array([[0.0, 5.0], [1.0, 5.0], [2.0, 5.0]])

    ttc = compute_ttc_seq(vehicle_a, vehicle_b, dt=1.0, d_thresh=1.0)

    assert ttc == [None, None]


def test_zero_relative_velocity_returns_none():
    vehicle_a = np.array([[0.0, 0.0], [0.0, 0.0]])
    vehicle_b = np.array([[5.0, 0.0], [5.0, 0.0]])

    ttc = compute_ttc_seq(vehicle_a, vehicle_b, dt=1.0, d_thresh=1.0)

    assert ttc == [None]


def test_perpendicular_crossing_returns_expected_ttc():
    vehicle_a = np.array([[-2.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])
    vehicle_b = np.array([[0.0, -2.0], [0.0, -1.0], [0.0, 0.0]])

    ttc = compute_ttc_seq(vehicle_a, vehicle_b, dt=1.0, d_thresh=1.0)

    assert ttc == pytest.approx([2.0, 1.0])


def test_no_collision_when_closest_approach_exceeds_threshold():
    vehicle_a = np.array([[0.0, 0.0], [1.0, 0.0]])
    vehicle_b = np.array([[10.0, 10.0], [9.0, 10.0]])

    ttc = compute_ttc_seq(vehicle_a, vehicle_b, dt=1.0, d_thresh=0.1)

    assert ttc == [None]


def test_valid_ttc_values_are_finite_and_positive():
    vehicle_a = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    vehicle_b = np.array([[8.0, 0.0], [7.0, 0.0], [6.0, 0.0]])

    ttc = compute_ttc_seq(vehicle_a, vehicle_b, dt=1.0, d_thresh=1.0)

    assert all(value is not None for value in ttc)
    assert all(np.isfinite(value) and value > 0 for value in ttc)


def test_threshold_controls_collision_detection():
    vehicle_a = np.array([[0.0, 0.0], [1.0, 0.0]])
    vehicle_b = np.array([[10.0, 0.5], [9.0, 0.5]])

    small_threshold = compute_ttc_seq(
        vehicle_a, vehicle_b, dt=1.0, d_thresh=0.1
    )
    large_threshold = compute_ttc_seq(
        vehicle_a, vehicle_b, dt=1.0, d_thresh=1.0
    )

    assert small_threshold == [None]
    assert large_threshold == pytest.approx([5.0])
