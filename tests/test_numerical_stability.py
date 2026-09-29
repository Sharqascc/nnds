"""Numerical stability tests for the metric modules.

Behavioral tests already cover the ordinary cases. These tests target the
boundaries where floating-point arithmetic is most likely to produce a
wrong answer instead of an exception:

  - extreme coordinate magnitudes
  - tiny and huge time steps
  - near-singular homographies
  - non-finite inputs
  - exact boundary values on thresholds and ratios

Each test asserts a property that must hold regardless of magnitude,
not the specific value the code happens to produce today.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from src.analysis.bev_error import (
    homography_error_metrics,
    position_errors,
    position_metrics,
    project_homography,
)
from src.analysis.detection_metrics import ap_at_iou, iou
from src.analysis.pet_conflict_checker import compute_pet
from src.analysis.tracking_metrics import hota, idf1, mota
from src.analysis.traj_error import acceleration, speed, velocity


# =====================================================================
# traj_error: time-step extremes
# =====================================================================
def test_velocity_finite_at_tiny_time_step():
    """Very small dt (high fps) should not overflow to inf."""
    traj = [(0, 0.0, 0.0), (1, 0.001, 0.0), (2, 0.002, 0.0)]
    v = velocity(traj, fps=1e6)  # dt = 1e-6 s
    assert all(np.isfinite(vx) and np.isfinite(vy) for _, vx, vy in v)
    # velocity is (x_next - x_prev) / dt
    # = (0.002 - 0.0) / 2e-6 = 1000 m/s
    # The point: it doesn't become inf or NaN.
    for _, vx, vy in v:
        assert vx == pytest.approx(1000.0, rel=1e-9)


def test_velocity_finite_at_huge_time_step():
    """Very large dt (low fps) should not underflow to exactly 0 in a
    misleading way."""
    traj = [(0, 0.0, 0.0), (1, 100.0, 0.0), (2, 200.0, 0.0)]
    v = velocity(traj, fps=0.001)  # dt = 1000 s
    for _, vx, vy in v:
        assert np.isfinite(vx) and np.isfinite(vy)
        # (200 - 0) / 2000 = 0.1 m/s
        assert vx == pytest.approx(0.1, rel=1e-9)


def test_acceleration_is_exactly_zero_on_perfectly_linear_motion():
    """Constant-velocity data has zero acceleration to machine precision.

    This is the canary: any float-noise amplification in the central-
    difference formula shows up here first.
    """
    # Linear motion: x = 3 * frame
    traj = [(f, 3.0 * f, -2.0 * f) for f in range(10)]
    a = acceleration(traj, fps=30.0)
    assert len(a) > 0
    for _, ax, ay in a:
        assert ax == pytest.approx(0.0, abs=1e-12)
        assert ay == pytest.approx(0.0, abs=1e-12)


def test_speed_is_scale_invariant_under_uniform_scaling():
    """Multiplying all coordinates by k should multiply speed by k."""
    traj = [(f, 0.5 * f, 0.3 * f) for f in range(6)]
    traj_scaled = [(f, 1e6 * 0.5 * f, 1e6 * 0.3 * f) for f in range(6)]
    s1 = [v for _, v in speed(traj, fps=30.0)]
    s2 = [v for _, v in speed(traj_scaled, fps=30.0)]
    # relative error should be at most a few ULP of float64
    for a, b in zip(s1, s2, strict=True):
        if a > 0:
            assert b / a == pytest.approx(1e6, rel=1e-12)


def test_velocity_duplicate_frames_do_not_produce_nan():
    """Two samples at the same frame index -> dt would be zero for the
    interior step. The implementation should skip it, not return NaN."""
    traj = [(0, 0.0, 0.0), (1, 1.0, 0.0), (1, 2.0, 0.0), (2, 3.0, 0.0)]
    v = velocity(traj, fps=30.0)
    for _, vx, vy in v:
        assert np.isfinite(vx) and np.isfinite(vy)


def test_velocity_nan_at_neighbor_propagates_not_swallowed():
    """A NaN at a neighbor frame must produce a NaN velocity, not silently
    compute a plausible-looking finite value.

    Central-difference velocity for frame i uses frames i-1 and i+1 (the
    center frame's own coordinate is not used). So a NaN placed at the
    *center* frame does not affect its own velocity -- that is intended,
    the difference is robust to a missing sample. A NaN at i-1 or i+1 is
    a data integrity problem and must propagate.
    """
    # NaN at the previous frame: interior frame's velocity must be NaN.
    traj_prev_nan = [(0, np.nan, 0.0), (1, 1.0, 0.0), (2, 2.0, 0.0)]
    v = velocity(traj_prev_nan, fps=30.0)
    row_for_frame_1 = [r for r in v if r[0] == 1]
    assert row_for_frame_1, "frame 1 missing from output"
    _, vx, _ = row_for_frame_1[0]
    assert np.isnan(vx), (
        "NaN at i-1 did not propagate to velocity at i; a wrong "
        "finite value is indistinguishable from valid data"
    )

    # Same for NaN at the next frame.
    traj_next_nan = [(0, 0.0, 0.0), (1, 1.0, 0.0), (2, np.nan, 0.0)]
    v = velocity(traj_next_nan, fps=30.0)
    row_for_frame_1 = [r for r in v if r[0] == 1]
    assert row_for_frame_1, "frame 1 missing from output"
    _, vx, _ = row_for_frame_1[0]
    assert np.isnan(vx), "NaN at i+1 did not propagate"


def test_velocity_skips_missing_center_sample():
    """Documented behavior: a NaN at the center frame does not affect that
    frame's own central-difference velocity. Useful robustness -- a single
    dropped detection does not zero out the whole trajectory."""
    traj = [(0, 0.0, 0.0), (1, np.nan, 0.0), (2, 2.0, 0.0)]
    v = velocity(traj, fps=30.0)
    row_for_frame_1 = [r for r in v if r[0] == 1]
    assert row_for_frame_1, "frame 1 missing from output"
    _, vx, _ = row_for_frame_1[0]
    # (x[2] - x[0]) / dt = (2.0 - 0.0) / (2/30) = 30.0
    assert vx == pytest.approx(30.0, rel=1e-12)


# =====================================================================
# bev_error: homography scale extremes
# =====================================================================
def test_homography_identity_at_extreme_coordinates():
    """Identity projection should be exact even at 1e12 magnitude."""
    H = np.eye(3)
    pts = np.array([[1e12, -1e12], [1e12, 1e12]], dtype=np.float64)
    out = project_homography(H, pts)
    assert np.allclose(out, pts, rtol=1e-12)


def test_homography_small_w_raises_not_returns_huge():
    """Points that map to w just above the 1e-12 guard should still
    produce finite coordinates, not 1e12-magnitude garbage."""
    # A homography that shrinks w by a large factor
    H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1e-8]], dtype=np.float64)
    pts = np.array([[1.0, 1.0]], dtype=np.float64)
    out = project_homography(H, pts)
    # w = 1e-8, so out ≈ (1e8, 1e8). That's large but finite.
    assert np.all(np.isfinite(out))
    assert np.all(np.abs(out) < 1e12)


def test_position_errors_symmetric_at_scale():
    """Error metric is symmetric even when coordinates differ by 1e9."""
    a = np.array([[0.0, 0.0], [1e9, 0.0]], dtype=np.float64)
    b = np.array([[0.0, 0.0], [1e9 + 1.0, 0.0]], dtype=np.float64)
    e1 = position_errors(a, b)
    e2 = position_errors(b, a)
    assert np.allclose(e1, e2, rtol=1e-12)
    # The +1.0 offset should be exactly the reported error on row 2
    assert e1[1] == pytest.approx(1.0, rel=1e-6)


# =====================================================================
# detection_metrics: exact boundaries
# =====================================================================
def test_iou_touching_boxes_is_exactly_zero():
    """Boxes that share an edge have zero intersection area. iou must
    return exactly 0.0, not a tiny epsilon."""
    b1 = (0.0, 0.0, 1.0, 1.0)
    b2 = (1.0, 0.0, 2.0, 1.0)
    assert iou(b1, b2) == 0.0


def test_iou_identical_boxes_is_exactly_one():
    """Identical boxes -> intersection == union -> 1.0 exactly.
    A tolerance here would mask a real precision bug."""
    b = (0.0, 0.0, 1.0, 1.0)
    assert iou(b, b) == 1.0


# =====================================================================
# tracking_metrics: exact 0/1 boundaries
# =====================================================================
def test_mota_perfect_tracking_is_exactly_one():
    from src.analysis.tracking_metrics import Track

    # Single box, single track, perfect match
    t = Track(track_id=1, frame=0, box=(0.0, 0.0, 1.0, 1.0))
    assert mota([t], [t]) == 1.0


def test_mota_no_tracks_no_gt_is_exactly_one():
    """Vacuous case: nothing to track, nothing tracked -> MOTA = 1.0."""
    assert mota([], []) == 1.0


def test_idf1_perfect_association_is_exactly_one():
    from src.analysis.tracking_metrics import Track

    t = Track(track_id=1, frame=0, box=(0.0, 0.0, 1.0, 1.0))
    assert idf1([t], [t]) == 1.0


# =====================================================================
# PET: exact boundary on min_valid_pet
# =====================================================================
def test_pet_exactly_min_valid_does_not_warn():
    """The warning fires only for pet < min_valid_pet. At exactly the
    threshold, no warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # passage times give pet = 0.01
        compute_pet([0.0, 0.5], [0.51], min_valid_pet=0.01)
        # only filter for RuntimeWarnings about near-zero PET
        near_zero = [w for w in caught if "Near-zero PET" in str(w.message)]
        assert not near_zero, "warning fired at exactly min_valid_pet (should be strict <)"


def test_pet_below_min_valid_warns():
    """Below threshold -> warning fires."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compute_pet([0.0], [0.005], min_valid_pet=0.01)
        near_zero = [w for w in caught if "Near-zero PET" in str(w.message)]
        assert near_zero, "warning did not fire below min_valid_pet"


def test_pet_simultaneous_passage_returns_exactly_zero():
    """Two actors in the zone at exactly the same time -> PET = 0.0,
    not NaN, not inf."""
    pet = compute_pet([1.0, 2.0], [1.0, 2.0])
    assert pet == 0.0


def test_pet_no_overlap_returns_large_finite_not_inf():
    """Well-separated passage times give a large but finite PET, not inf."""
    pet = compute_pet([0.0], [1000.0])
    assert np.isfinite(pet)
    assert pet == pytest.approx(1000.0, rel=1e-12)


def test_pet_empty_returns_inf():
    """No passage times -> no conflict -> PET = inf (documented)."""
    assert compute_pet([], []) == np.inf
    assert compute_pet([1.0], []) == np.inf


def test_pet_non_finite_raises():
    """NaN or inf in passage times is a data error, must raise."""
    with pytest.raises(ValueError):
        compute_pet([0.0, np.nan], [1.0])
    with pytest.raises(ValueError):
        compute_pet([0.0], [np.inf])
