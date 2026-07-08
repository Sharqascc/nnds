"""
Regression tests for core/gate_counter.py, specifically covering the
hysteresis threshold bug fixed in commit 00cad6e: GATE_HYSTERESIS values
are pixel-scale, and signed_distance() must return raw pixel distance
for those thresholds to correctly filter tracker/detector jitter while
still registering genuine crossings.
"""
import sys
sys.path.insert(0, "/content/nnds_verify")

import pytest
from core.gate_counter import VirtualGate, GATE_HYSTERESIS


def make_gate(name="North_Gate", entry_side="left"):
    # Horizontal-ish gate line, similar scale to a real camera frame.
    return VirtualGate(
        name=name,
        p1=(200, 360),
        p2=(1000, 340),
        entry_side=entry_side,
    )


def test_signed_distance_is_pixel_scale():
    """signed_distance should return raw pixel distance, not a normalized
    fraction -- this is what GATE_HYSTERESIS thresholds assume."""
    gate = make_gate()
    # A point ~50px below the line should return a magnitude on the
    # order of tens of pixels, not a small fraction like 0.06.
    d = gate.signed_distance((600, 400))
    assert abs(d) > 10.0, (
        f"signed_distance magnitude {d} is too small to be pixel-scale; "
        "check for accidental normalization by gate length."
    )


def test_jitter_does_not_trigger_crossing():
    """Small (~4px) jitter around the gate line, simulating detector/tracker
    noise on a stationary or near-stationary object, must NOT register as
    an entry or exit."""
    gate = make_gate()

    # Points straddling the line by only a few pixels.
    positions = [(600, 348), (600, 352), (600, 348), (600, 352)]

    prev = positions[0]
    frame_idx = 0
    for curr in positions[1:]:
        frame_idx += 20  # spaced out so min_frames_between_crossings never blocks it
        status = gate.check_crossing(prev, curr, track_id=1, frame_idx=frame_idx)
        assert status is None, (
            f"jitter from {prev} to {curr} incorrectly registered as {status}"
        )
        prev = curr

    assert gate.entry_count == 0
    assert gate.exit_count == 0


def test_real_crossing_is_detected():
    """A genuine full traversal of the gate (~100px perpendicular movement)
    must register as exactly one crossing event."""
    gate = make_gate()

    prev_pos = (600, 300)  # above the line
    curr_pos = (600, 400)  # below the line

    status = gate.check_crossing(prev_pos, curr_pos, track_id=2, frame_idx=1)

    assert status in ("entry", "exit")
    assert gate.entry_count + gate.exit_count == 1


def test_entry_exit_direction_is_consistent_with_entry_side():
    """Crossing in the same physical direction should be labeled consistently
    for entry_side='left' vs 'right' (the two conventions should be exact
    opposites of each other)."""
    gate_left = make_gate(entry_side="left")
    gate_right = make_gate(entry_side="right")

    prev_pos = (600, 300)
    curr_pos = (600, 400)

    status_left = gate_left.check_crossing(prev_pos, curr_pos, track_id=1, frame_idx=1)
    status_right = gate_right.check_crossing(prev_pos, curr_pos, track_id=1, frame_idx=1)

    assert status_left is not None
    assert status_right is not None
    assert status_left != status_right, (
        "the same physical crossing should be entry for one side convention "
        "and exit for the other"
    )


def test_hysteresis_prevents_rapid_double_count():
    """A single real crossing, followed immediately (within
    min_frames_between_crossings) by a small back-and-forth wobble on the
    same track, should not be double-counted."""
    gate = make_gate()
    gate.min_frames_between_crossings = 10

    # Real crossing.
    status1 = gate.check_crossing((600, 300), (600, 400), track_id=1, frame_idx=1)
    assert status1 is not None

    # Wobble back across the line almost immediately -- frame_idx too close
    # to the last crossing, should be suppressed even though the geometry
    # would otherwise qualify.
    status2 = gate.check_crossing((600, 400), (600, 300), track_id=1, frame_idx=3)
    assert status2 is None, "crossing within min_frames_between_crossings should be suppressed"

    assert gate.entry_count + gate.exit_count == 1


def test_hysteresis_thresholds_exist_for_all_configured_gates():
    """Sanity check that GATE_HYSTERESIS values are in a plausible pixel
    range (not accidentally reverted to the old 0-1 normalized scale)."""
    for name, cfg in GATE_HYSTERESIS.items():
        side_mag = cfg["min_side_mag"]
        delta_side = cfg["min_delta_side"]

        assert side_mag >= 1.0, (
            f"{name} min_side_mag={side_mag} looks like a normalized "
            "fraction, not a pixel value -- possible regression of the "
            "hysteresis scale bug."
        )
        assert delta_side >= 1.0, (
            f"{name} min_delta_side={delta_side} looks like a normalized "
            "fraction, not a pixel value -- possible regression of the "
            "hysteresis scale bug."
        )
