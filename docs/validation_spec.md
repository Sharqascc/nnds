# NNDS GITI Validation Specification — Draft

## Status and scope
Draft based on docs/code_audit.md, merged in commit dcd1d7a.
This document separates current behavior, proposed requirements, and open
decisions. It does not establish empirical validity or authorize code changes.
Tests of current behavior must not be presented as scientific validation.

## 1. Coordinates and trajectory points
Current behavior:
- Image coordinates have top-left origin, x rightward, y downward.
- Trajectories use bounding-box bottom-center: ((x1 + x2)/2, y2).
- This is a ground-contact proxy, not a measured physical contact point.

Proposed requirements:
- Record video resolution and coordinate system in run metadata.
- Gate geometry and trajectory points must use compatible coordinates.
- Never compare actor velocities expressed in different coordinate systems.
- Record calibration provenance before interpreting world coordinates
  as physically accurate distances or speeds.

## 2. Occupancy
Current behavior:
- Occupancy uses observed trajectory-point inclusion in a square.
- The helper returns minimum and maximum inside-frame indices.
- Separate visits are collapsed into one window.
- No observed inside point returns no occupancy window.

Open decisions:
- Exact conflict-square boundary inclusion.
- Whether distinct visits must become separate occupancy episodes.
- Treatment of missing observations and boundary-crossing interpolation.
- Conflict-point construction, square size, and coordinate units.
- Treatment of truncated tracks and unobserved entry or exit.

Proposed requirements:
- Specify each decision before scientific acceptance tests are finalized.
- Do not interpret first/last observed inside frames as exact physical
  boundary-crossing times without qualification.

## 3. PET definition and current screening policy
For non-overlapping occupancy windows:

PET = t_second_entry - t_first_exit

With frame indices f and positive finite FPS:

PET = (f_second_entry - f_first_exit) / FPS

Current behavior:
- Either actor may occupy the zone first.
- Touching windows produce zero in the legacy helper.
- Overlapping windows return None in the legacy helper.
- The runner retains only 0 < PET <= threshold.
- Default threshold: 2.0 seconds.

Worked sequential example:
- At 30 FPS, actor A exits at frame 324 (10.8 seconds).
- Actor B enters at frame 369 (12.3 seconds).
- PET = (369 - 324)/30 = 1.5 seconds.

Open decisions:
- Whether zero-gap events belong in a separate category or retained output.
- How overlap events are represented and reported.
- Invalid, reversed, missing, or non-finite occupancy-window handling.
- Reconciliation with ANNOTATION_GUIDE.md.

Proposed tests:
- Both actor orders, zero gap, overlap, and identical windows.
- Threshold boundaries, including exactly 2.0 seconds.
- Constant frame-index shifts and proportional frame/FPS scaling.
- Invalid FPS and malformed occupancy windows.
- Float-tolerant agreement between equivalent PET representations.

pet_time_based uses the same frame windows and FPS as PET; agreement is
an internal consistency check, not independent measurement validation.

## 4. Gate attribution
Current configuration:
- West_Gate, SouthWest_Gate, South_Gate, East_Gate, North_Gate.
- Exact coordinates and entry sides remain in
  configs/sites/giti/gate_config.yaml.
- Gates are two-endpoint lines, not polygons.

Current behavior:
- Strict side-sign change across an infinite supporting line.
- Entry direction is checked.
- Exact line touches do not qualify.
- First qualifying observation interval wins.
- Configuration order resolves multiple qualifying gates in one interval.
- No crossing or insufficient history gives unknown.

Open decisions:
- Finite segment versus infinite supporting-line crossing.
- Endpoint touches and trajectories sampled exactly on the line.
- Multiple-crossing priority and interpolated crossing time.
- Which trajectory stage supplies attribution.
- Visual confirmation of placement, resolution, and entry direction.

Proposed requirements:
- Preserve unknown rather than silently substituting a nearest-gate label.
- Any approximate attribution must use separate fields and provenance.
- Test each configured gate, wrong directions, touches, line extensions,
  multiple crossings, and no-crossing trajectories.
- Transformation tests must transform geometry, directions, and any
  distance thresholds consistently.

## 5. Track splitting and identity
Current runner defaults:
- Frame-gap limit: 5 frames.
- Spatial-jump limit: 30 pixels.
- Prediction-error tolerance: 80 pixels.

Current split condition:
(gap > gap_limit OR jump > jump_limit)
AND prediction_error > prediction_tolerance

The helper defaults differ from runner defaults. Tests must pass explicit
parameters when testing the intended run configuration.

Current identity encoding:
composite_id = original_id * 1000 + segment_index

Decoding assumes segment_index is between 0 and 999.
Same-decoded-origin event pairs are excluded.

Proposed requirements:
- Every input observation is preserved exactly once unless a documented
  discard policy explicitly applies.
- Composite IDs are unique and original identity remains recoverable.
- Define duplicate-frame handling before asserting sibling non-overlap.
- Test strict threshold boundaries, prediction exceptions, repeated splits,
  empty tracks, and segment-ID overflow.
- Do not use historical 539-to-614 accounting as a golden target without
  verified source artifacts and run metadata.

## 6. Conflict classification
Current observed labels include:
rear_end, side_swipe, head_on, crossing, other.

Inspected rules:
- Zero estimated velocity or insufficient history gives other.
- Angle <30 degrees: rear_end for speed difference >50% of larger speed;
  otherwise side_swipe.
- Angle >150 degrees: head_on.
- Angle 60–120 degrees inclusive: crossing.
- The final fallback still requires source verification.

Velocity estimation:
- Up to five observations strictly before the conflict frame.
- Endpoint displacement divided by frame difference.
- World coordinates when available at both selected endpoints; otherwise
  pixels, independently for each actor.

Proposed requirements:
- Describe labels as heuristic classifier outputs until empirically validated.
- Do not treat similar headings and speeds as proof of side-swipe.
- Do not treat a speed difference alone as proof of a closing rear-end conflict.
- Test stationary and insufficient-history cases separately.
- Test coordinate consistency, non-finite inputs, angle boundaries,
  speed-difference boundaries, and actor-order effects.
- Define scientific conflict categories and precedence separately from
  implementation-regression expectations.

## 7. Output consistency and duplicate events
Proposed requirements:
- Define and version required CSV schemas.
- Retained PET is finite, positive, and within the configured threshold.
- Referenced trajectory IDs resolve in the documented output identity space.
- Same-original-track pairs are excluded.
- Frame and timestamp fields agree within justified tolerances.
- Structured PET fields agree with the documented status policy.

Open decisions:
- Exact event identity and duplicate-event key.
- Whether repeated encounters of the same pair are separate episodes.
- Required cross-file frame coverage and effects of filtering.
- Full structured-PET and output-schema semantics.

Do not assert arbitrary duplicate suppression or cross-file invariants
before these relationships are traced and specified.

## 8. Reproducibility and empirical validation
For each canonical run, preserve:
- Code commit and working-tree state.
- Exact command, effective parameters, configuration, and seeds.
- Video and model checksums.
- Environment and dependency versions.
- FPS, resolution, decoded frame count, logs, and exit status.
- PET, detection, segment, and quality-report artifacts.

Historical counts of 88 events and 539-to-614 tracks remain unverified
in the current runtime. Recover or reproduce evidence before adoption.

Software tests alone do not establish event validity. Visual review and
manually annotated events and non-events are required for empirical claims.
Define annotation rules, sampling, matching, and denominators before
reporting precision, recall, gate accuracy, or PET error.

## Acceptance status
No scientific acceptance is granted by this draft.
Next: resolve open decisions, finish the remaining source trace, and add
small deterministic tests tied to explicit requirements.
