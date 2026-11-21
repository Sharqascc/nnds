# NNDS Initial Implementation Audit

Audited commit: `ec6fa73fdecacefc285b86b58b759f4d855de4e4`

## Scope
Partial source review of traffic_analyzer.py, uvh_coco_fused_grid_pet.py,
conflict_classifier.py, and the GITI gate configuration.
These findings establish inspected code behavior, not runtime correctness,
ground-truth agreement, or publication readiness.

## Trajectory and occupancy
- Stored trajectory points use bounding-box bottom-center: (det.cx, det.y2).
- This is a ground-contact proxy, not a measured physical contact point.
- Occupancy returns the first and last observed inside-square frames.
- Separate visits are merged into one occupancy window.
- Boundary inclusion, conflict-point construction, and missing observations
  still require explicit specification and tests.

## PET
- Sequential PET = (second entry frame - first exit frame) / FPS.
- The legacy helper returns zero for touching windows and None for overlap.
- The runner retains only 0 < PET <= pet_threshold; default: 2.0 seconds.
- The source notes disagreement with ANNOTATION_GUIDE.md over zero-gap PET.
- pet_time_based uses the same frames and FPS; it is not independent evidence.
- Structured PET status and overlap fields are serialized, but their
  consistency with legacy PET still needs verification.
- Floating-point equivalence tests should use a justified tolerance.

## Gate attribution
- Five configured gates are two-endpoint lines in image pixels:
  West_Gate, SouthWest_Gate, South_Gate, East_Gate, North_Gate.
- Origin is top-left; x increases rightward and y downward.
- Attribution uses strict sign changes across infinite supporting lines.
- Finite segment bounds are not enforced by the inspected helper.
- Exact line touches do not count.
- Observations are processed in frame order.
- Multiple qualifying gates within one interval use configuration order.
- No qualifying crossing or insufficient observations returns unknown.
- Polygon terminology in the validation plan does not match this geometry.
- Gate placement and entry directions still require visual validation.

## Track splitting and identity
- The runner replaces tracks with the splitter output and forwards parameters.
- Runner defaults: gap 5 frames, jump 30 pixels, prediction tolerance 80 pixels.
- Helper defaults differ: gap 10 frames and jump 50 pixels.
- Splitting requires (gap > limit OR jump > limit) AND prediction error > limit.
- Good prediction can prevent a split across a long gap.
- Encoding: composite_id = original_id * 1000 + segment_index.
- Decoding uses integer division and remainder by 1000.
- This requires segment indices below 1000 to avoid identity collisions.
- Same-decoded-origin event pairs are explicitly excluded.
- Observation preservation, ID flow, repeated splits, threshold boundaries,
  and sibling temporal consistency remain untested in this audit.

## Conflict classification
- The runner uses analysis/conflict_classifier.py.
- Trajectories are supplied in PET first-actor/second-actor order.
- Empty input, JSON parsing failure, or zero estimated velocity gives other.
- Angle < 30 degrees: rear_end if speed difference exceeds 50% of the
  larger speed; otherwise side_swipe.
- Angle > 150 degrees: head_on.
- Angle from 60 through 120 degrees: crossing.
- Final fallback lines were not inspected.
- Shown rules do not check lateral separation, leader position, or closing.
- Labels are descriptive classifier outputs, not validated conflict types.
- The validation plan omits the possible head_on label.

## Velocity and coordinate handling
- Velocity uses up to five observations strictly before the conflict frame.
- It uses endpoint displacement divided by elapsed frames.
- Five observations do not necessarily span five frames.
- Insufficient history returns zero, like stationary motion.
- Units are world-coordinate units/frame or pixels/frame.
- Each actor independently prefers world coordinates at both endpoints.
- Partial mapping failures could cause incompatible actor coordinate systems.
- Serializer field names match those expected by the velocity helper.
- Non-finite coordinates and malformed inputs need dedicated tests.

## Temporal prefilter
- Candidate lifetime separation allowance is int(max_pet * fps) + 5 frames.
- This is not conflict-zone PET.
- Sorted-frame assumptions and boundary behavior need tests.

## Provenance and documentation gaps
- Historical scientific_audit.md mixes methods and results requiring provenance.
- Its overlap-counting claim is not supported by the inspected legacy helper,
  which returns None for overlapping windows.
- The 88-event count and 539-to-614 track accounting remain unverified here.
- Do not make those counts golden targets without supporting run artifacts.
- Previous untracked audit files and generated PET outputs are absent from
  this fresh checkout.

## Next steps
1. Write validation_spec.md: distinguish intended rules from current behavior.
2. Resolve zero-gap PET, occupancy visits, and gate geometry/priority policies.
3. Define supported conflict labels and limitations.
4. Test PET boundaries and structured-output consistency.
5. Test splitting accounting, IDs, and threshold boundaries.
6. Test gate direction, touches, extensions, and multiple crossings.
7. Test classification history and coordinate consistency.
8. Finish tracing conflict points, filtering, serialization, and CLI dispatch.
9. Recover or reproduce run artifacts with complete metadata.
10. Perform visual review and manual ground-truth validation.

No pipeline behavior was changed. This is an initial source audit only.
