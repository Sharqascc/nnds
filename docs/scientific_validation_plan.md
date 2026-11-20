# NNDS Scientific Validation and Quality Assurance Plan

## Objective

Establish that the NNDS traffic-safety pipeline produces scientifically trustworthy GITI conflict and surrogate-safety results.

The plan separates:

1. **Software correctness**: the code implements its documented rules.
2. **Scientific correctness**: the documented rules represent traffic-safety phenomena correctly.
3. **Empirical validity**: pipeline outputs agree with what actually occurs in the GITI video.

## Guiding principles

- Do not treat `exit code 0` or a successful full-video run as proof of scientific validity.
- Write the scientific specification before implementing tests.
- Prioritize tests that protect PET, gate attribution, track splitting, and conflict classification.
- Use regression tests for known past failures before adding speculative tests.
- Use small deterministic fixtures for routine tests; reserve full-video runs for release validation.
- Perform visual auditing early, because incorrect detector, tracker, gate, or conflict-zone assumptions invalidate downstream tests.
- Create manually labeled ground truth before publication.
- Report sensitivity to thresholds and configuration choices.
- Preserve complete reproducibility metadata for every canonical run.

## Implementation phases

### Phase 1: Scientific specifications

1. Audit `src/pipeline/traffic_analyzer.py` and identify every module involved in detection, tracking, track splitting, grid assignment, gate attribution, PET computation, conflict classification, and CSV output.
2. Create a module inventory with inputs, outputs, dependencies, and scientific role.
3. Identify all hardcoded constants, thresholds, coordinate-system assumptions, and geometry.
4. Inventory existing tests and classify them by type.
5. Identify gaps between existing tests and required scientific rules.
6. Produce `docs/code_audit.md`.

7. Define PET exactly in `docs/validation_spec.md`.
8. Define PET as `PET = t_second_entry - t_first_exit`, in seconds.
9. Define frame-to-second conversion.
10. Define the overlap policy: zero PET, negative PET, undefined PET, or separate overlap category.
11. Define missing entry/exit behavior.
12. Define behavior when exit precedes entry.
13. Define whether the 2.0-second threshold is inclusive or exclusive.
14. Define duplicate-event policy.
15. Define multiple-occupancy behavior.
16. Add worked examples.

17. Document the gate coordinate system and video resolution.
18. Document the five gates: `West_Gate`, `SouthWest_Gate`, `South_Gate`, `East_Gate`, and `North_Gate`.
19. Store exact gate polygon geometry.
20. Define trajectory-gate crossing and entry-time rules.
21. Define gate priority when multiple gates are crossed.
22. Define the `unknown` condition.
23. Define whether attribution uses detections, split tracks, or valid tracks.

24. Document track-splitting gap and jump thresholds.
25. Define whether gaps use frames or seconds.
26. Define whether jump distance uses pixels or normalized coordinates.
27. Define whether tracks can be split multiple times.
28. Define whether observations may be discarded.
29. Define original-to-split track ID relationships.
30. Define the accounting invariant between original and split tracks.

31. Define `side_swipe`, `crossing`, `rear_end`, and `other` explicitly.
32. Define classification order and precedence.
33. Define whether classification depends on vehicle class, conflict-zone occupancy, and PET.
34. Document all classification thresholds.
35. Add synthetic examples for every conflict type.

36. Define conflict-zone geometry, coordinate system, and occupancy rule.
37. Define whether occupancy uses bounding-box intersection, centroid inclusion, or another rule.
38. Define minimum dwell time and partial-overlap behavior.

39. Define canonical run command, input video, model weights, configuration, outputs, checksums, environment, logs, seeds, and quality report.

### Phase 2: Regression tests

40. Create `tests/regression/`.
41. Add a regression test that fails if valid gate-attributable trajectories return only `unknown`.
42. Add tests that each of the five gates can be assigned from a synthetic trajectory.
43. Add tests that validate gate configuration loading, names, polygon validity, and non-ambiguity.
44. Add a test that a trajectory outside all gates returns `unknown`.

45. Add track-splitting regression tests.
46. Verify that splitting does not lose or duplicate observations.
47. Verify that split segments do not overlap in time.
48. Verify unique split-track IDs.
49. Verify the 539-to-614 canonical split accounting.
50. Verify that total split frames equal total original frames minus explicitly discarded frames.

51. Add PET-output regression tests.
52. Verify the canonical PET CSV exists and contains 88 events.
53. Verify required columns, finite PET values, threshold compliance, conflict types, and gate vocabulary.
54. Verify every referenced track ID exists in the segments CSV.

55. Add cross-file consistency tests.
56. Verify PET track IDs exist in segments.
57. Verify PET event frames exist in detections.
58. Verify entry frames do not exceed exit frames.
59. Verify timestamps agree with frame indices and FPS.
60. Verify no duplicate PET event for the same ordered track pair and event frame.

### Phase 3: Scientific correctness tests

61. Create `tests/scientific_correctness/`.
62. Create PET physics tests.
63. Test positive sequential PET.
64. Test A occupying 10.0–10.8 seconds and B entering at 12.3 seconds; expected PET is 1.5 seconds.
65. Test zero gap, overlap, identical timestamps, and missing timestamps.
66. Test frame-to-second conversion at 30 FPS and another FPS.
67. Test PET threshold boundaries at 1.999, 2.000, and 2.001 seconds.
68. Test duplicate track pairs.
69. Test that PET is unchanged when all timestamps shift by a constant.
70. Test that PET is unchanged when FPS changes and frame indices scale proportionally.

71. Create gate-geometry tests.
72. Verify gate and detection coordinate systems match.
73. Verify polygons are valid, bounded, and non-ambiguous.
74. Test each of the five gates with a synthetic entering trajectory.
75. Test no-gate trajectories return `unknown`.
76. Test multi-gate trajectories follow the documented priority rule.
77. Test translation, mirroring, and scaling metamorphic invariants.

78. Create trajectory and tracking tests.
79. Verify monotonic timestamps, finite coordinates, valid bounds, and non-negative duration.
80. Test gap and jump thresholds.
81. Test disappearance/reappearance and duplicate IDs.

82. Create conflict-geometry tests.
83. Test parallel vehicles, non-intersecting approaches, separating vehicles, stationary vehicles, reversing vehicles, and simultaneous occupancy.
84. Test synthetic side-swipe, crossing, rear-end, and other cases.
85. Test conflict-type precedence.

86. Add TTC, MTTC, and DRAC tests only for indicators actually implemented and used in the paper.
87. Define each indicator mathematically before testing it.

### Phase 4: Adversarial tests

88. Create `tests/adversarial/`.
89. Test duplicate IDs, ID switches, disappearance/reappearance, large jumps, long gaps, single-observation tracks, and zero-duration tracks.
90. Test simultaneous entry/exit, persistent occupancy, missing exits, one-frame gaps, threshold boundaries, and overlapping occupancy.
91. Test boundary-touching trajectories, vertex crossings, simultaneous gate crossings, conflict-zone boundary cases, out-of-frame trajectories, negative coordinates, and non-finite coordinates.
92. Test false collision predictions, parallel motion, separating vehicles, reversing vehicles, stationary vehicles, and identical trajectories.

### Phase 5: Property tests

93. Create `tests/property/`.
94. Add PET properties: constant time-shift invariance, FPS-scaling invariance, finite valid output, documented overlap policy, and threshold compliance.
95. Add gate properties: translation, scaling, and mirroring invariance; valid label set; `unknown` outside gates.
96. Add track properties: frame preservation, no overlapping sibling segments, unique IDs, non-negative duration, finite coordinates, and deterministic splitting.

### Phase 6: Integration and regression fixtures

97. Create `tests/integration/`.
98. Create a small deterministic `tests/fixtures/giti_mini.mp4` fixture.
99. Record its checksum, FPS, resolution, duration, and frame count.
100. Add CLI tests for `--video`, `--max-frames`, `--out-csv`, `--video-source`, and `--time-of-day`.
101. Verify `--video` defaults to `None`.
102. Verify valid mini-fixture execution exits with code 0.
103. Verify missing video and invalid output path fail appropriately.
104. Add PET, detections, and segments output-schema tests.
105. Add a golden regression test comparing stable fields: event count, gate labels, conflict types, PET within tolerance, schemas, and row counts.
106. Do not compare raw detection coordinates exactly.

### Phase 7: Visual and empirical validation

107. Create `results/visual_audit/`.
108. Select 10–20 canonical events covering all gates, conflict types, extreme PET values, unknown gates, and threshold-adjacent events.
109. Render overlays with track IDs, boxes, trajectories, gate polygons, conflict-zone geometry, and PET labels.
110. Review each event and record whether it is plausible, a false positive, a missed event, or affected by an ID switch or geometry error.
111. Save `audit_table.csv` and `audit_notes.md`.

112. Create `results/ground_truth/`.
113. Define an annotation protocol.
114. Annotate 30–50 events and non-events.
115. Record track IDs, manual PET, conflict type, gate attribution, and event validity.
116. Have a second annotator review a subset.
117. Calculate inter-annotator agreement.
118. Calculate precision, recall, false-positive rate, false-negative rate, gate accuracy, conflict-type accuracy, PET absolute error, PET RMSE, and PET bias.

### Phase 8: Sensitivity analysis

119. Create `results/sensitivity/`.
120. Test PET thresholds including 1.5, 2.0, and 2.5 seconds.
121. Test track-gap thresholds including 4, 5, and 6 frames.
122. Test maximum jump-distance thresholds.
123. Test alternative conflict-zone and gate-polygon definitions where scientifically justified.
124. Compare event counts, PET distributions, conflict types, and gate pairs.
125. Save `sensitivity_report.md` and `sensitivity_results.csv`.

### Phase 9: Coverage, mutation testing, and CI

126. Run `pytest --cov=src --cov-report=term-missing --cov-report=html`.
127. Prioritize uncovered branches in PET, gates, splitting, conflicts, and CSV serialization.
128. Add targeted mutation testing for PET, gate attribution, conflict classification, and track splitting.
129. Run Ruff, Ruff format check, and MyPy.
130. Create `.github/workflows/quality.yml`.
131. Add fast CI for unit, property, scientific, adversarial, and consistency tests.
132. Add slow CI for integration and regression tests.
133. Add scheduled or release CI for full-video validation.
134. Upload coverage and test artifacts.

### Phase 10: Reproducibility and reporting

135. Create `results/canonical_run/`.
136. Save configuration, environment, input manifest, run metadata, detections, tracks, segments, PET events, quality report, logs, Git commit, model checksums, video checksum, seeds, FPS, resolution, and frame count.
137. Create a final quality report separating software correctness, scientific correctness, and empirical validity.
138. Prepare a paper-facing validation section reporting PET distribution, conflict types, gates, visual audit, ground-truth metrics, sensitivity findings, and limitations.

## Priority order

1. Audit implementation.
2. Write specifications.
3. Add gate and track-splitting regression tests.
4. Add PET scientific tests.
5. Add gate and conflict-geometry tests.
6. Add consistency and adversarial tests.
7. Add property tests.
8. Add integration and golden regression tests.
9. Perform visual audit.
10. Build ground truth and calculate validation metrics.
11. Run sensitivity analyses.
12. Add coverage, mutation testing, static analysis, and CI.
13. Create reproducibility metadata and final quality report.
