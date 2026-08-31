# NNDS Scientific Audit Matrix
## Claim 1: PET from conflict region
- Code: `uvh_coco_fused_grid_pet.py` + `pet_conflict_checker.py`
- Config: `configs/sites/giti/grid_config.json`
- Test: `tests/test_pet_logic.py`
- Limitation: Conflict zone is synthetic geometry
## Claim 2: Same-origin exclusion
- Code: `if orig_a == orig_b: continue`
- Test: `tests/test_modules_smoke.py`
- Limitation: Split-ID mapping must be correct
## Claim 3: Conflict type from geometry
- Code: `conflict_classifier.py`
- Test: `tests/test_conflict_classifier.py`
- Limitation: Heuristic; `other` = uncertain
## Claim 4: Velocity from trajectories
- Code: `pet_conflict_checker.py`
- Test: `tests/test_pet_velocity.py`, `tests/test_savgol_velocity.py`
- Limitation: No ground-truth speed
## Claim 5: Calibration sensitivity
- Code: `monte_carlo_calibration_benchmark.py`
- Results: `configs/sites/giti/speed_perturbation_sensitivity.json`
- Limitation: Conditional simulation, not field accuracy
## Claim 6: Reproducible
- Code: `scripts/reproduce_pipeline.sh`
- Manifest: `outputs/reproducibility_manifest.json`
- Limitation: Requires model download / network
## Limitations
1. Synthetic calibration (20×16m), not field-validated.
2. 300 frames (≈10s) per site – proof-of-concept.
3. Tracking errors may persist; low-PET needs manual review.
4. Conflict type heuristic.
5. PET cutoff at 3.0s.

## MRC Conflict Type Limitation

At MRC, 25/34 screened events (73.5%) are classified as "other" by the geometric conflict classifier.
This high proportion indicates that the heuristic classifier (based on velocity-vector angle)
is not well-suited for MRC's traffic layout. This is a **stated limitation**: conflict-type
labels are descriptive and not validated against ground truth. The PET values themselves are
not affected by conflict-type classification.

**For the paper:** Report conflict types as "descriptive" and do not draw safety conclusions
based on MRC conflict-type breakdown.

## Temporal-Duplicate Rule Justification (10 frames)

**Rule:** Same vehicle pair in same grid cell must be separated by ≥ 10 frames (0.33s) to be considered distinct episodes.

**Justification:**
- At 30 FPS, 10 frames = 0.33s.
- Grid cell size = 100 px = 1 m (at 10 px/m BEV resolution).
- At typical urban speeds (5-10 m/s), a vehicle takes 0.1-0.2s to traverse 1 m.
- However, considering acceleration, deceleration, and detection lag, 10 frames (0.33s) is a conservative lower bound for a vehicle to fully enter, traverse, and exit a conflict zone.
- Two events separated by < 10 frames likely represent the same physical interaction counted twice due to tracking noise or ID switches.
- Events separated by ≥ 10 frames are distinct temporal episodes where vehicles interacted, left the zone, and interacted again.

**Empirical validation from current GITI screened dataset:**
- Minimum temporal separation between duplicate (pair, grid_cell) events = 33 frames (1.10s) — well above the 10-frame threshold.
- All 7 duplicate groups have separations ≥ 33 frames (≥ 1.10s), confirming they are distinct interactions.

## MRC Gate Function Correction

The `_get_entry_gate` function in `uvh_coco_fused_grid_pet.py` originally only handled
`entry_side='left'` and `entry_side='right'`. All MRC gates have `entry_side='both'`,
so the function always returned `'unknown'`. This was fixed by adding a condition to
handle `entry_side='both'` (treating it as a crossing in either direction).

**Result:** 11/34 MRC events now have at least one known gate entry (up from 0/34).
