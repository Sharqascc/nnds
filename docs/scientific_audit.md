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
