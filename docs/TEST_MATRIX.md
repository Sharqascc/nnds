# NNDS Module Test Matrix

This document tracks testing requirements for every module in the repository.

## Core Scientific Modules (Must Have Tests)

| Module | Scientific Role | Existing Tests | Needed Tests |
|--------|----------------|----------------|--------------|
| `src/bev/bev_mapper.py` | Pixel → World coordinate mapping | `test_bev_validation.py` | Homography accuracy, boundary cases |
| `src/bev/giti_bev_calib.py` | GITI calibration | None | Reprojection error, condition number |
| `src/bev/calibration/grid_validation_calibration.py` | Grid validation | None | Cross-validation, error metrics |
| `src/bev/calibration/monte_carlo_calibration_benchmark.py` | Monte Carlo sensitivity | None | Sensitivity analysis, statistical properties |
| `src/analysis/pet_conflict_checker.py` | PET computation | `test_pet_conflict.py`, `test_pet_velocity.py` | PET formula, uncertainty, velocity |
| `src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py` | Detection + tracking + PET | `test_modules_smoke.py` | Event generation, deduplication, gap handling |
| `src/analysis/grid_trajectory/spatial_grid.py` | Grid cell assignment | `test_configs_smoke.py` | Cell boundaries, naming |
| `src/analysis/conflict_classifier.py` | Conflict type | `test_conflict_classifier.py` | All conflict types, edge cases |
| `src/pipeline/custom_tracker.py` | Multi-object tracking | None | ID stability, gap handling |
| `src/pipeline/reid_encoder.py` | Re-identification | None | Feature extraction, matching |
| `src/pipeline/traffic_analyzer.py` | Pipeline orchestration | `test_traffic_analyzer_cli.py` | End-to-end flow, config handling |

## Utility Modules (Should Have Basic Tests)

| Module | Purpose | Tests |
|--------|---------|-------|
| `src/core/types.py` | Data types | Type validation |
| `src/core/validation.py` | Validation functions | Error metrics, validation logic |
| `src/utils/debug_helpers.py` | Debug utilities | Basic functionality |
| `src/utils/seed.py` | Random seed | Determinism |
| `src/utils/interactive.py` | Interactive tools | Basic functionality |

## Analysis Modules (Should Have Logic Tests)

| Module | Purpose | Tests |
|--------|---------|-------|
| `src/analysis/gate_counter.py` | Gate counting | Gate crossing detection |
| `src/analysis/pet_summary.py` | PET summaries | Summary statistics |
| `src/analysis/ssm/ssm_verification.py` | SSM verification | Verification logic |
| `src/analysis/verification/statistical_testing.py` | Statistical tests | Test correctness |
| `src/analysis/visualization/*.py` | Plots | Plot generation (smoke) |

## Experimental Modules (Should Be Excluded from Tests)

| Module | Status |
|--------|--------|
| `src/diffusion/` | 🔴 Experimental - not used in paper |
| `src/vlm/` | 🔴 Experimental - not used in paper |
| `scripts/train_*.py` | 🔴 Experimental - training scripts |
