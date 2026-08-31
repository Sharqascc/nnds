# NNDS Module Scientific Manifest
## Canonical Pipeline
```
Video -> Detector (UVH-COCO fused) -> Tracking (CustomTracker + ReIDEncoder) -> BEV (bev_mapper) -> Spatial Grid -> PET (pet_conflict_checker) -> Conflict Classifier
```
## Core Modules (used for paper results)
| Module | Role |
|--------|------|
| `src/pipeline/traffic_analyzer.py` | Entry point, CLI |
| `src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py` | Detector + tracking + PET generation |
| `src/pipeline/custom_tracker.py` | Multi-object tracking |
| `src/pipeline/reid_encoder.py` | Appearance re-identification |
| `src/bev/bev_mapper.py` | Pixel -> local planar |
| `src/analysis/grid_trajectory/spatial_grid.py` | Grid cells |
| `src/analysis/pet_conflict_checker.py` | PET + velocity |
| `src/analysis/conflict_classifier.py` | Conflict type |
## Validation Modules
| Module | Role |
|--------|------|
| `src/bev/calibration/monte_carlo_calibration_benchmark.py` | Sensitivity |
| `scripts/sensitivity_pet_fragmentation.py` | Fragmentation sensitivity |
| `scripts/validate_all.py` | Full validation |
## Experimental (NOT used in paper)
| Module | Status |
|--------|--------|
| `src/diffusion/` | 🔴 Experimental |
| `src/vlm/` | 🔴 Experimental |
| `scripts/train_*.py` | 🔴 Experimental |
## Config Source of Truth
- GITI: `configs/sites/giti/`
- MRC: `configs/sites/mrc/`
## Camera Intrinsics
Not used (video uses different sensor mode).
