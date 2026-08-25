# Full Repository Details Report for External AI Assessment

**Generated on:** 2026-08-25T12:30:00.712228  
**Repository:** Sharqascc/nnds  
**Branch:** cleanup/system-reorganization  
**Commit:** 3ab3ed7  
**Working tree clean:** True

## 1. Repository Overview
- Total tracked files: 193
- Total size (tracked, not .git): 140.26 MB
- Python source files: 69 (20253 lines)
- Scripts: 27 files (3265 lines)
- Tests: 22 files (1066 lines)
- Baselines: 4 files (164 lines)

## 2. Recent Commits
```
3ab3ed7 Add optional polish: paired t-test, GT sample, anonymized video, CI artifact, changelog
de8ff5a Fix constant acceleration test expected values
4c50062 Add final repository assessment report
fa6be27 Add anonymization, Docker, logging, expanded statistics, and tests
c723ef0 Fix tests and run_pipeline import order
```

## 3. Test Suite
- Result: 55 passed in 10.12s
- All tests passed: True

## 4. BEV Validation Report
```
============================================================
BEV Homography Validation Report
============================================================
Rank: 3 (should be 3)
Condition number (raw): 3.58e+13
Condition number (normalized): 1.79e+17
Normalized condition number (after Hartley pre-conditioning) should be < 1e6 for good numerical stability.
Reprojection errors (world units): [0.03064413 0.03064413 0.03064413 0.03064413 0.03064413 0.03064413]
  Mean: 0.031
  Max:  0.031
============================================================
✅ Overlay image saved to outputs/bev_validation_overlay.png
```

## 5. Top-Level Structure
```
  dir  .github
  file CHANGELOG.md (1156 bytes)
  file CITATION.cff (296 bytes)
  file CONTRIBUTING.md (647 bytes)
  file DATA_LICENSE (718 bytes)
  file Dockerfile (324 bytes)
  file LICENSE (1066 bytes)
  file Makefile (1819 bytes)
  file PRIVACY.md (706 bytes)
  file README.md (2995 bytes)
  dir  baselines
  dir  configs
  dir  data
  file docker-compose.yml (214 bytes)
  dir  docs
  file environment.yml (125 bytes)
  dir  examples
  dir  model_cards
  dir  outputs
  file pyproject.toml (902 bytes)
  file requirements.txt (165 bytes)
  dir  scripts
  dir  src
  dir  tests
```

## 6. Requirements (Minimal Direct Dependencies)
```
torch>=2.0.0
torchvision
ultralytics>=8.0.0
opencv-python
numpy
pandas
scipy
matplotlib
tqdm
pyyaml
openvino>=2024.0.0
lap>=0.5.12
pytest>=8.0.0
scikit-learn
PyYAML

```

## 7. CI Workflow
```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Fetch small LFS test fixtures
        run: |
          git lfs install --local
          git lfs pull --include "configs/**" --include "docs/data_samples/petevents_bev_demo.csv"

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pytest
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

      - name: Run smoke test
        run: |
          pytest -q tests/
      - name: Run unified validation (BEV + outputs)
        run: |
          python scripts/validate_all.py --skip-pytest \
            --detections tests/fixtures/sample_detections.csv \
            --detections-split tests/fixtures/sample_split_detections.csv \
            --pet tests/fixtures/sample_pet.csv \
            --video-frames 100
      - name: Validate pipeline outputs
        run: |
          python scripts/validate_outputs.py             --detections tests/fixtures/sample_detections.csv             --detections-split tests/fixtures/sample_split_detections.csv             --pet tests/fixtures/sample_pet.csv             --video-frames 100
      - name: Generate experiment log
        run: |
          python scripts/experiment_logger.py \
            --detections tests/fixtures/sample_detections.csv \
            --pet tests/fixtures/sample_pet.csv \
            --output outputs/experiment_log.json
      - name: Upload experiment log
        uses: actions/upload-artifact@v4
        with:
          name: experiment-log
          path: outputs/experiment_log.json


```

## 8. .gitignore
```
# Model weights
*.pt
*.pth
*.onnx
*.engine

# Video files
*.mp4
*.avi
*.mov

# Data outputs
outputs/
data/models/
sample_data/
!data/sample_data/
!data/sample_data/traffic_video.mp4

# Backup files
*.bak

# Python caches
__pycache__/
*.pyc

# Colab / Jupyter
.ipynb_checkpoints/

# Large local data / checkpoints
data/checkpoints/
data/uvh26_data/

# Generated CSV outputs
*.csv

!data/sample_data/anonymized_traffic_video_50f.mp4

```

## 9. License
```
MIT License

Copyright (c) 2026 Sharqascc

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```

## 10. Configuration Files
### GITI_grid_config.json
Keys: ['corners', 'configuration']
```json
{
  "corners": {
    "top_left": [
      0,
      0
    ],
    "top_right": [
      1600,
      0
    ],
    "bottom_left": [
      0,
      720
    ],
    "bottom_right": [
      1600,
      720
    ]
  },
  "configuration": {
    "cell_size": 50,
    "naming_style": "CELL_{col}_{row}"
  }
}
```
### bev_config.json
Keys: ['x_min', 'x_max', 'y_min', 'y_max', 'resolution', 'bev_resolution', 'H_pixel_to_world', 'notes']
```json
{
  "x_min": 730900.97,
  "x_max": 730920.97,
  "y_min": 221998.35,
  "y_max": 222014.35,
  "resolution": [
    1000,
    800
  ],
  "bev_resolution": [
    1000,
    800
  ],
  "H_pixel_to_world": [
    [
      0.01586042823157627,
      3.799465845860288e-14,
      730897.923076923
    ],
    [
      2.559903913728168e-15,
      0.02707275803723223,
      221994.98672800336
    ],
    [
      -1.8521905298165306e-20,
      -5.0326215803389403e-20,
      1.0
    ]
  ],
  "notes": {
    "updated
```
### giti_calibration_points.json
Keys: ['metadata', 'calibration_points']
```json
{
  "metadata": {
    "timestamp": "2024-02-22",
    "description": "G-ITI Intersection Calibration Points",
    "coordinate_system": "Northing, Easting, Elevation (feet)",
    "origin": "Point 1 (222006.49, 730911.81, 197)",
    "total_points": 6,
    "image_size": "1600x720",
    "camera_used": "Mobile Phone",
    "updated": "expanded calibration footprint",
    "target_width_m": 20.0,
    "target_height_m": 16.0,
    "review": "mid-edge points corrected for BEV rectangle consistency"
  },
  "
```

## 11. Essential Publication Files
- [x] README.md
- [x] LICENSE
- [x] CITATION.cff
- [x] CONTRIBUTING.md
- [x] CHANGELOG.md
- [x] DATA_LICENSE
- [x] PRIVACY.md
- [x] Dockerfile
- [x] docker-compose.yml

## 12. Model Cards
### uvh26.md
# UVH-26 Model Card

- **Model:** UVH-26-MV-YOLOv11-S
- **Source:** Hugging Face iisc-aim/UVH-26
- **Task:** Multi-class vehicle detection (pedestrian, car, bike, bus, truck, auto, etc.)
- **Training data:** Proprietary Indian traffic dataset (not included)
- **Input size:** 640x640 typical
- **Framework:** Ultralytics YOLO (PyTorch)
- **Intended use:** Research in traffic conflict analysis
- **Limitations:** Trained on Indian traffic scenes; performance may vary in other regions.

### yolo11n.md
# YOLO11n Model Card

- **Model:** YOLO11n
- **Source:** Ultralytics official release
- **Task:** COCO object detection (person fallback)
- **Training data:** COCO dataset
- **Input size:** 640x640
- **Framework:** Ultralytics YOLO (PyTorch)
- **Intended use:** Person detection to supplement UVH-26
- **Limitations:** General COCO classes; not traffic-specific.


## 13. Baselines Overview
- constant_acceleration.py (39 lines)
- constant_velocity.py (34 lines)
- kalman_filter.py (35 lines)
- social_force.py (56 lines)

## 14. Complete Python File Inventory (with sizes)
| File | Size (bytes) |
|------|-------------|
| baselines/constant_acceleration.py | 1172 |
| baselines/constant_velocity.py | 1032 |
| baselines/kalman_filter.py | 1062 |
| baselines/social_force.py | 1950 |
| examples/quickstart.py | 1469 |
| scripts/anonymize_video.py | 1524 |
| scripts/convert_del4_to_diffusion.py | 6577 |
| scripts/convert_pet_to_diffusion_csv.py | 3313 |
| scripts/debug_tracking_video.py | 2949 |
| scripts/diagnose_tracking.py | 2901 |
| scripts/ensure_models.py | 2284 |
| scripts/evaluate_ground_truth.py | 2960 |
| scripts/evaluate_position_ddpm.py | 4627 |
| scripts/evaluate_transformer_diffusion.py | 4943 |
| scripts/experiment_logger.py | 2017 |
| scripts/export_openvino.py | 877 |
| scripts/grid_search_smoothing.py | 5276 |
| scripts/inspect_pet.py | 2842 |
| scripts/paired_ttest.py | 1797 |
| scripts/run_pipeline.py | 1760 |
| scripts/split_detections.py | 2181 |
| scripts/tracking_report.py | 7218 |
| scripts/tracking_report_fast.py | 4391 |
| scripts/traffic_analyzer_demo.py | 5270 |
| scripts/train_position_ddpm.py | 3863 |
| scripts/train_transformer_diffusion.py | 3604 |
| scripts/validate_all.py | 5909 |
| scripts/validate_bev.py | 3954 |
| scripts/validate_outputs.py | 8968 |
| scripts/validation_report.py | 10788 |
| scripts/visualize_pet.py | 4785 |
| scripts/visualize_pet_live.py | 4073 |
| src/__init__.py | 0 |
| src/analysis/__init__.py | 0 |
| src/analysis/analysis/__init__.py | 10343 |
| src/analysis/analysis/logging/__init__.py | 652 |
| src/analysis/analysis/logging/reproducibility_audit.py | 25744 |
| src/analysis/analysis/pet_diffusion_analysis.py | 22810 |
| src/analysis/analysis/pet_summary.py | 21657 |
| src/analysis/analysis/research_run.py | 6862 |
| src/analysis/analysis/safety_eval_diffusion.py | 8488 |
| src/analysis/analysis/safety_eval_diffusion_notebook.py | 8586 |
| src/analysis/analysis/ssm/__init__.py | 1185 |
| src/analysis/analysis/ssm/ssm_verification.py | 21865 |
| src/analysis/analysis/ssm/uncertainty_quantifier.py | 19840 |
| src/analysis/analysis/verification/__init__.py | 1051 |
| src/analysis/analysis/verification/statistical_testing.py | 27222 |
| src/analysis/analysis/visualization/__init__.py | 16325 |
| src/analysis/analysis/visualization/industry_standard_viz.py | 35983 |
| src/analysis/analysis/visualization/pet_diffusion_plots.py | 19694 |
| src/analysis/analysis/visualization/pet_event_plots.py | 17942 |
| src/analysis/analysis/visualization/video_overlays.py | 22409 |
| src/analysis/audit/__init__.py | 0 |
| src/analysis/gate_counter.py | 20375 |
| src/analysis/grid_trajectory/__init__.py | 461 |
| src/analysis/grid_trajectory/pet_grid.py | 13284 |
| src/analysis/grid_trajectory/sam3_grid_pet.py | 13860 |
| src/analysis/grid_trajectory/spatial_grid.py | 9808 |
| src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py | 25501 |
| src/analysis/grid_trajectory/yolo_cpu_grid_pet.py | 7976 |
| src/analysis/pet_conflict_checker.py | 34672 |
| src/bev/__init__.py | 0 |
| src/bev/bev_mapper.py | 15483 |
| src/bev/calibration/__init__.py | 0 |
| src/bev/calibration/grid_validation_calibration.py | 21079 |
| src/bev/calibration/monte_carlo_calibration_benchmark.py | 19710 |
| src/bev/giti_bev_calib.py | 4962 |
| src/core/__init__.py | 131 |
| src/core/types.py | 2293 |
| src/core/validation.py | 1625 |
| src/diffusion/__init__.py | 0 |
| src/diffusion/complete_ddpm.py | 11551 |
| src/diffusion/traffic_diffusion/__init__.py | 0 |
| src/diffusion/traffic_diffusion/data/__init__.py | 0 |
| src/diffusion/traffic_diffusion/evaluate_fixed.py | 8833 |
| src/diffusion/traffic_diffusion/model_and_sampler.py | 2959 |
| src/diffusion/traffic_diffusion/sampling_utils.py | 1514 |
| src/diffusion/traffic_diffusion/split_dataset.py | 2197 |
| src/diffusion/traffic_diffusion/train_trajectory_diffusion.py | 4538 |
| src/diffusion/traffic_diffusion/training_utils.py | 22062 |
| src/diffusion/traffic_diffusion/trajectory_diffusion.py | 4751 |
| src/diffusion/traffic_diffusion/transformer_diffusion.py | 1952 |
| src/diffusion/traj_diffusion_normalized.py | 9411 |
| src/pipeline/__init__.py | 0 |
| src/pipeline/custom_tracker.py | 10684 |
| src/pipeline/reid_encoder.py | 1782 |
| src/pipeline/rt_detr_detector.py | 979 |
| src/pipeline/traffic_analyzer.py | 45390 |
| src/utils/__init__.py | 0 |
| src/utils/debug_helpers.py | 3739 |
| src/utils/interactive.py | 971 |
| src/utils/seed.py | 353 |
| src/vlm/__init__.py | 35 |
| src/vlm/analyzer.py | 4831 |
| src/vlm/config.py | 3983 |
| src/vlm/gate_validator.py | 9321 |
| src/vlm/test_free_models.py | 4213 |
| src/vlm/utils/__init__.py | 311 |
| src/vlm/utils/image_utils.py | 2558 |
| src/vlm/utils/visualization.py | 2511 |
| src/vlm/vlm_enhanced_pipeline.py | 6644 |
| tests/__init__.py | 0 |
| tests/conftest.py | 144 |
| tests/test_baselines_extra.py | 863 |
| tests/test_baselines_seed.py | 850 |
| tests/test_bev_validation.py | 264 |
| tests/test_configs_smoke.py | 8334 |
| tests/test_diffusion_smoke.py | 899 |
| tests/test_imports_smoke.py | 1164 |
| tests/test_modules_smoke.py | 3729 |
| tests/test_new_scripts.py | 1182 |
| tests/test_paired_ttest.py | 748 |
| tests/test_pet_conflict.py | 53 |
| tests/test_pet_logic.py | 1960 |
| tests/test_pet_output_schema.py | 1660 |
| tests/test_repo_smoke.py | 1769 |
| tests/test_research_run_smoke.py | 750 |
| tests/test_rtdetr_stub.py | 699 |
| tests/test_smoke.py | 143 |
| tests/test_speed_estimation.py | 57 |
| tests/test_traffic_analyzer_cli.py | 8899 |
| tests/test_traffic_analyzer_demo_smoke.py | 219 |
| tests/test_validation.py | 769 |

## 15. Potential Observations (self-assessment)
- Requirements are minimal and direct.
- Stub modules removed; no empty Python files (except __init__.py).
- BEV reprojection error is low (0.031 ft), but raw condition number high due to coordinate scaling.
- Test coverage includes unit, integration, baseline, and validation tests.
- CI includes full tests, BEV validation, output validation, and experiment logging.
- Statistical reporting includes skewness, kurtosis, Shapiro-Wilk, bootstrap CIs, and paired t-test script.