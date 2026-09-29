# Repository Quality Report

Generated at: 2026-09-02 18:39:16

## 1. General Information

- **Branch:** `cleanup/system-reorganization`
- **Latest commit:** `37ee6bca3987be70cb9e0b873caf9d8d94eb4c2d`

## 2. Directory Tree (excluding .git, caches)

```
docs
  mrc_bev_raw.jpg
  final_submission_summary.md
  mrc_bev_grid_final.jpg
  undistortion_test_estimated.jpg
  DEBUGGING.md
  repository_assessment.md
  EXPERIMENTAL_MODULES.md
  mrc_sample_frame_with_grid.jpg
  mrc_frame_for_gate_annotation.jpg
  mrc_bev_check_refined.jpg
  TEST_MATRIX.md
  mrc_points_with_grid.jpg
  sensitivity_table.tex
  mrc_bev_grid_publication.png
  final_assessment_report.md
  MIGRATION_GUIDE.md
  tracking_system_report.md
  REVIEWER_REPORT.md
  scientific_audit.md
  mrc_annotated_points.jpg
  CLEANUP_SUMMARY.md
  mrc_bev_publication.jpg
  mrc_click_frame.jpg
  mrc_points_current.jpg
  FREE_VLM_MODELS.md
  sensitivity_deconfounded.tex
  PUBLICATION_READINESS.md
  bev_detection_validation_results.md
  CANONICAL_PIPELINE.md
  mrc_bev_check.jpg
  mrc_grid_overlay.jpg
  comprehensive_assessment_report.md
  data_samples
    petevents_bev_demo.csv
  mrc_frame_resized.jpg
  calibration_provenance.md
  STATUS.md
  mrc_bev_check2.jpg
  mrc_sample_frame.jpg
  mrc_bev_check3.jpg
  mrc_gates_visualized.jpg
  sensitivity_prediction_tolerance_300f.tex
  detection_system_report.md
  MODULE_MANIFEST.md
  repo_full_details.md
  figures
    bev_dual_panel_validation.png
    nnds_full_deps.png
    bev_calibration_geometry.png
    pipeline_architecture.png
    bev_dual_panel.png
    pet_by_conflict_type.png
    pet_distribution.png
    bev_validation_overlay.png
    dependency_graph.png
    README.md
    conflict_type_distribution.png
examples
  quickstart.py
Makefile
outputs
  giti_screened.csv
  mrc_raw.csv
  giti_screened_with_gates.csv
  mrc_screened.csv
  mrc_ablation_intersection_bev.csv
  giti_ablation_intersection_bev.csv
  final_screened_summary.json
  petevents_bev.csv
  giti_raw.csv
  reproducibility_manifest.json
  final_dual_site_figure.png
  logs
    pet_conflicts_20260902_180010.log
  mrc_screened_with_gates.csv
src
  core
    types.py
    __init__.py
    __pycache__
    validation.py
  __init__.py
  vlm
    config.py
    requirements.txt
    test_free_models.py
    __init__.py
    gate_validator.py
    analyzer.py
    utils
      __init__.py
      visualization.py
      __pycache__
      image_utils.py
    __pycache__
    vlm_enhanced_pipeline.py
  utils
    debug_helpers.py
    interactive.py
    __init__.py
    __pycache__
    seed.py
  __pycache__
  analysis
    research_run.py
    logging
      __init__.py
      reproducibility_audit.py
      __pycache__
    audit
      __init__.py
      audit_config.json
    safety_eval_diffusion.py
    conflict_classifier.py
    gate_counter.py
    pet_diffusion_analysis.py
    ssm
      uncertainty_quantifier.py
      __init__.py
      __pycache__
      ssm_verification.py
    visualization
      pet_event_plots.py
      __init__.py
      video_overlays.py
      industry_standard_viz.py
      __pycache__
      pet_diffusion_plots.py
    pet_summary.py
    __init__.py
    grid_trajectory
      spatial_grid.py
      sam3_grid_pet.py
      pet_grid.py
      __init__.py
      uvh_coco_fused_grid_pet.py
      __pycache__
      yolo_cpu_grid_pet.py
    safety_eval_diffusion_notebook.py
    __pycache__
    pet_conflict_checker.py
    verification
      statistical_testing.py
      __init__.py
      __pycache__
  pipeline
    custom_tracker.py
    rt_detr_detector.py
    __init__.py
    reid_encoder.py
    traffic_analyzer.py
    __pycache__
  bev
    bev_mapper.py
    giti_bev_calib.py
    __init__.py
    __pycache__
    calibration
      REPRODUCIBILITY.md
      monte_carlo_calibration_notes.md
      __init__.py
      monte_carlo_calibration_benchmark.py
      MANIFEST.json
      __pycache__
      PROVENANCE.md
      grid_validation_calibration.py
      README.md
  diffusion
    traj_diffusion_normalized.py
    traffic_diffusion
      trajectory_diffusion.py
      evaluate_fixed.py
      mypy.ini
      __init__.py
      train_trajectory_diffusion.py
      transformer_diffusion.py
      split_dataset.py
      __pycache__
      training_utils.py
      model_and_sampler.py
      sampling_utils.py
      data
        trajdiff_inputs.npy
        __init__.py
        trajdiff_targets.npy
        trajdiff_meta.parquet
    __init__.py
    __pycache__
    complete_ddpm.py
output
configs
  GITI_grid_config.json
  sites
    mrc
      speed_perturbation_sensitivity.json
      grid_config.json
      calibration_points.json
      H_pixel_to_world.npy
      perturbation_sensitivity.json
      gate_config.yaml
      provenance.md
      bev_config.json
    giti
      speed_perturbation_sensitivity.json
      grid_config.json
      calibration_points.json
      gate_config.yaml
      provenance.md
      bev_config.json
  camera_matrix_video_est.npy
  camera_matrix.npy
  tracktrack_reid_strong.yaml
  tracktrack_reid.yaml
  giti_calibration_points.json
  distortion_coeffs_video_est.npy
  distortion_coeffs.npy
  gate_config.yaml
  bev_config.json
tests
  test_gate_counter_full.py
  test_modules_smoke.py
  test_traffic_analyzer.py
  test_property_based_more.py
  test_rtdetr_stub.py
  test_reid_encoder.py
  test_statistical_testing.py
  test_splitter_wiring.py
  test_snapshot_pet_summary.py
  test_pet_output_schema.py
  test_giti_bev_calib.py
  test_imports_smoke.py
  test_repo_smoke.py
  test_event_utilities.py
  test_grid_validation_calibration.py
  test_pet_velocity.py
  test_validation.py
  test_grid_trajectory_init_imports.py
  test_vlm_analyzer_mock.py
  test_pet_summary_full.py
  test_pet_summary.py
  test_video_overlays.py
  test_diffusion_smoke.py
  test_savgol_velocity.py
  test_giti_bev_calib_full.py
  test_import_all_zero_coverage.py
  test_ssm_verification.py
  test_snapshot_bev_mapper.py
  test_research_run_smoke.py
  test_scientific_invariants.py
  test_vlm_config.py
  test_time_of_day.py
  test_spatial_grid.py
  test_traffic_analyzer_cli.py
  conftest.py
  test_pet_conflict_checker.py
  test_traffic_analyzer_full.py
  test_baselines_extra.py
  test_industry_standard_viz.py
  fixtures
    sample_pet.csv
    sample_split_detections.csv
    sample_detections.csv
  test_custom_tracker.py
  test_pet_conflict_checker_full.py
  test_pet_grid_full.py
  test_core_small_modules.py
  test_analysis_init_imports.py
  test_gate_counter.py
  __init__.py
  test_conflict_classifier.py
  __snapshots__
    test_snapshot_pet_summary.ambr
    test_snapshot_bev_mapper.ambr
  test_reproducibility_audit_full.py
  test_heavy_smoke.py
  test_uvh_coco_fused_grid_pet.py
  test_statistical_testing_full.py
  test_gate_counter_extra.py
  test_baselines_seed.py
  test_traffic_analyzer_100.py
  test_bev_validation.py
  test_bev_mapper.py
  test_validate_outputs.py
  test_configs_smoke.py
  test_paired_ttest.py
  test_ssm_verification_full.py
  test_pet_conflict.py
  test_bev_calibration.py
  test_baselines_missing.py
  __pycache__
  test_new_metrics_scripts.py
  test_monte_carlo_calibration_benchmark.py
  test_pet_conflict_checker_extra.py
  test_traffic_analyzer_demo_smoke.py
  test_new_scripts.py
  test_pet_logic.py
  test_uncertainty_quantifier_full.py
  test_speed_estimation.py
  test_visualization_modules.py
  test_smoke.py
  test_traffic_analyzer_missing_coverage.py
  test_uncertainty_quantifier.py
  test_reproducibility_audit.py
  test_pet_computation.py
  test_metrics_scripts.py
LICENSE
pytest.ini
model_cards
  yolo11n.md
  uvh26.md
requirements.txt
CONTRIBUTING.md
PRIVACY.md
CITATION.cff
.gitattributes
.pre-commit-config.yaml
environment.yml
DATA_LICENSE
pyproject.toml
CHANGELOG.md
docker-compose.yml
TESTING.md
scripts
  traffic_analyzer_demo.py
  tracking_full_log.py
  sensitivity_pet_fragmentation.py
  inspect_pet.py
  anonymize_video.py
  evaluate_tracking_metrics.py
  sensitivity_deconfounded.py
  paired_ttest.py
  validate_outputs.py
  evaluate_transformer_diffusion.py
  train_transformer_diffusion.py
  export_openvino.py
  evaluate_position_ddpm.py
  classify_conflict_type_vlm.py
  convert_pet_to_diffusion_csv.py
  diagnose_tracking.py
  run_tracking_baselines.py
  ensure_models.py
  convert_del4_to_diffusion.py
  split_detections.py
  extract_event_frames.py
  generate_safety_report_groq.py
  generate_event_descriptions.py
  evaluate_ground_truth.py
  tracking_report.py
  run_pipeline.py
  visualize_pet_live.py
  train_position_ddpm.py
  grid_search_smoothing.py
  estimate_time_of_day.py
  reproduce_pipeline.sh
  detection_confidence_analysis.py
  tracking_report_fast.py
  debug_tracking_video.py
  __pycache__
  validate_all.py
  bev_heldout_validation.py
  evaluate_detection_metrics.py
  validation_report.py
  experiment_logger.py
  validate_bev.py
  download_models.sh
  generate_results_table.py
  visualize_pet.py
  tracking_assessment.py
Dockerfile
requirements-vlm.txt
.gitignore
.github
  workflows
    ci.yml
    nightly.yml
README.md
baselines
  kalman_filter.py
  social_force.py
  constant_acceleration.py
  __pycache__
  constant_velocity.py
data
  sample_data
    anonymized_traffic_video_50f.mp4
.coverage
```

## 3. Test Collection Summary

- **Total collected tests:** 940 tests collected in 19.22s
- **Collection return code:** 0

- **Test files:** 78

## 4. Coverage Configuration

```ini
[pytest]
addopts = --cov=src --cov-report=term-missing --cov-branch
markers =
    integration: heavy tests requiring real data or external services
    slow: tests that take long to run
    smoke: smoke tests for heavy modules
testpaths = tests
```

## 5. CI/CD Workflows

### `.github/workflows/ci.yml`

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

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pytest pytest-cov pytest-timeout hypothesis pytest-mock syrupy
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

      - name: Run fast unit tests with coverage
        run: |
          pytest tests/ -m "not integration and not slow" \
            --cov=src --cov-branch --cov-report=term-missing \
            --cov-fail-under=84

      - name: Run smoke tests (integration)
        run: pytest tests/ -m "integration or smoke" -q
```

### `.github/workflows/nightly.yml`

```yaml
name: Nightly Integration

on:
  schedule:
    - cron: '0 3 * * *'  # 3 AM UTC daily
  workflow_dispatch:

jobs:
  integration:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-timeout hypothesis pytest-mock syrupy
      - name: Run integration and smoke tests
        run: |
          pytest tests/ -m "integration or smoke" -q
```

## 6. Testing Documentation

# Testing Strategy

## Coverage Overview
- Current unit test coverage on core testable modules: ~99% statement coverage, ~95% branch coverage.
- Heavy modules (diffusion training, VLM, video processing) are covered by smoke tests and integration tests, not full unit tests.

## Excluded/Untestable Modules
| Module | Reason |
|--------|--------|
| `src/diffusion/*` | Requires GPU and long training runs |
| `src/vlm/*` | Requires API keys or large models |
| `src/analysis/grid_trajectory/sam3_grid_pet.py` | Requires SAM3 model and video |
| `src/analysis/grid_trajectory/yolo_cpu_grid_pet.py` | Requires YOLO model and video |
| `src/utils/interactive.py` | Interactive display functions |
| `src/utils/debug_helpers.py` | Debug printing and debugging helpers |

## Mutation Testing
The installed `mutmut` version (3.7.0) does not support `--paths-to-mutate`.
A manual mutation test can be performed as follows:

1. Copy a core file, e.g., `src/analysis/conflict_classifier.py`.
2. Introduce a small mutation (e.g., change `<` to `>`).
3. Run the relevant test file.
4. Confirm tests fail; revert the mutation.

Example manual mutation:
- Original: `if pet < 1.0: return ConflictSeverity.CRITICAL`
- Mutated: `if pet > 1.0: return ConflictSeverity.CRITICAL`
- Expected: tests should fail.

## CI
GitHub Actions runs the fast unit suite (excluding integration/slow tests) on every push and PR with:
```bash
pytest tests/ -m "not integration and not slow" --cov=src --cov-branch --cov-fail-under=84
```

## Property-Based Testing
Uses Hypothesis to test invariants in PET computation and severity classification. See `tests/test_property_based.py`.

## 7. Canonical Pipeline Documentation

# Canonical PET Pipeline

The authoritative workflow for final PET event CSV and figures.

## Command

```bash
make reproduce-final
```

This runs:
- `python src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py --detector uvh-coco-fused`
- `python scripts/validate_outputs.py`
- `python scripts/generate_results_table.py`

## Coordinate System

Current PET uses a **pixel-space conflict zone** (20 px radius). This is a trajectory-based PET proxy, not a physically calibrated BEV PET. Document this in any manuscript.

## Outputs

Actual final outputs in `outputs/`:
- `giti_screened.csv`
- `mrc_screened.csv`
- `giti_screened_with_gates.csv`
- `mrc_screened_with_gates.csv`
- `giti_raw.csv`
- `mrc_raw.csv`
- `final_screened_summary.json`
- `final_dual_site_figure.png`
- `reproducibility_manifest.json`
- `petevents_bev.csv`

The `make reproduce-final` target is expected to generate these files.

## 8. Contributing Guide

# Contributing to NNDS

We welcome contributions! Please follow these guidelines:

## Reporting Issues
- Use the GitHub issue tracker.
- Provide a clear description and steps to reproduce.

## Pull Requests
- Fork the repository and create a new branch.
- Ensure your code passes all tests (`pytest`).
- Format your code with `black` and `isort`.
- Write clear commit messages.

## Development Setup

```bash
pip install -e .[dev]
pre-commit install
```

## Code Style
- We follow PEP 8.
- Use `black` (line length 100) and `isort`.

## Testing
- Write tests for new features.
- Run `pytest` to ensure no regressions.

Thank you for contributing!

## 9. Recent Git History

37ee6bc Generate updated reviewer report
8cee185 Align canonical pipeline outputs with actual generated files
828a17f Fix collection error by removing duplicate property test alias
4ebeef7 Fix CI coverage guard, align Python versions, add missing tests, canonical docs
f0700a9 Add nightly CI workflow and fix property-based test import
0ee5e3e Add property-based and VLM mock tests, snapshot tests, CI workflows, and testing docs
e19f34f Add VLM mocks, BEV snapshot, nightly CI, and mutation results
2e9e368 Add snapshot test for PET summary and adjust pytest.ini
f16a468 Add CI workflow, testing docs, smoke and property-based tests
9014a7f Achieve 100% coverage for pet_conflict_checker
