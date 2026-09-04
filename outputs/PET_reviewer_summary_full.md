# NNDS PET Output Reviewer Summary

## 1. Overview

This repository contains Post-Encroachment Time (PET) conflict events for two sites:

- **GITI**
- **MRC**


## 2. PET Output Files (Simplified)

- `giti_screened_simplified.csv` — 153 events

- `mrc_screened_simplified.csv` — 34 events

- `combined_screened_simplified.csv` — 187 total events


## 3. Key Statistics

- **Total events**: 187

- **Site counts**: {'GITI': 153, 'MRC': 34}

- **PET mean**: 1.589 s

- **PET median**: 1.566 s

- **PET min / max**: 0.133 s / 2.999 s


### Severity Distribution

- critical (<1.0s): 57

- serious (1.0-1.5s): 32

- moderate (1.5-3.0s): 98

- safe (>3.0s): 0


### Conflict Type Distribution

```json

{
  "other": 78,
  "side_swipe": 38,
  "head_on": 33,
  "crossing": 31,
  "rear_end": 7
}

```


## 4. Methods Box

- **PET definition**: time between first road user leaving conflict region and second entering it.

- **Coordinate system**: world/BEV metres for final outputs; pixel coordinates stored in full trajectories for reference.

- **Conflict zone geometry/size**: configurable; see `uvh_coco_fused_grid_pet.py` for exact zone size used.

- **Frame rate**: 30 FPS (default) unless noted.

- **Overlap policy**: simultaneous-occupancy pairs are excluded from sequential PET.

- **Quality policy**: tracks with gaps/jumps are split; low-coverage occupancy windows handled with validation.


## 5. Column Descriptions

See `PET_data_summary.md` for full column descriptions.


## 6. Directory Tree

```

.github/
    workflows/
        ci.yml
        nightly.yml
baselines/
    constant_acceleration.py
    constant_velocity.py
    kalman_filter.py
    social_force.py
configs/
    sites/
        giti/
            bev_config.json
            calibration_points.json
            gate_config.yaml
            grid_config.json
            provenance.md
            speed_perturbation_sensitivity.json
        mrc/
            bev_config.json
            calibration_points.json
            gate_config.yaml
            grid_config.json
            H_pixel_to_world.npy
            perturbation_sensitivity.json
            provenance.md
            speed_perturbation_sensitivity.json
    bev_config.json
    camera_matrix.npy
    camera_matrix_video_est.npy
    distortion_coeffs.npy
    distortion_coeffs_video_est.npy
    gate_config.yaml
    giti_calibration_points.json
    GITI_grid_config.json
    tracktrack_reid.yaml
    tracktrack_reid_strong.yaml
data/
    sample_data/
        anonymized_traffic_video_50f.mp4
docs/
    data_samples/
        petevents_bev_demo.csv
    figures/
        bev_calibration_geometry.png
        bev_dual_panel.png
        bev_dual_panel_validation.png
        bev_validation_overlay.png
        conflict_type_distribution.png
        dependency_graph.png
        nnds_full_deps.png
        pet_by_conflict_type.png
        pet_distribution.png
        pipeline_architecture.png
        README.md
    bev_detection_validation_results.md
    calibration_provenance.md
    CANONICAL_PIPELINE.md
    CLEANUP_SUMMARY.md
    comprehensive_assessment_report.md
    DEBUGGING.md
    detection_system_report.md
    EXPERIMENTAL_MODULES.md
    final_assessment_report.md
    final_submission_summary.md
    FREE_VLM_MODELS.md
    MIGRATION_GUIDE.md
    MODULE_MANIFEST.md
    mrc_annotated_points.jpg
    mrc_bev_check.jpg
    mrc_bev_check2.jpg
    mrc_bev_check3.jpg
    mrc_bev_check_refined.jpg
    mrc_bev_grid_final.jpg
    mrc_bev_grid_publication.png
    mrc_bev_publication.jpg
    mrc_bev_raw.jpg
    mrc_click_frame.jpg
    mrc_frame_for_gate_annotation.jpg
    mrc_frame_resized.jpg
    mrc_gates_visualized.jpg
    mrc_grid_overlay.jpg
    mrc_points_current.jpg
    mrc_points_with_grid.jpg
    mrc_sample_frame.jpg
    mrc_sample_frame_with_grid.jpg
    PUBLICATION_READINESS.md
    repo_full_details.md
    repository_assessment.md
    REVIEWER_REPORT.md
    scientific_audit.md
    sensitivity_deconfounded.tex
    sensitivity_prediction_tolerance_300f.tex
    sensitivity_table.tex
    STATUS.md
    TEST_MATRIX.md
    tracking_system_report.md
    undistortion_test_estimated.jpg
examples/
    quickstart.py
model_cards/
    uvh26.md
    yolo11n.md
outputs/
    combined_screened_simplified.csv
    final_dual_site_figure.png
    final_screened_summary.json
    giti_ablation_intersection_bev.csv
    giti_full_trajectories.jsonl
    giti_raw.csv
    giti_screened.csv
    giti_screened_simplified.csv
    giti_screened_with_gates.csv
    mrc_ablation_intersection_bev.csv
    mrc_full_trajectories.jsonl
    mrc_raw.csv
    mrc_screened.csv
    mrc_screened_simplified.csv
    mrc_screened_with_gates.csv
    PET_data_summary.md
    PET_reviewer_summary_full.md
    reproducibility_manifest.json
    sample_pet_events_for_review.csv
scripts/
    anonymize_video.py
    bev_heldout_validation.py
    classify_conflict_type_vlm.py
    convert_del4_to_diffusion.py
    convert_pet_to_diffusion_csv.py
    debug_tracking_video.py
    detection_confidence_analysis.py
    diagnose_tracking.py
    download_models.sh
    ensure_models.py
    estimate_time_of_day.py
    evaluate_detection_metrics.py
    evaluate_ground_truth.py
    evaluate_position_ddpm.py
    evaluate_tracking_metrics.py
    evaluate_transformer_diffusion.py
    experiment_logger.py
    export_openvino.py
    extract_event_frames.py
    generate_event_descriptions.py
    generate_results_table.py
    generate_safety_report_groq.py
    grid_search_smoothing.py
    inspect_pet.py
    paired_ttest.py
    reproduce_pipeline.sh
    run_pipeline.py
    run_tracking_baselines.py
    sensitivity_deconfounded.py
    sensitivity_pet_fragmentation.py
    split_detections.py
    tracking_assessment.py
    tracking_full_log.py
    tracking_report.py
    tracking_report_fast.py
    traffic_analyzer_demo.py
    train_position_ddpm.py
    train_transformer_diffusion.py
    validate_all.py
    validate_bev.py
    validate_outputs.py
    validation_report.py
    visualize_pet.py
    visualize_pet_live.py
src/
    analysis/
        audit/
            __init__.py
            audit_config.json
        grid_trajectory/
            __init__.py
            pet_grid.py
            sam3_grid_pet.py
            spatial_grid.py
            uvh_coco_fused_grid_pet.py
            yolo_cpu_grid_pet.py
        logging/
            __init__.py
            reproducibility_audit.py
        ssm/
            __init__.py
            ssm_verification.py
            uncertainty_quantifier.py
        verification/
            __init__.py
            statistical_testing.py
        visualization/
            __init__.py
            industry_standard_viz.py
            pet_diffusion_plots.py
            pet_event_plots.py
            video_overlays.py
        __init__.py
        conflict_classifier.py
        gate_counter.py
        pet_conflict_checker.py
        pet_diffusion_analysis.py
        pet_summary.py
        research_run.py
        safety_eval_diffusion.py
        safety_eval_diffusion_notebook.py
    bev/
        calibration/
            __init__.py
            grid_validation_calibration.py
            MANIFEST.json
            monte_carlo_calibration_benchmark.py
            monte_carlo_calibration_notes.md
            PROVENANCE.md
            README.md
            REPRODUCIBILITY.md
        __init__.py
        bev_mapper.py
        giti_bev_calib.py
    core/
        __init__.py
        types.py
        validation.py
    diffusion/
        traffic_diffusion/
            data/
            __init__.py
            evaluate_fixed.py
            model_and_sampler.py
            mypy.ini
            sampling_utils.py
            split_dataset.py
            train_trajectory_diffusion.py
            training_utils.py
            trajectory_diffusion.py
            transformer_diffusion.py
        __init__.py
        complete_ddpm.py
        traj_diffusion_normalized.py
    pipeline/
        __init__.py
        custom_tracker.py
        reid_encoder.py
        rt_detr_detector.py
        traffic_analyzer.py
    utils/
        __init__.py
        debug_helpers.py
        interactive.py
        seed.py
    vlm/
        utils/
            __init__.py
            image_utils.py
            visualization.py
        __init__.py
        analyzer.py
        config.py
        gate_validator.py
        requirements.txt
        test_free_models.py
        vlm_enhanced_pipeline.py
    __init__.py
tests/
    __snapshots__/
        test_snapshot_bev_mapper.ambr
        test_snapshot_pet_summary.ambr
    fixtures/
        sample_detections.csv
        sample_pet.csv
        sample_split_detections.csv
    __init__.py
    conftest.py
    test_analysis_init_imports.py
    test_baselines_extra.py
    test_baselines_missing.py
    test_baselines_seed.py
    test_bev_calibration.py
    test_bev_mapper.py
    test_bev_validation.py
    test_configs_smoke.py
    test_conflict_classifier.py
    test_core_small_modules.py
    test_custom_tracker.py
    test_diffusion_smoke.py
    test_event_utilities.py
    test_gate_counter.py
    test_gate_counter_extra.py
    test_gate_counter_full.py
    test_giti_bev_calib.py
    test_giti_bev_calib_full.py
    test_grid_trajectory_init_imports.py
    test_grid_validation_calibration.py
    test_heavy_smoke.py
    test_import_all_zero_coverage.py
    test_imports_smoke.py
    test_industry_standard_viz.py
    test_metrics_scripts.py
    test_modules_smoke.py
    test_monte_carlo_calibration_benchmark.py
    test_new_metrics_scripts.py
    test_new_scripts.py
    test_paired_ttest.py
    test_pet_computation.py
    test_pet_conflict.py
    test_pet_conflict_checker.py
    test_pet_conflict_checker_extra.py
    test_pet_conflict_checker_full.py
    test_pet_grid_full.py
    test_pet_logic.py
    test_pet_output_schema.py
    test_pet_summary.py
    test_pet_summary_full.py
    test_pet_velocity.py
    test_property_based_more.py
    test_reid_encoder.py
    test_repo_smoke.py
    test_reproducibility_audit.py
    test_reproducibility_audit_full.py
    test_research_run_smoke.py
    test_rtdetr_stub.py
    test_savgol_velocity.py
    test_scientific_invariants.py
    test_smoke.py
    test_snapshot_bev_mapper.py
    test_snapshot_pet_summary.py
    test_spatial_grid.py
    test_speed_estimation.py
    test_splitter_wiring.py
    test_ssm_verification.py
    test_ssm_verification_full.py
    test_statistical_testing.py
    test_statistical_testing_full.py
    test_time_of_day.py
    test_traffic_analyzer.py
    test_traffic_analyzer_100.py
    test_traffic_analyzer_cli.py
    test_traffic_analyzer_demo_smoke.py
    test_traffic_analyzer_full.py
    test_traffic_analyzer_missing_coverage.py
    test_uncertainty_quantifier.py
    test_uncertainty_quantifier_full.py
    test_uvh_coco_fused_grid_pet.py
    test_validate_outputs.py
    test_validation.py
    test_video_overlays.py
    test_visualization_modules.py
    test_vlm_analyzer_mock.py
    test_vlm_config.py
.gitattributes
.gitignore
.pre-commit-config.yaml
CHANGELOG.md
CITATION.cff
CONTRIBUTING.md
DATA_LICENSE
docker-compose.yml
Dockerfile
environment.yml
LICENSE
Makefile
PRIVACY.md
pyproject.toml
pytest.ini
README.md
requirements-vlm.txt
requirements.txt
TESTING.md

```
