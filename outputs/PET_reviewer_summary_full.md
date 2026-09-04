# NNDS PET Output Reviewer Summary

## 1. Overview

This repository contains Post-Encroachment Time (PET) conflict events derived from two sites:

- **GITI** (giti)

- **MRC** (mrc)


Processed outputs are available in the `outputs/` directory.


## 2. PET Output Files

### Simplified CSVs (ready for quick review)

- `giti_screened_simplified.csv` – 153 events

- `mrc_screened_simplified.csv` – 34 events

- `combined_screened_simplified.csv` – 187 total events


### Full Trajectories

- `giti_full_trajectories.jsonl`

- `mrc_full_trajectories.jsonl`


### Raw / Gate-Annotated (for reproducibility)

- `giti_raw.csv`, `mrc_raw.csv`

- `giti_screened_with_gates.csv`, `mrc_screened_with_gates.csv`

- `giti_ablation_intersection_bev.csv`, `mrc_ablation_intersection_bev.csv`

- `final_screened_summary.json`


## 3. Key Statistics (combined screened dataset)

- **Total events**: 187

- **PET mean**: 1.589 s

- **PET median**: 1.566 s

- **PET min / max**: 0.133 s / 2.999 s


### Severity Distribution (based on PET)

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


## 4. Column Descriptions (Simplified CSVs)

| Column | Description |

|--------|-------------|

| `event_id` | Unique event identifier |

| `pet` | Post-Encroachment Time (seconds) |

| `pet_time_based` | Time-based PET (seconds) |

| `frame` | Conflict frame number |

| `track_a` | Track ID of vehicle A |

| `track_b` | Track ID of vehicle B |

| `orig_track_a` | Original track ID for A before splitting |

| `seg_a` | Segment index for A |

| `orig_track_b` | Original track ID for B before splitting |

| `seg_b` | Segment index for B |

| `conflict_type` | Geometric conflict classification |

| `grid_cell` | Grid cell where conflict occurred |

| `track_a_entry_frame` | Frame when A entered cell |

| `track_a_exit_frame` | Frame when A exited cell |

| `track_a_exit_time_sec` | Exit time for A (seconds) |

| `track_b_entry_frame` | Frame when B entered cell |

| `track_b_entry_time_sec` | Entry time for B (seconds) |

| `track_b_exit_frame` | Frame when B exited cell |

| `video_source` | Source site label (giti/mrc) |

| `time_of_day_label` | Time of day label |

| `gate_a_entry` | Entry gate for A |

| `gate_b_entry` | Entry gate for B |


## 5. Directory Tree (top levels)

```

.github/
    workflows/
        ci.yml
        nightly.yml
.hypothesis/
    constants/
        0757442185fcb769
        0993501df3f0942e
        0d18d738dd725a6f
        19dc3f45591f8fb9
        1a3f1197b69522e9
        1d139ad9dc5f034b
        1d4f85dfd32e718c
        1db8ebf3a3bc86f7
        1ff7e4e90a77b6ce
        24d9f18b0cd4f2c6
        25872e25ee63d43b
        2e47045688d3293c
        370731f937770942
        3801528f4a84334e
        383d104fbd1ad768
        39f7db5fdb904788
        3b0248fb9b24fca5
        3bd642dede8b5ea2
        4434950fa8d57404
        46bd39f0a861cf08
        46c685f4e3b44ede
        49d933f7220fb09f
        4a866f4c7b1f01b2
        4ff25df38e027fa6
        5172d903824edae5
        5247f36489093626
        54f54cbc5ae97301
        5b80943ff5e5548a
        6142636251f9a04b
        622c8148b6b504a9
        652a23c207a37b78
        68c722ad3f026fd9
        6bf5eed95875b987
        708cf2cf32f51ea7
        7a6985d4b58782c3
        7d7b9464c3e361ae
        7e92f7c91ddc16ad
        85e5fc4d7b1adcda
        878790e2baac6f6a
        8934d3d99b988089
        89678e7d53b3f4a5
        898f209a7d0c805f
        8af928072745381e
        9079f3a2ec1b5fa5
        97a1bf9cf98d828b
        9867061af866e66a
        9eb49533bb2d55d6
        a039849360ac4db7
        a1f75194639cb21e
        ab380059d2500602
        adff5efb9bc2e5f4
        b1b07702d1777f1b
        b99e74ff1b367c2e
        bb38c7361ca1a464
        bd229f63bc894a43
        c14376ab5c94a1e5
        cc7b8eda2131f611
        d0a29b9c086ca1fd
        d24bf2f2a9f957e9
        d470eb37bfe7c99e
        da39a3ee5e6b4b0d
        dc33e5b750416fd7
        e557e1c8f197ba04
        e600719b5ff1210c
        e7bcfa6a301a817e
        ec003401bf517a99
        ef1e1f68a945d7e6
        f150cda2b34f3038
        f3c8646b8604f695
        f7e683ee895a71a9
        f9a4c45492b61675
        f9be518bb3518d2a
        fc998c0560a826a7
        fcabfb9d96330029
        fef758bb97783a50
    .gitignore
.pytest_cache/
    v/
        cache/
            lastfailed
            nodeids
    .gitignore
    CACHEDIR.TAG
    README.md
baselines/
    __pycache__/
        constant_acceleration.cpython-313.pyc
        constant_velocity.cpython-313.pyc
        kalman_filter.cpython-313.pyc
        social_force.cpython-313.pyc
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
    NonTechnical_PET_Explainer.md
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
output/
outputs/
    logs/
        pet_conflicts_20260904_082059.log
        pet_conflicts_20260904_084510.log
        pet_conflicts_20260904_084753.log
        pet_conflicts_20260904_084754.log
        pet_conflicts_20260904_092027.log
        pet_conflicts_20260904_094244.log
        pet_conflicts_20260904_094245.log
        pet_conflicts_20260904_095314.log
        pet_conflicts_20260904_101256.log
        pet_conflicts_20260904_101934.log
        pet_conflicts_20260904_104846.log
        pet_conflicts_20260904_104847.log
        pet_conflicts_20260904_110715.log
        pet_conflicts_20260904_110716.log
        pet_conflicts_20260904_112147.log
        pet_conflicts_20260904_113257.log
        pet_conflicts_20260904_113554.log
        pet_conflicts_20260904_114201.log
        pet_conflicts_20260904_114202.log
        pet_conflicts_20260904_115025.log
        pet_conflicts_20260904_115356.log
        pet_conflicts_20260904_115357.log
        pet_conflicts_20260904_120228.log
        pet_conflicts_20260904_120618.log
        pet_conflicts_20260904_120619.log
        pet_conflicts_20260904_123602.log
        pet_conflicts_20260904_125223.log
        pet_conflicts_20260904_125224.log
        pet_conflicts_20260904_130401.log
        pet_conflicts_20260904_130402.log
    reviewer_traffic_frames/
        frame_000.jpg
        frame_001.jpg
        frame_002.jpg
        frame_003.jpg
        frame_004.jpg
        frame_005.jpg
        frame_006.jpg
        frame_007.jpg
        frame_008.jpg
        frame_009.jpg
        frame_010.jpg
        frame_011.jpg
        frame_012.jpg
        frame_013.jpg
        frame_014.jpg
        frame_015.jpg
        frame_016.jpg
        frame_017.jpg
        frame_018.jpg
        frame_019.jpg
        frame_020.jpg
        frame_021.jpg
        frame_022.jpg
        frame_023.jpg
        frame_024.jpg
        frame_025.jpg
        frame_026.jpg
        frame_027.jpg
        frame_028.jpg
        frame_029.jpg
        frame_030.jpg
        frame_031.jpg
        frame_032.jpg
        frame_033.jpg
        frame_034.jpg
        frame_035.jpg
        frame_036.jpg
        frame_037.jpg
        frame_038.jpg
        frame_039.jpg
        frame_040.jpg
        frame_041.jpg
        frame_042.jpg
        frame_043.jpg
        frame_044.jpg
        frame_045.jpg
        frame_046.jpg
        frame_047.jpg
        frame_048.jpg
        frame_049.jpg
    verification_videos/
        event_0073.mp4
        event_0094.mp4
        event_0124.mp4
        event_0137.mp4
        event_0137_test.mp4
        event_0153.mp4
    verification_videos_mrc/
        mrc_event_0018.mp4
        mrc_event_0019.mp4
        mrc_event_0021.mp4
        mrc_event_0025.mp4
        mrc_event_0028.mp4
    combined_screened_simplified.csv
    final_dual_site_figure.png
    final_screened_summary.json
    giti_ablation_intersection_bev.csv
    giti_full_trajectories.jsonl
    giti_merged_for_visualization.csv
    giti_raw.csv
    giti_screened.csv
    giti_screened_simplified.csv
    giti_screened_with_gates.csv
    mrc_ablation_intersection_bev.csv
    mrc_full_trajectories.jsonl
    mrc_merged_for_visualization.csv
    mrc_raw.csv
    mrc_screened.csv
    mrc_screened_simplified.csv
    mrc_screened_with_gates.csv
    PET_data_summary.md
    PET_reviewer_summary_full.md
    pet_verification_event_video.mp4
    PET_Verification_Report.md
    petevents_bev.csv
    reproducibility_manifest.json
    reviewer_traffic_frames.zip
    reviewer_traffic_frames_manifest.json
    sample_pet_events_for_review.csv
    verification_videos.zip
    verification_videos_mrc.zip
scripts/
    __pycache__/
        validate_outputs.cpython-313.pyc
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
    generate_pet_verification_video.py
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
    __pycache__/
        __init__.cpython-313.pyc
    analysis/
        __pycache__/
            __init__.cpython-313.pyc
            conflict_classifier.cpython-313.pyc
            gate_counter.cpython-313.pyc
            pet_conflict_checker.cpython-313.pyc
            pet_diffusion_analysis.cpython-313.pyc
            pet_summary.cpython-313.pyc
            research_run.cpython-313.pyc
            safety_eval_diffusion.cpython-313.pyc
        audit/
            __init__.py
            audit_config.json
        grid_trajectory/
            __pycache__/
            __init__.py
            pet_grid.py
            sam3_grid_pet.py
            spatial_grid.py
            uvh_coco_fused_grid_pet.py
            yolo_cpu_grid_pet.py
        logging/
            __pycache__/
            __init__.py
            reproducibility_audit.py
        ssm/
            __pycache__/
            __init__.py
            ssm_verification.py
            uncertainty_quantifier.py
        verification/
            __pycache__/
            __init__.py
            statistical_testing.py
        visualization/
            __pycache__/
            __init__.py
            industry_standard_viz.py
            pet_diffusion_plots.py
            pet_event_plots.py
            pet_verification_visualizer.py
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
        __pycache__/
            __init__.cpython-313.pyc
            bev_mapper.cpython-313.pyc
            giti_bev_calib.cpython-313.pyc
        calibration/
            __pycache__/
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
        __pycache__/
            __init__.cpython-313.pyc
            types.cpython-313.pyc
            validation.cpython-313.pyc
        __init__.py
        types.py
        validation.py
    diffusion/
        __pycache__/
            __init__.cpython-313.pyc
            complete_ddpm.cpython-313.pyc
            traj_diffusion_normalized.cpython-313.pyc
        traffic_diffusion/
            __pycache__/
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
        __pycache__/
            __init__.cpython-313.pyc
            custom_tracker.cpython-313.pyc
            reid_encoder.cpython-313.pyc
            rt_detr_detector.cpython-313.pyc
            traffic_analyzer.cpython-313.pyc
        __init__.py
        custom_tracker.py
        reid_encoder.py
        rt_detr_detector.py
        traffic_analyzer.py
    utils/
        __pycache__/
            __init__.cpython-313.pyc
            debug_helpers.cpython-313.pyc
            interactive.cpython-313.pyc
            seed.cpython-313.pyc
        __init__.py
        debug_helpers.py
        interactive.py
        seed.py
    vlm/
        __pycache__/
            __init__.cpython-313.pyc
            analyzer.cpython-313.pyc
            config.cpython-313.pyc
            gate_validator.cpython-313.pyc
            test_free_models.cpython-313-pytest-8.4.2.pyc
            vlm_enhanced_pipeline.cpython-313.pyc
        utils/
            __pycache__/
            __init__.py
            image_utils.py
            visualization.py
        __init__.py
        analyzer.py
        config.py
        gate_validator.py
        requirements.txt
        vlm_enhanced_pipeline.py
    __init__.py
tests/
    __pycache__/
        __init__.cpython-313.pyc
        conftest.cpython-313-pytest-8.4.2.pyc
        test_analysis_init_imports.cpython-313-pytest-8.4.2.pyc
        test_baselines_extra.cpython-313-pytest-8.4.2.pyc
        test_baselines_missing.cpython-313-pytest-8.4.2.pyc
        test_baselines_seed.cpython-313-pytest-8.4.2.pyc
        test_bev_calibration.cpython-313-pytest-8.4.2.pyc
        test_bev_mapper.cpython-313-pytest-8.4.2.pyc
        test_bev_validation.cpython-313-pytest-8.4.2.pyc
        test_configs_smoke.cpython-313-pytest-8.4.2.pyc
        test_conflict_classifier.cpython-313-pytest-8.4.2.pyc
        test_core_small_modules.cpython-313-pytest-8.4.2.pyc
        test_custom_tracker.cpython-313-pytest-8.4.2.pyc
        test_diffusion_smoke.cpython-313-pytest-8.4.2.pyc
        test_event_utilities.cpython-313-pytest-8.4.2.pyc
        test_gate_counter.cpython-313-pytest-8.4.2.pyc
        test_gate_counter_extra.cpython-313-pytest-8.4.2.pyc
        test_gate_counter_full.cpython-313-pytest-8.4.2.pyc
        test_giti_bev_calib.cpython-313-pytest-8.4.2.pyc
        test_giti_bev_calib_full.cpython-313-pytest-8.4.2.pyc
        test_grid_trajectory_init_imports.cpython-313-pytest-8.4.2.pyc
        test_grid_validation_calibration.cpython-313-pytest-8.4.2.pyc
        test_heavy_smoke.cpython-313-pytest-8.4.2.pyc
        test_import_all_zero_coverage.cpython-313-pytest-8.4.2.pyc
        test_imports_smoke.cpython-313-pytest-8.4.2.pyc
        test_industry_standard_viz.cpython-313-pytest-8.4.2.pyc
        test_metrics_scripts.cpython-313-pytest-8.4.2.pyc
        test_modules_smoke.cpython-313-pytest-8.4.2.pyc
        test_monte_carlo_calibration_benchmark.cpython-313-pytest-8.4.2.pyc
        test_new_metrics_scripts.cpython-313-pytest-8.4.2.pyc
        test_new_scripts.cpython-313-pytest-8.4.2.pyc
        test_outputs_validation.cpython-313-pytest-8.4.2.pyc
        test_paired_ttest.cpython-313-pytest-8.4.2.pyc
        test_pet_computation.cpython-313-pytest-8.4.2.pyc
        test_pet_conflict.cpython-313-pytest-8.4.2.pyc
        test_pet_conflict_checker.cpython-313-pytest-8.4.2.pyc
        test_pet_conflict_checker_extra.cpython-313-pytest-8.4.2.pyc
        test_pet_conflict_checker_full.cpython-313-pytest-8.4.2.pyc
        test_pet_grid_full.cpython-313-pytest-8.4.2.pyc
        test_pet_logic.cpython-313-pytest-8.4.2.pyc
        test_pet_output_schema.cpython-313-pytest-8.4.2.pyc
        test_pet_summary.cpython-313-pytest-8.4.2.pyc
        test_pet_summary_full.cpython-313-pytest-8.4.2.pyc
        test_pet_velocity.cpython-313-pytest-8.4.2.pyc
        test_pet_verification_visualizer.cpython-313-pytest-8.4.2.pyc
        test_property_based_more.cpython-313-pytest-8.4.2.pyc
        test_reid_encoder.cpython-313-pytest-8.4.2.pyc
        test_repo_smoke.cpython-313-pytest-8.4.2.pyc
        test_reproducibility_audit.cpython-313-pytest-8.4.2.pyc
        test_reproducibility_audit_full.cpython-313-pytest-8.4.2.pyc
        test_research_run_smoke.cpython-313-pytest-8.4.2.pyc
        test_rtdetr_stub.cpython-313-pytest-8.4.2.pyc
        test_savgol_velocity.cpython-313-pytest-8.4.2.pyc
        test_scientific_invariants.cpython-313-pytest-8.4.2.pyc
        test_scripts_smoke.cpython-313-pytest-8.4.2.pyc
        test_smoke.cpython-313-pytest-8.4.2.pyc
        test_spatial_grid.cpython-313-pytest-8.4.2.pyc
        test_speed_estimation.cpython-313-pytest-8.4.2.pyc
        test_splitter_wiring.cpython-313-pytest-8.4.2.pyc
        test_ssm_verification.cpython-313-pytest-8.4.2.pyc
        test_ssm_verification_full.cpython-313-pytest-8.4.2.pyc
        test_statistical_testing.cpython-313-pytest-8.4.2.pyc
        test_statistical_testing_full.cpython-313-pytest-8.4.2.pyc
        test_time_of_day.cpython-313-pytest-8.4.2.pyc
        test_traffic_analyzer.cpython-313-pytest-8.4.2.pyc
        test_traffic_analyzer_100.cpython-313-pytest-8.4.2.pyc
        test_traffic_analyzer_cli.cpython-313-pytest-8.4.2.pyc
        test_traffic_analyzer_demo_smoke.cpython-313-pytest-8.4.2.pyc
        test_traffic_analyzer_full.cpython-313-pytest-8.4.2.pyc
        test_traffic_analyzer_missing_coverage.cpython-313-pytest-8.4.2.pyc
        test_uncertainty_quantifier.cpython-313-pytest-8.4.2.pyc
        test_uncertainty_quantifier_full.cpython-313-pytest-8.4.2.pyc
        test_uvh_coco_fused_grid_pet.cpython-313-pytest-8.4.2.pyc
        test_validate_outputs.cpython-313-pytest-8.4.2.pyc
        test_validation.cpython-313-pytest-8.4.2.pyc
        test_video_overlays.cpython-313-pytest-8.4.2.pyc
        test_visualization_modules.cpython-313-pytest-8.4.2.pyc
        test_vlm_analyzer_mock.cpython-313-pytest-8.4.2.pyc
        test_vlm_config.cpython-313-pytest-8.4.2.pyc
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
    test_outputs_validation.py
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
    test_pet_verification_visualizer.py
    test_property_based_more.py
    test_reid_encoder.py
    test_repo_smoke.py
    test_reproducibility_audit.py
    test_reproducibility_audit_full.py
    test_research_run_smoke.py
    test_rtdetr_stub.py
    test_savgol_velocity.py
    test_scientific_invariants.py
    test_scripts_smoke.py
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
    vlm_free_models_check.py
.coverage
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


## 6. Notes for Reviewers

- Full trajectory columns (`world_traj_i`, `world_traj_j`, `traj_a_json`, `traj_b_json`) are excluded from simplified CSVs for readability.

- Complete trajectories are preserved in `.jsonl` files.

- Data generated after filtering/screening. For validation, see `scripts/validate_outputs.py` and tests.

- We welcome feedback on data format, quality, and additional fields.
