# NNDS Evaluation Metrics

Every metric used by the gold-standard evaluation: name, where it is
implemented, which script computes it, and the JSON key that script emits.

## Detection

| Metric | Module | JSON key | Gold-standard row |
|---|---|---|---|
| Precision (micro) | src/analysis/detection_metrics.py::precision_recall_f1 | precision | precision |
| Recall (micro) | same | recall | recall |
| F1 (micro) | same | f1 | f1 |
| mAP@50 | map_at_iou_range | map50 | mAP50 |
| mAP@50:95 | same | map50_95 | mAP50:95 |
| AP@75 | same | ap75 | AP75 |
| Per-class recall | per_class_recall | printed only | - |
| APs / APm / APl | ap_by_size | printed only | - |

## Tracking

| Metric | Module | JSON key | Gold-standard row |
|---|---|---|---|
| HOTA | src/analysis/tracking_metrics.py::hota | hota | HOTA |
| IDF1 | idf1 | idf1 | IDF1 |
| MOTA | mota | mota | MOTA |
| ID switches | count_id_switches | printed only | - |
| Fragmentation | local to evaluate_tracking_metrics.py | printed only | - |

## Trajectory / BEV

| Metric | Module | JSON key | Gold-standard row |
|---|---|---|---|
| Position RMSE (m) | scripts/evaluate_trajectory_metrics.py | position_rmse_m | position_RMSE |
| Position MAE (m) | src/analysis/bev_error.py::position_metrics | mae | - |
| Position p95 (m) | same | p95 | - |
| Velocity MAE (m/s) | src/analysis/traj_error.py::velocity_metrics | velocity_mae_mps | velocity_MAE |
| Acceleration MAE (m/s^2) | traj_error.py::acceleration_metrics | accel_mae_mps2 | accel_MAE |

## SSM (PET / TTC / conflict)

| Metric | Module | JSON key | Gold-standard row |
|---|---|---|---|
| PET MAE (s) | src/analysis/ssm_error.py::pet_value_metrics | pet_mae_s | PET_MAE |
| TTC MAE (s) | ttc_value_metrics | ttc_mae_s | TTC_MAE |
| Conflict P / R / F1 | conflict_prf | printed via evaluate_ssm_metrics.py | - |
| Critical-conflict recall | critical_conflict_recall | critical_conflict_recall | critical_conflict_recall |
| R^2 (SSM agreement) | src/analysis/ssm_agreement.py::agreement_metrics | pet_r2, ttc_r2 | - |
| Spearman rho | same | pet_spearman, ttc_spearman | - |

## Definitions and caveats

Detection.
- IoU on axis-aligned boxes in pixel space.
- Greedy 1-to-1 matching per frame per class, confidence-descending.
- AP is VOC-style (monotone precision envelope).
- mAP@50:95 is the mean AP over IoU 0.50, 0.55, ..., 0.95.
- precision_recall_f1 is micro-averaged: TP/FP/FN summed over classes at IoU
  0.5 with no confidence threshold, then ratios taken.

Tracking.
- All metrics use bipartite matching (Hungarian on -IoU) per frame.
- HOTA averages over IoU 0.05..0.95; score is sqrt(DetA * AssA).
- MOTA counts IDFN + IDFP + IDSW over total GT.
- IDF1 uses global Hungarian on the ID-level overlap matrix.

BEV / trajectory.
- Distances are in the world coordinate unit of the GT file. For NNDS
  calibration that unit is meters (easting/northing).
- Velocity and acceleration use central finite difference (forward/backward
  at endpoints); units are m/s, m/s^2 only if x, y are meters and fps is
  correct.
- Acceleration MAE on noisy GT can be large - double differentiation
  amplifies noise. Report alongside velocity MAE, never in isolation.

SSM.
- PET_MAE and TTC_MAE are computed only over matched unordered pairs
  (track_a, track_b) present in both GT and prediction.
- Duplicate events on the same pair keep the minimum non-None value.
- critical_conflict_recall counts only GT events below --critical-threshold
  (default 1.5 s). If GT has zero critical events, recall is reported as
  0.0, not 1.0 - "nothing to recall" is not evidence of good performance.

## What passing tests mean

The test suite (unit + Hypothesis property tests) verifies that every metric
function is implemented correctly: bounded, monotone where it should be,
correct on controlled inputs. It does not verify that NNDS outputs are
accurate. That requires real ground truth.

## Converting pipeline output to metric schemas

`scripts/pipeline_to_metric_schemas.py` converts `run_pipeline.py` output into
the four PRED CSVs the metric scripts consume:

    python scripts/pipeline_to_metric_schemas.py \
        --detections-csv outputs/run_detections.csv \
        --pet-csv        outputs/run_pet.csv \
        --bev-config     configs/bev_config.json \
        --out-dir        outputs/schemas

Writes:
- `pred_detection.csv`  (frame, x1, y1, x2, y2, class_name, conf)
- `pred_tracking.csv`   (frame, track_id, x, y, w, h)
- `pred_trajectory.csv` (frame, track_id, x, y) -- world coords in meters
- `pred_ssm.csv`        (track_a, track_b, pet)

The trajectory conversion projects each detection's box center through
`H_pixel_to_world` from the BEV config.

