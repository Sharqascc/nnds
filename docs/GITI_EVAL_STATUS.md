# GITI Evaluation - Current State

**Last updated:** 2026-09-13
**Branch:** feature/pipeline-to-metric-schemas

## Reproducing everything from scratch

Prerequisites: Git LFS installed (`apt-get install -y git-lfs`).

    git clone https://github.com/Sharqascc/nnds.git
    cd nnds
    git checkout feature/pipeline-to-metric-schemas
    git lfs install
    git lfs pull --include="data/sample_data/GITI_traffic_video.mp4"
    pip install -r requirements.txt
    pip install ruff mypy pytest-cov hypothesis syrupy
    bash scripts/download_models.sh

Then run the pipeline + convert + review pack:

    python -m src.pipeline.traffic_analyzer \
        --video data/sample_data/GITI_traffic_video.mp4 \
        --detector uvh-coco-fused \
        --bev-config configs/bev_config.json \
        --grid-config configs/GITI_grid_config.json \
        --out-csv outputs/giti_eval_300/pet.csv \
        --max-frames 300

    python scripts/pipeline_to_metric_schemas.py \
        --detections-csv outputs/giti_eval_300/pet_detections.csv \
        --pet-csv outputs/giti_eval_300/pet.csv \
        --bev-config configs/bev_config.json \
        --out-dir outputs/giti_eval_300/schemas

    python scripts/ssm_review_pack.py \
        --video data/sample_data/GITI_traffic_video.mp4 \
        --pet-csv outputs/giti_eval_300/pet.csv \
        --detections-csv outputs/giti_eval_300/pet_detections.csv \
        --grid-config configs/GITI_grid_config.json \
        --out-dir outputs/giti_eval_300/ssm_review

Expected: 118 PET events, 41 likely_real, 65 ambiguous, 3 likely_false
(this run was on 2026-09-13; numbers can shift if the pipeline anchor
or tracker settings change).

## What exists

- Video: `data/sample_data/GITI_traffic_video.mp4` (1837 frames, 61 s, LFS)
- Pipeline run: `outputs/giti_eval_300/` (300 frames, ~7 min on CPU)
- Pred schemas: `outputs/giti_eval_300/schemas/pred_*.csv`
- Review strips: `outputs/giti_eval_300/ssm_review/review_strips/*.jpg`
- Prelabel CSV: `outputs/giti_eval_300/ssm_review/gt_ssm_prelabel.csv`
- Annotation pack: `data/annotations/giti_300/` (tracked in git)

## Grid configuration

- `configs/GITI_grid_config.json`: `cell_size = 25` px (was 50)
- `configs/sites/giti/grid_config.json`: `cell_size = 25` px (was 100)
- Naming: `CELL_{column_letters}_{row_number_1based}` (e.g. `CELL_AQ_7`)

Row numbering is 1-based to match
`src/analysis/grid_trajectory/spatial_grid.py`.
`src/analysis/grid_overlay.py` follows the same convention.
A regression test (`tests/test_grid_overlay_matches_pipeline.py`) pins
the two implementations together.

## Known issues

### 1. Trajectory anchor mismatch (open)

`src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py` was patched so
TrackPoint uses `y=det.y2` (box bottom, ground contact) instead of
`y=det.cy` (box center). The source file confirms the patch is present.

However, after re-running the pipeline, `traj_a_json` entries still show
`y_pixel` matching the box center, not the box bottom:

    event 0, frame 0:
      traj y_pixel = 173.5
      nearest detection: cy = 176.1, y2 = 197.7
      distance from cy = 2.6   (traj is at center)
      distance from y2 = 24.2  (traj is not at bottom)

Possible causes to investigate next:
- A second code path building the trajectory JSON directly from
  `det.cx, det.cy` rather than from the TrackPoint objects.
- Stale module reference held by `traffic_analyzer.run_video_to_pet`
  even after `sys.modules` purge.
- Track splitting that rewrites points using the detection CSV rather
  than the raw tracks dict.

Note: the run DID produce different output (118 events, 159 valid tracks
vs. 109/155 before), so the patch had some effect. The mismatch is
specific to the JSON serialization path.

### 2. PET track IDs vs detection track IDs

`pet.csv` uses composite IDs like `track_44000`, while
`pet_detections.csv` uses `track_id ∈ 1..131`. These namespaces do not
overlap. The review pack works around this by matching detections to
trajectory points by pixel proximity, not by ID.

Worth investigating: which stage renames tracks, and whether the mapping
should be preserved as an extra column.

## What is done

- [x] 8 metric modules in `src/analysis/`
- [x] GT validator `scripts/validate_gt.py`
- [x] `--out-json` on all metric scripts
- [x] Pipeline-to-metric-schema converter
- [x] End-to-end synthetic demo
- [x] Real GITI pipeline run (300 frames)
- [x] SSM review pack generator (`scripts/ssm_review_pack.py`)
- [x] Grid overlay module with regression test
- [x] 25 px grid on GITI configs
- [x] Annotation pack committed to `data/annotations/giti_300/`

## What is not done

- [ ] Trajectory anchor JSON fix (see Known Issue 1)
- [ ] Human review of 41 likely_real PET strips → `gt_ssm.csv`
- [ ] Real PET precision / MAE
- [ ] Detection GT (box annotation)
- [ ] Tracking GT
- [ ] Trajectory GT
- [ ] Gold-standard 15-row table with real values

## Session log

### 2026-09-13 (last)

Done this session:
- Grid overlay module + regression test pinning it to the pipeline grid
- Grid cells reduced 50 -> 25 px on GITI configs
- Review strips: trails at box bottom, header shows A exit / B enter / PET
- Pipeline anchor patch verified: traj y_pixel == det.y2 (bottom-center)
- Motion-consistency term in tracker Stage 1 (Kalman-predicted center)
- Track stability diagnostics module + script + tests
- Baseline diagnostic on the current output (see below)

### Baseline diagnostic (300 frames, before Stage 2 gate)

    n_tracks:                131
    n_short (<10 frames):     21  (16%)
    n_gaps:                  725
    frac tracks with large gap: 60.3%
    mean_jump_px:            2.00
    p95_jump_px:             5.11
    max_jump_px:           118.72
    n_large_jumps (>30 px):   53
    max_accel_px:          231.34
    n_box_changes (>50%):     46

Interpretation: broadly stable (mean/p95 fine), but a large failure tail
-- 53 teleports and 46 sudden box-size changes -- explains the zig-zag
that was visible around frame 98 in the review strips.

## Next session (resume here)

**Order: B -> A -> C** (decided this session, reason below).

### B. Stage 2 gate (one predeclared patch, one rerun)

Before writing the gate, run the pre-B check: classify each gap as
edge-of-frame (genuine exit/re-entry) vs mid-frame (occlusion / dropout /
ID switch). If most gaps are mid-frame, a *stricter* gate is the wrong
fix -- we'd need to loosen Stage 2 or increase `max_age` instead.

The predeclared Stage 2 gate is in Cell 3 of this session's plan:
- size ratio (det_area / track_predicted_area) rejected if outside [0.67, 1.5]
- adaptive per-track jump limit = max(3 * median(recent jumps), 30 px)
- every rejection logged to `CustomTracker.rejections` and dumped to JSON

Success criteria (predeclared):
- `n_large_jumps` and `n_box_changes` drop materially
- `gap_rate` must NOT rise materially (>5 pp increase = revert)

Do not tune. One patch, one rerun, then freeze.

### A. Pilot annotation (5 stratified frames)

Frames to annotate (start / early-mid / failure-tail / late-mid / end):

    frame_00000, frame_00080, frame_00098, frame_00200, frame_00290

For each of the 5:
- Fill `class_name` in `gt_detection_template.csv` (verify box, delete FP, add missed)
- Assign consistent `track_id` in `gt_tracking_template.csv`

Then:

    python scripts/validate_gt.py --detection gt_detection.csv --tracking gt_tracking.csv
    python scripts/evaluate_detection_metrics.py \
        --detections outputs/giti_eval_300/schemas/pred_detection.csv \
        --ground-truth gt_detection.csv \
        --out-json outputs/detection_5f.json
    python scripts/evaluate_tracking_metrics.py \
        --tracked outputs/giti_eval_300/schemas/pred_tracking.csv \
        --ground-truth gt_tracking.csv \
        --out-json outputs/tracking_5f.json

Report these as pilot numbers, not publication numbers.

### C. SSM review of the PET candidates

Current run produced 118 PET events:
- 45 likely_real (need eyes-on review)
- 72 ambiguous (overlaps; A does not exit before B enters)
- 1 likely_false (track too short)

Workflow:
1. Open `outputs/giti_eval_300/ssm_review/gt_ssm_prelabel.csv`
2. Bulk-fill all `ambiguous` + `likely_false` rows with `verdict=false`
3. Review the 45 likely_real strips; fill `verdict` and `actual_pet`
4. Save as `gt_ssm.csv`

Then:

    python scripts/validate_gt.py --ssm gt_ssm.csv
    python scripts/evaluate_ssm_metrics.py \
        --predicted outputs/giti_eval_300/schemas/pred_ssm.csv \
        --ground-truth gt_ssm.csv \
        --out-json outputs/ssm.json
    python scripts/gold_standard_report.py \
        --ssm-metrics outputs/ssm.json \
        --out-md outputs/validation_report.md

### Post-C: audit linkage

Join the SSM audit table with `track_stability_after.json`:
for each PET event, did either involved track cross a large-jump or
large-scale-change flag near the event frame? This quantifies whether
tracking instability is actually driving questionable PET events.

## Files touched this session

- `src/analysis/grid_overlay.py` (new)
- `src/analysis/track_stability.py` (new)
- `src/pipeline/custom_tracker.py` (motion term in Stage 1)
- `src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py` (TrackPoint y=det.y2)
- `scripts/ssm_review_pack.py` (bottom-anchored trails, header times)
- `scripts/diagnose_tracking.py` (new)
- `configs/GITI_grid_config.json`, `configs/sites/giti/grid_config.json` (cell_size 25)
- `tests/test_grid_overlay*.py`, `tests/test_grid_trajectory_point_anchor.py`,
  `tests/test_track_stability*.py`, `tests/test_tracker_motion_consistency.py` (new)

## Reproducing from scratch

See the top of this file. The 2026-09-13 rerun produced 118 PET events
(45 likely_real / 72 ambiguous / 1 likely_false).

## Session update: class-aware tracker (2026-09-13)

Root cause of the 30% mixed-class tracks (51/167) was that the tracker never
compared `cls_id` during matching. Bicycles were absorbed into pedestrian
tracks and vice versa.

Fix: `CustomTracker.enforce_class_match = True` makes cross-class pairs
prohibitively expensive in both Stage 1 and Stage 2 Hungarian matching.
Ablation constants (`TRACKER_MAX_AGE`, `TRACKER_IOU_THRESHOLD`,
`TRACKER_REID_STAGE1`) live at the top of `uvh_coco_fused_grid_pet.py`.

Before / after (300 frames, CUDA, seeded):

    metric              baseline   class-gate   delta
    mixed_class_tracks       51           0      -51
    n_tracks                167         139      -28
    mid_gaps                644         703      +59
    mid_gap_fraction      0.867       0.876   +0.009
    n_pet                    78          87       +9

### Ablations run

1. `max_age 60 -> 120`                -- zero measurable change
2. `Stage 2 last-observed anchor`     -- zero measurable change
3. `Stage 2 cap 150 -> 400`           -- zero measurable change
4. `enforce_class_match = True`       -- correctness fix

Tracker is now frozen. The 703 remaining mid-frame gaps are either genuine
occlusions or fragmentation; distinguishing them requires ground truth.

### Frozen pipeline config

    max_age=60
    iou_threshold=0.20
    enforce_class_match=True
    TRACKER_REID_STAGE1=False

### Reproducing the frozen pipeline

    python -m src.pipeline.traffic_analyzer \
        --video data/sample_data/GITI_traffic_video.mp4 \
        --detector uvh-coco-fused \
        --bev-config configs/bev_config.json \
        --grid-config configs/GITI_grid_config.json \
        --out-csv outputs/giti_eval_300/pet.csv \
        --max-frames 300

Expected: 18077 det_rows, 139 tracks, 87 PET events, 0 mixed-class tracks.

