# GITI Evaluation - Current State

**Last updated:** 2026-09-13
**Branch:** feature/pipeline-to-metric-schemas
**Where to resume:** "Next session" section at the bottom.

## What exists

### Pipeline run on real GITI data
- Video: `data/sample_data/GITI_traffic_video.mp4` (1837 frames, 61 s, LFS-tracked)
- Run output: `outputs/giti_eval_300/` -- 300 frames processed
- Results: 18,077 detections, 114 PET events
- Pred schemas: `outputs/giti_eval_300/schemas/pred_{detection,tracking,trajectory,ssm}.csv`
- NOTE: `outputs/` is gitignored, so these must be regenerated on a new machine:
      python -m src.pipeline.traffic_analyzer --video data/sample_data/GITI_traffic_video.mp4 \
        --detector uvh-coco-fused --bev-config configs/bev_config.json \
        --grid-config configs/GITI_grid_config.json --out-csv outputs/giti_eval_300/pet.csv \
        --max-frames 300
      python scripts/pipeline_to_metric_schemas.py \
        --detections-csv outputs/giti_eval_300/pet_detections.csv \
        --pet-csv outputs/giti_eval_300/pet.csv \
        --bev-config configs/bev_config.json \
        --out-dir outputs/giti_eval_300/schemas

### SSM review pack (generated)
- Script: `scripts/ssm_review_pack.py`
- Writes one strip image per PET event with both tracks highlighted
- Also writes `gt_ssm_prelabel.csv` with a pre-label suggestion column

Run it:

    python scripts/ssm_review_pack.py \
        --video data/sample_data/GITI_traffic_video.mp4 \
        --pet-csv outputs/giti_eval_300/pet.csv \
        --detections-csv outputs/giti_eval_300/pet_detections.csv \
        --out-dir outputs/giti_eval_300/ssm_review

### Annotation pack (committed)
- `data/annotations/giti_300/` -- 30 JPEG frames + editable GT templates + README
- For detection GT, tracking GT, trajectory GT annotation (box drawing)

## Key finding

Of 114 PET events reported by the pipeline:

| Suggestion         | Count | Meaning                                              |
|--------------------|-------|------------------------------------------------------|
| likely_real        | 42    | Sequential gap; candidate real PET                   |
| ambiguous          | 69    | A did not exit before B entered (overlap, not PET)   |
| likely_false       | 3     | Track too short, tracking artifact                   |

**The 69 "ambiguous" events are NOT PETs.** Both vehicles were in the conflict
zone at the same time -- that is an overlap, not a sequential post-encroachment
time. The pipeline currently reports them as PET events, which means PET
precision is at best ~37% pending human review of the 42 likely_real cases.

This is a reportable finding: the pipeline's PET output needs filtering
against the "A exits before B enters" invariant before it can be treated as
a PET event list.

## Known issue (unfixed)

PET track IDs (e.g. 22000, 44000) do not match detection track IDs (1-131).
They come from separate indexing namespaces. Consequences:

- Cannot join PET events back to detections by ID
- SSM review script works around it by matching on per-frame pixel coordinates
- Worth investigating: which stage renames tracks, and whether the mapping
  should be preserved

## What is done

- [x] 8 metric modules (`src/analysis/{detection,tracking,bev,traj,ssm,ssm_agreement,gold_standard}*.py`)
- [x] GT validator (`scripts/validate_gt.py`)
- [x] `--out-json` on all metric scripts
- [x] Pipeline-to-metric-schema converter (`scripts/pipeline_to_metric_schemas.py`)
- [x] End-to-end synthetic demo (`scripts/demo_end_to_end.py`)
- [x] Real GITI run (300 frames) + schema conversion
- [x] SSM review pack generator (`scripts/ssm_review_pack.py`)
- [x] Annotation pack (30 frames + templates, committed)

## What is NOT done

- [ ] Human review of 42 likely_real PET strips
- [ ] `gt_ssm.csv` produced
- [ ] PET precision / MAE computed against GT
- [ ] Detection GT (box annotation) -- 30 frames, ~2-4 hours
- [ ] Tracking GT
- [ ] Trajectory GT
- [ ] Gold-standard 15-number table with real values

## Next session

**Fastest path to a real number:**

1. Download the review pack:

       python -c "import shutil; shutil.make_archive('/tmp/ssm_review', 'zip', 'outputs/giti_eval_300/ssm_review')"

   Or copy `outputs/giti_eval_300/ssm_review/review_strips/` and
   `gt_ssm_prelabel.csv` out of Colab.

2. Open `gt_ssm_prelabel.csv` in a spreadsheet.

3. Bulk-fill the 69 "ambiguous" rows and 3 "likely_false" rows with
   `verdict=false`.

4. For each of the 42 "likely_real" rows, open the matching strip
   (`review_strips/event_NNN.jpg`) and:
   - If a genuine gap exists: `verdict=real`, `actual_pet=<seconds>`
   - If they overlap: `verdict=false`

5. Save as `gt_ssm.csv`.

6. Back in Colab:

       python scripts/validate_gt.py --ssm gt_ssm.csv
       python scripts/evaluate_ssm_metrics.py \
           --predicted outputs/giti_eval_300/schemas/pred_ssm.csv \
           --ground-truth gt_ssm.csv \
           --out-json outputs/ssm.json
       python scripts/gold_standard_report.py --ssm-metrics outputs/ssm.json \
           --out-md outputs/validation_report.md

7. Read `outputs/validation_report.md`.

That produces real PET precision, PET MAE, and critical-conflict recall for
the NNDS pipeline on the GITI clip.
