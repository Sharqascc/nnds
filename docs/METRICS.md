# Metric Status

Standing rule: **every commit must include a numerical verification for
each core component it touches.** No PR merges without the report below
being run and the deltas pasted into the PR description.

## Core components and current verification status

| component | metric | current value | GT source | status |
|---|---|---|---|---|
| Detection | precision | 0.871 | 5 frames, human (2026-09-14) | ✅ verified |
| Detection | recall | 1.000 | same | ✅ verified |
| Detection | mAP@50 | 0.950 | same | ✅ verified |
| Detection | mAP@50:95 | 0.944 | same | ✅ verified |
| Detection | AP@75 | 0.941 | same | ✅ verified |
| Tracking | HOTA / MOTA / IDF1 | — | not annotated | ❌ blocked |
| Tracking | ID switches | — | not annotated | ❌ blocked |
| Tracking | fragmentation rate | 32/189 (17%) | auto from segments.csv | ⚠️ derived |
| BEV | position MAE | — | no world GT | ❌ blocked |
| BEV | homography reproj error | — | calibration points available | ⚠️ runnable via validate_bev.py |
| Trajectory | velocity MAE | — | no world GT | ❌ blocked |
| Trajectory | acceleration MAE | — | no world GT | ❌ blocked |
| PET / SSM | precision | 0.857 | 53-event review (reconstructed) | ✅ verified |
| PET / SSM | recall | 0.857 | same | ✅ verified |
| PET / SSM | F1 | 0.857 | same | ✅ verified |
| PET / SSM | PET MAE vs GT | 0.0 (circular) | GT PET = pipeline PET | ⚠️ meaningless |
| PET / SSM | critical-conflict recall | 0.857 | 7 real events with PET<1.0s | ✅ verified |

## Reproduction commands

    # detection
    python scripts/evaluate_detection_metrics.py \
        --detections data/annotations/giti_300/first_real_metric/pred_detection_5f_only.csv \
        --ground-truth data/annotations/giti_300/first_real_metric/gt_detection_5f_reviewed.csv \
        --out-json outputs/detection_metrics.json

    # PET / SSM
    python scripts/evaluate_ssm_metrics.py \
        --predicted outputs/pet_pred_14.csv \
        --ground-truth outputs/pet_gt_20.csv \
        --out-json outputs/ssm_metrics.json

## What unblocks the blocked rows

| blocked metric | needs | effort |
|---|---|---|
| Tracking HOTA/MOTA/IDF1 | 5 frames labeled with consistent track IDs | ~30 min |
| BEV position MAE | 10 pixel clicks on ground-plane features + their world coords | ~20 min |
| Trajectory MAE | same world-coord annotations | ~20 min |
| PET MAE vs GT | ground-truth PET values for 20 events (currently only binary Y/N) | requires GT device or manual re-annotation |

## Rule for future changes

Whenever a core component is modified:
1. Run the relevant reproduction command above
2. Paste the JSON output diff in the PR description
3. Update the table in this file with the new value

If a component lacks GT, state that explicitly in the PR. Do not
claim an improvement that cannot be measured.


## Caveats

- Detection numbers (0.871 precision, 0.944 mAP50:95) come from **5 frames**.
  Small sample; treat as a pilot, not a validated metric.
- PET/SSM precision, recall, and critical recall come from the
  **53-event human review**. Solid, but single-site, single-clip.
- **PET numeric accuracy is unmeasured.** The PET MAE we can compute
  uses the pipeline's own PET values as GT. It is circular and
  reports 0.0 for that reason. To get real PET MAE we need manually
  measured PET for a subset of events: click the frame where A exits
  the zone, click the frame where B enters it, compute the gap.
- Tracking HOTA / MOTA / IDF1: no GT. Fragmentation rate (32 of 189
  tracks split) is the only tracking number we can compute today,
  and it is derived from the pipeline's own output, not external
  truth.
- BEV position and trajectory velocity/acceleration: no world GT.
  Unmeasurable today.

## Reproduction commands (verified)

    # detection (5-frame pilot)
    python scripts/evaluate_detection_metrics.py \
        --detections data/annotations/giti_300/first_real_metric/pred_detection_5f_only.csv \
        --ground-truth data/annotations/giti_300/first_real_metric/gt_detection_5f_reviewed.csv \
        --out-json outputs/detection_metrics.json

    # SSM / PET (53-event review, single-site)
    python scripts/evaluate_ssm_metrics.py \
        --predicted outputs/pet_pred_14.csv \
        --ground-truth outputs/pet_gt_y_only.csv \
        --critical-threshold 1.0 \
        --out-json outputs/ssm_metrics.json
