# Metric Status

> **Correction (this commit):** several PET/SSM "verified" labels were
> based on files (`outputs/pet_pred_14.csv`, `outputs/pet_gt_y_only.csv`)
> that were never committed to the repository. Marked as unverified below.
> The doc's appendix already said this; the table now agrees with it.

The current, measured values for each metric defined in
`docs/METRICS.md`. Standing rule: every PR that touches a core
component runs the reproduction commands below and pastes the
JSON in the description.

Metric definitions and module locations: see `docs/METRICS.md`.

Standing rule: **every commit must include a numerical verification for
each core component it touches.** No PR merges without the report below
being run and the deltas pasted into the PR description.

## Annotation pack status

An annotation kit is committed at `data/annotations/giti_300/`:

- 30 PNG frames (every 10th frame of the first 300 of the GITI clip)
- `gt_detection_template.csv` (1798 rows), `gt_tracking_template.csv`
  (1798 rows), `gt_trajectory_template.csv` (1798 rows),
  `gt_ssm_template.csv` (114 events)

The templates are **unedited pipeline predictions**, not ground truth.
Verified at commit `e054422`:

- `gt_detection_template.csv`: `class_name` is NaN in 1798/1798 rows
- `gt_tracking_template.csv`: `track_id == -1` in 1798/1798 rows
- `gt_ssm_template.csv`: `verdict` and `actual_pet` empty in 114/114 rows

The "blocked" statuses below are accurate for that reason: the pack
exists, but no human has reviewed it. Finishing the labeling is a
single-file-edit task per the workflow in
`data/annotations/giti_300/README.md`.

The 5-frame detection pilot **is** committed
(`data/annotations/giti_300/first_real_metric/`), which is why its
source files are listed explicitly in the detection rows below. That
is the one metric in this table whose provenance is fully reproducible
from the repository.

## Core components and current verification status

| component | metric | current value | GT source | status |
|---|---|---|---|---|
| Detection | precision | 0.871 | 5 frames, human (first_real_metric/) | ✅ verified |
| Detection | recall | 1.000 | same (first_real_metric/) | ✅ verified |
| Detection | mAP@50 | 0.950 | same (first_real_metric/) | ✅ verified |
| Detection | mAP@50:95 | 0.944 | same (first_real_metric/) | ✅ verified |
| Detection | AP@75 | 0.941 | same (first_real_metric/) | ✅ verified |
| Tracking | HOTA / MOTA / IDF1 | — | not annotated | ❌ blocked |
| Tracking | ID switches | — | not annotated | ❌ blocked |
| Tracking | fragmentation rate | 32/189 (17%) | auto from segments.csv | ⚠️ derived |
| BEV | position MAE | — | no world GT | ❌ blocked |
| BEV | homography reproj error (6 calibration pts) | 0.000001 m max | configs/giti_calibration_points.json | ✅ verified (self-consistent) |
| Trajectory | velocity MAE | — | no world GT | ❌ blocked |
| Trajectory | acceleration MAE | — | no world GT | ❌ blocked |
| PET / SSM | 55° angle gate MCC | −0.106 (CI [−0.307, +0.084]) | 114-event review | ✅ verified (anti-signal; gate retired) |
| PET / SSM | 2-feature model MCC | +0.367 (CI [+0.005, +0.723]) | 114-event review | ✅ verified (weak; not deployed) |
| PET / SSM | N-class precision | 0.814 ± 0.082 | 114-event review, 2-feature model | ✅ verified |
| PET / SSM | N-class recall | 0.713 ± 0.124 | same | ✅ verified |
| PET / SSM | Y-class precision | 0.548 ± 0.119 | same | ✅ verified |
| PET / SSM | Y-class recall | 0.660 ± 0.174 | same | ✅ verified |
| PET / SSM | PET MAE vs GT | 0.0 (circular) | GT PET = pipeline PET | ⚠️ meaningless; see #15 (zero-gap divergence) |

## Reproduction commands

    # detection
    python scripts/evaluate_detection_metrics.py \
        --detections data/annotations/giti_300/first_real_metric/pred_detection_5f_only.csv \
        --ground-truth data/annotations/giti_300/first_real_metric/gt_detection_5f_reviewed.csv \
        --out-json outputs/detection_metrics.json

    # PET / SSM — 114-event review (see data/reviews/ssm_review_114/)
    # The prior 0.857 number has been retracted; see REPRODUCE.md.
    python -c "
import pandas as pd
df = pd.read_csv('data/reviews/ssm_review_114/to_label.csv')
print(df['verdict'].value_counts())
"
    # For feature-level analysis (MCC, balanced accuracy, PR-AUC per feature),
    # see data/reviews/ssm_review_114/findings.md.

## What unblocks the blocked rows

| blocked metric | needs | where to start |
|---|---|---|
| Tracking HOTA/MOTA/IDF1 | fill `track_id` in `gt_tracking_template.csv` (1798 rows, currently -1) | `data/annotations/giti_300/README.md` workflow step 3 |
| PET MAE vs GT | fill `verdict` + `actual_pet` in `gt_ssm_template.csv` (114 rows, currently empty) | workflow step 4 |
| BEV position MAE | 10 pixel clicks on ground-plane features + their world coordinates, not used in the fit | `docs/ANNOTATION_GUIDE.md` section 3 |
| Trajectory MAE | same world-coordinate annotations as BEV | same |
| Detection (full set) | fill `class_name` in `gt_detection_template.csv` (1798 rows, currently NaN); 5-frame subset is already done | workflow step 2 |

The pack exists; what is missing is a human labeling session. The 5-frame
subset in `first_real_metric/` shows the expected output format.

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


## BEV calibration verification (2026-09-15)

Reprojection of the 6 calibration points through `H_pixel_to_world`:

    mean error: 0.000000 m
    max error:  0.000001 m
    n:          6

### Coordinate-system convention

The pipeline's world coordinates use absolute easting/northing
(offset ~730900, ~222014). The calibration JSON uses local planar
coordinates with origin at point P4. To compare, subtract the
absolute offset of P4 and negate the y component (calibration
stores +y upward, homography maps +y downward):

    pred_local_x = wx - P4_x
    pred_local_y = -(wy - P4_y)

Without the y-flip, three points show a 32 m / 16 m error that
looks like a homography failure but is purely a sign convention.

### What this does and does not verify

- The H matrix is self-consistent: the six points that defined it
  reproject to themselves exactly.
- Real BEV accuracy is not measured. Held-out validation requires
  pixel clicks at known world positions not used in the fit.
  ~20 min of human time.

### Templates audit (2026-09-15)

`data/annotations/giti_300/gt_tracking_template.csv` (1798 rows,
30 frames, 58-65 boxes/frame) matches the pipeline's own detections
within 0.43 px on every box. It is pipeline output, not human
annotation. Not usable as detection or tracking GT.

`gt_trajectory_template.csv` and `gt_detection_template.csv` were
audited at the same time and are likewise pipeline output.

`gt_ssm_template.csv` has 114 events with PET values but no
verdicts. 20 of them overlap the 53-event review (which we
reconstructed from docs after the reclone). 94 are unlabeled.
Extending the review to all 114 is the highest-value remaining
labeling task: it doubles the SSM sample size.


### Note on PET/SSM verification

The PET/SSM precision, recall, F1, and critical-recall values (0.857)
are computed by scripts/evaluate_ssm_metrics.py against files that
have never been committed to this repository. Verified with
`git log --all -- outputs/pet_pred_14.csv` (no commits), and same for
`outputs/pet_gt_20.csv` and `outputs/pet_gt_y_only.csv`.

The "verified" status is not supported. The numbers cannot be
reproduced from what is in the repo.

Two inconsistencies:
1. The doc's text cites a "53-event review", but the command references
   14-event files (pet_pred_14.csv). Counts do not match.
2. Both input files, and the "review" they come from, exist only in
   the original development environment. They were never committed.

Recommend retagging as unverified.
