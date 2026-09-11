# NNDS Ground-Truth Annotation Guide

Four independent annotation files are needed.

## 1. Detection ground truth

File: gt_detection.csv

| column | type | notes |
|---|---|---|
| frame | int | consistent across all files |
| x1, y1, x2, y2 | float | pixel coords, x2 > x1, y2 > y1 |
| class_name | str | car, truck, bus, motorcycle, bicycle, pedestrian |

Rules:
- Annotate every frame in the sequence.
- Boxes must enclose the whole visible object.
- Do not annotate objects smaller than about 10x10 px.
- One row per object per frame.

## 2. Tracking ground truth

File: gt_tracking.csv

| column | type | notes |
|---|---|---|
| frame | int | same indexing as detection GT |
| track_id | int | unique per real-world object across the whole sequence |
| x, y | float | pixel coords of the box top-left |
| w, h | float | box width, height in pixels |

Rules:
- Each object gets exactly one track_id for its entire visible lifetime.
- If an object leaves and re-enters, give it the same ID.
- Occlusion < 5 frames: keep the track alive (interpolate).
- Occlusion >= 5 frames: treat as exit + re-entry with the same ID.

## 3. Trajectory / world-coordinate ground truth

File: gt_trajectory.csv

| column | type | notes |
|---|---|---|
| frame | int | same as above |
| track_id | int | must match gt_tracking.csv |
| x, y | float | world coords in meters (easting/northing) |

Rules:
- Convert from pixel to world using the reference homography.
- Position tolerance should be 0.1 m or better for gold GT.

## 4. SSM ground truth

File: gt_ssm.csv

| column | type | notes |
|---|---|---|
| track_a | int | lower of the two IDs |
| track_b | int | higher of the two IDs |
| pet | float | seconds; positive; finite |
| ttc | float | optional; seconds; positive; finite |

Rules:
- Record only real conflicts (pair within the conflict-zone threshold).
- PET = t_b_entry - t_a_exit. Overlap => do NOT record a PET.
- Two independent annotators; disagreements go to a third.

## Quality checklist

- Every CSV has the exact required columns.
- frame indexing is consistent across all four files.
- All boxes satisfy x2 > x1 and y2 > y1.
- All pet and ttc values are positive and finite.
- Every track_a/track_b appears in gt_tracking.csv.
- Run scripts/validate_gt.py and resolve every error.

## Running the evaluation

    python scripts/evaluate_detection_metrics.py  --detections pred_det.csv  --ground-truth gt_detection.csv
    python scripts/evaluate_tracking_metrics.py   --tracked    pred_trk.csv  --ground-truth gt_tracking.csv
    python scripts/evaluate_trajectory_metrics.py --predicted  pred_traj.csv --ground-truth gt_trajectory.csv --fps 30
    python scripts/evaluate_ssm_metrics.py        --predicted  pred_ssm.csv  --ground-truth gt_ssm.csv
    python scripts/pet_agreement_report.py        --predicted  pred_ssm.csv  --ground-truth gt_ssm.csv
