# GITI annotation pack

Source video: `GITI_traffic_video.mp4` (first 300 frames).

## Contents

- `frames/frame_XXXXX.png` - one PNG per sampled frame (every 10th frame).
- `gt_detection_template.csv` - predicted boxes for those frames. **EDIT the `class_name` column; replace wrong boxes; delete false detections; add missed objects.**
- `gt_tracking_template.csv` - same boxes with a `track_id` column. **Assign consistent IDs across frames.**
- `gt_trajectory_template.csv` - world coordinates. Fill after tracking GT exists (can be projected from your corrected detection GT).
- `gt_ssm_template.csv` - every predicted PET event. Watch the video and mark each: `verdict = real` or `false`. If `real`, record the PET you measure (seconds, t_b_entry - t_a_exit).

## Workflow

1. Open the video at the same frame numbers used in the CSVs.
2. For each sampled frame, correct the boxes and fill `class_name`.
3. Assign `track_id` in `gt_tracking_template.csv`: same ID for the same physical object across frames.
4. Review each PET event in `gt_ssm_template.csv` against the video.
5. Save your edited files as `gt_detection.csv`, `gt_tracking.csv`, `gt_trajectory.csv`, `gt_ssm.csv`.

## DO NOT

- Do not leave the prediction values in place and call it GT. That biases every metric toward the model.
- Do not mark a PET event `real` without watching it in the video. Predicted events can be tracking artifacts.

## Next step

    python scripts/validate_gt.py --detection gt_detection.csv --tracking gt_tracking.csv --trajectory gt_trajectory.csv --ssm gt_ssm.csv

Then run the four evaluate_* scripts per `docs/ANNOTATION_GUIDE.md`.
