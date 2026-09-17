# Concrete Problems Found in NNDS Pipeline

## CRITICAL — Bugs producing wrong results or crashes

### 1. `src/pipeline/traffic_analyzer.py:84` — `validate_bev` returns duplicate metrics
```python
return {"mean_error_all": mean_all, "mean_error": mean_all, "rmse": rmse}
```
Both `"mean_error_all"` and `"mean_error"` contain the same value. The `"mean_error"` key should report mean error on inlier points only (using `self.inlier_mask`), but currently duplicates `"mean_error_all"`.

### 2. `src/bev/bev_mapper.py:71` — `pixel_to_world` lacks division-by-zero guard
```python
world_h = self.H @ pixel_h
return (world_h[:2] / world_h[2]).ravel()
```
If `world_h[2] == 0` (point maps to infinity), this raises `ZeroDivisionError`. No check for `abs(world_h[2]) < 1e-9`.

### 3. `src/pipeline/traffic_analyzer.py:106` — `estimate_speed` accepts `fps` but never uses it
```python
def estimate_speed(self, pixel_positions, frame_times, fps: float = 30.0):
```
The `fps` parameter is documented as "frames per second" but the function computes speed using `frame_times` directly (which are already in seconds). The `fps` argument is completely unused, misleading callers.

### 4. `src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py:645` — Track splitting uses wrong coordinate for ground contact
```python
tracks.setdefault(track_id, []).append(
    TrackPoint(frame=det.frame, x=det.cx, y=det.cy, ...)
)
```
The comment at line 642 says "Trajectory points are stored at the box bottom-center so BEV projection, grid-cell assignment, and PET all refer to the ground-contact point rather than the visual center of the detection box." But the code uses `det.cy` (box center Y) instead of `det.y2` (box bottom Y). This causes systematic vertical offset in all downstream PET/BEV calculations.

### 5. `src/analysis/pet_conflict_checker.py:28` — `_get_velocity_vector` returns zero for same-frame points
```python
if dt <= 0:
    return (0.0, 0.0)
```
When two detections have the same frame number (possible with duplicate detections), `dt = 0` triggers zero velocity instead of skipping or handling gracefully. This silently corrupts velocity-based uncertainty estimates.

### 6. `src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py:792` — `max_frame_gap=5` causes excessive track fragmentation
```python
tracks = _split_tracks_by_gaps(tracks, max_frame_gap=5, ...)
```
With 30 FPS video, `max_frame_gap=5` = 0.167 seconds. Normal occlusion at intersections often exceeds this (vehicles stopping at red lights, turning). This splits single physical tracks into multiple track IDs, inflating track count and creating false PET events between fragments of the same vehicle.

### 7. `src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py:588` — `_compute_histogram` uses full frame instead of crop
```python
hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
```
The variable `crop` is correctly extracted, but `cv2.cvtColor` is called on `frame` (the full frame) instead of `crop`. This makes appearance features useless for ReID matching.

### 8. `src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py:385` — UVH class mapping misses "auto" class
```python
UVH_DISPLAY_MAP = { ... "Three-wheeler": "auto", ... }
```
But `CLASS_NAME_TO_ID` has `"auto": 8`. The display map uses "Three-wheeler" as key, but the detector may output "auto" directly. This causes `mapped_name = UVH_DISPLAY_MAP.get(raw_name)` to return `None` for "auto" class detections, silently dropping them.

### 9. `src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py:430` — COCO person suppression uses wrong overlap metric
```python
covered = any(_overlap_over_person(person_box, veh_box) >= person_suppress_overlap for veh_box in uvh_boxes_for_suppression)
```
`_overlap_over_person` computes `intersection_area / person_box_area`. But the comment says "suppress COCO person if overlap/person-area exceeds threshold". The code checks if person box is mostly covered by vehicle box, but the intent is likely the reverse: suppress person detection if it overlaps significantly with a vehicle detection (i.e., vehicle box covers person box).

### 10. `src/pipeline/custom_tracker.py:189` — Stage 2 matching uses wrong track center for motion cost
```python
pred_center = self.tracks[tid].center
```
`KalmanTrack.center` returns `statePost[:2]` (predicted center after `predict()`). But in Stage 2, tracks have already been `update()`d with matched detections, so `statePost` reflects the *updated* position, not the predicted position. This makes motion cost compare detection to updated position instead of predicted position, defeating the purpose of motion consistency.

---

## IMPORTANT — Design flaws, missing validation, misleading docstrings

### 11. `src/analysis/grid_trajectory/spatial_grid.py:156` — `get_cell_from_pixels` returns `OUT_OF_BOUNDS` for valid boundary points
```python
if not (self.x_min <= px_x <= self.x_max and self.y_min <= px_y <= self.y_max):
    return OUT_OF_BOUNDS_CELL
```
Uses `<=` for both bounds. A point at exactly `x_max` or `y_max` (e.g., 1600, 720 for GITI) is valid pixel coordinate but returns `OUT_OF_BOUNDS`. Should use `<` for max bounds.

### 12. `src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py:329` — FPS fallback hardcoded to 30.0
```python
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 30.0
```
Many traffic cameras run at 25 FPS (PAL) or 29.97 FPS (NTSC). Hardcoding 30.0 introduces systematic timing errors in PET calculations for non-30FPS sources.

### 13. `src/analysis/pet_conflict_checker.py:156` — `estimate_pet_uncertainty` uses hardcoded error sources
```python
error_sources = {
    "detection": 0.2,
    "homography": 0.3,
    "tracking": 0.1,
}
```
These values are not configurable and not derived from actual calibration data. They're arbitrary constants presented as principled uncertainty quantification.

### 14. `src/analysis/conflict_classifier.py:95` — `classify_conflict_geometry` uses arbitrary speed difference threshold
```python
if speed_diff > 0.5 * max(speed_a, speed_b):
    return "rear_end"
```
The `0.5` multiplier is a magic number with no citation or validation. Small speed differences could be measurement noise.

### 15. `src/bev/bev_mapper.py:158` — `test_with_real_calibration` swallows exceptions silently
```python
except Exception:
    return None
```
Multiple `try/except` blocks catch all exceptions and return `None` without logging. Failures in calibration validation become silent `None` returns, making debugging impossible.

### 16. `src/analysis/gate_counter.py:200` — `VirtualGate.check_crossing` treats near-zero as no crossing
```python
eps = 1e-6
if abs(prev_side) < eps or abs(curr_side) < eps:
    return None
```
A track exactly on the gate line (side = 0) is treated as "no crossing" rather than a valid crossing event. This loses events where a vehicle travels exactly along the gate boundary.

### 17. `src/analysis/grid_trajectory/pet_grid.py:180` — `compute_pet` sorts intervals by `t_enter` but compares `j > i` only
```python
sorted_intervals = sorted(cell_intervals, key=lambda iv: iv.t_enter)
for i in range(n):
    for j in range(i + 1, n):
```
This only checks pairs where B enters after A. But if B enters before A but exits after A enters (overlap), the reverse case (A enters during B's occupancy) is never checked because `j > i` prevents it. The overlap case is handled, but the sequential B→A case where B exits before A enters is only caught when B is first in sorted order.

### 18. `src/pipeline/traffic_analyzer.py:200` — `run_video_to_pet` for "uvh-coco-fused" ignores `max_frames` in some paths
```python
result = uvh_mod.run_uvh_coco_fused_grid_pet(
    ...
    max_frames=max_frames,
    ...
)
```
But `run_uvh_coco_fused_grid_pet` has its own `max_frames` parameter defaulting to `None`. The caller passes it correctly, but the function signature shows `max_frames: int | None = None` while the internal loop uses `if max_frames is not None and frame_idx >= max_frames: break`. This works, but the parameter is documented as "max frames to process" while the actual limit is on `frame_idx` (0-indexed), so `max_frames=100` processes 101 frames (0-100).

### 19. `src/analysis/grid_trajectory/yolo_cpu_grid_pet.py:100` — `run_yolo_cpu_grid_pet` uses `frame_idx` from outer scope
```python
for r in results:
    if max_frames is not None and frame_idx >= max_frames:
        break
    ...
    frame_idx += 1
```
`frame_idx` is initialized to 0 before the loop, but if `results` is empty, `frame_idx` remains 0 and the function returns empty results without error. No validation that video was actually processed.

### 20. `src/analysis/ssm/ssm_verification.py:85` — `verify_pet_calculation` uses hardcoded severity thresholds
```python
critical = np.sum(data < 0.5)
serious = np.sum((data >= 0.5) & (data < 1.0))
moderate = np.sum((data >= 1.0) & (data < 1.5))
```
These thresholds (0.5, 1.0, 1.5) are FHWA guidelines but hardcoded. The class accepts `tolerance` parameter but doesn't use it for severity classification.

---

## DOC CONTRADICTIONS — Claims in docs not supported by code

### 21. `README.md:45` — Claims "stable 32-column schema" for PET CSV
```markdown
The PET CSV uses a stable 32-column schema, including when no PET events are detected.
```
But `src/pipeline/traffic_analyzer.py:200` `_write_events_to_csv` builds columns dynamically from event dict keys. The column set varies based on which optional fields are present in events (e.g., `gate_a_entry`, `time_of_day_label` only appear when gates/time-of-day are configured). Not stable.

### 22. `docs/METRICS.md` — Claims "PET MAE vs GT: 0.0 (circular)"
```markdown
| PET / SSM | PET MAE vs GT | 0.0 (circular) | GT PET = pipeline PET | ⚠️ meaningless |
```
But `scripts/evaluate_ssm_metrics.py` computes MAE between predicted and ground truth PET values. The "circular" claim only applies if GT PET values come from the same pipeline run. The doc presents this as a general truth.

### 23. `docs/ANNOTATION_GUIDE.md:35` — Claims "PET = t_b_entry - t_a_exit. Overlap => do NOT record a PET"
```markdown
PET = t_b_entry - t_a_exit. Overlap => do NOT record a PET.
```
But `src/analysis/grid_trajectory/pet_grid.py:180` `compute_pet` returns events for overlapping intervals with `pet=0.0` (when `a_exit == b_entry`). The code treats zero-gap as sequential, not overlap.

### 24. `docs/VALIDATION.md:12` — Claims "run `scripts/reproduce_pipeline.sh`"
```markdown
bash scripts/reproduce_pipeline.sh
```
But `scripts/reproduce_pipeline.sh` does not exist in the repository. The actual reproduction script is `scripts/run_pipeline.py` or `Makefile` target `reproduce-final`.

### 25. `src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py:1` — Module docstring says "UVH-COCO fused grid PET" but uses SAM3/YOLO fallbacks
```python
"""UVH-COCO fused grid PET pipeline."""
```
The module imports and uses `sam3_grid_pet` and `yolo_cpu_grid_pet` as fallback detectors, but the docstring doesn't mention this. The "UVH-COCO fused" name implies a specific detector combination, not a multi-detector pipeline.

---

## STALE NUMBERS — Metrics referencing files/scripts that don't exist

### 26. `docs/METRIC_STATUS.md:15` — References `outputs/pet_gt_y_only.csv` and `outputs/pet_pred_14.csv`
```markdown
python scripts/evaluate_ssm_metrics.py \
    --predicted outputs/pet_pred_14.csv \
    --ground-truth outputs/pet_gt_y_only.csv \
    --critical-threshold 1.0 \
    --out-json outputs/ssm_metrics.json
```
Neither `outputs/pet_pred_14.csv` nor `outputs/pet_gt_y_only.csv` exist in the repository. The `outputs/` directory is gitignored. These files were from a local development run, not committed.

### 27. `docs/STATUS.md:45` — Event count table references version numbers 156, 168, 164, 153
```markdown
| Version | Count | Reason |
|---------|-------|--------|
| 156 | Pre-fix | BEV mapper bug... |
| 168 | First fix | BEV mapper fixed... |
| 164 | Current raw | After same-origin exclusion... |
| **153** | **Current screened** | Raw 164 minus 11 events... |
```
Version numbers decrease (168 → 164 → 153) but are presented as chronological. These appear to be git commit counts or CI run numbers, not semantic versions. The table is misleading.

### 28. `docs/METRIC_STATUS.md:8` — Claims "0.857 precision/recall verified" for PET/SSM
```markdown
| PET / SSM | precision | 0.857 | 53-event review (reconstructed) | ✅ verified |
| PET / SSM | recall | 0.857 | 53-event review (reconstructed) | ✅ verified |
```
The "53-event review (reconstructed)" refers to a manual review that was never committed. The ground truth files (`outputs/pet_gt_20.csv`, `outputs/pet_gt_y_only.csv`) don't exist in the repo. The verification cannot be reproduced.

### 29. `docs/GITI_EVAL_STATUS.md:200` — References `data/annotations/giti_300/first_real_metric/detection_5f_real.json`
```json
{
  "precision": 0.8709677419354839,
  "recall": 1.0,
  "f1": 0.9310344827586207,
  ...
}
```
This file exists but claims "first real metric" from 5-frame pilot. The parent directory `data/annotations/giti_300/first_real_metric/` contains only template CSVs, not human-annotated ground truth. The metrics are from pipeline self-comparison, not real ground truth.

### 30. `configs/bev_config.json:15` — Homography matrix has near-zero off-diagonals
```json
"H_pixel_to_world": [
  [0.01586042861296038, 1.1541369407046988e-28, 730897.8930766084],
  [3.645114990734786e-29, 0.027072759088983777, 221994.9929775622],
  [1.6418967253570837e-34, 1.579039018534226e-34, 1.0]
]
```
The off-diagonal elements are ~1e-28 to 1e-34 (effectively zero). This suggests the homography was computed from a synthetic rectangle calibration, not real camera geometry. The matrix is essentially a scale+translation, not a true perspective transform.