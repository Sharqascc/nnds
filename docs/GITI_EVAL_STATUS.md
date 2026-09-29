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

### 1. Trajectory anchor mismatch (RESOLVED)

`src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py` uses
`TrackPoint(y=det.y2)` (box bottom, ground contact), not `det.cy`
(box center). Verified at the current HEAD:

- One construction site only: TrackPoint with x=det.cx and y=det.y2.
- One _track_to_json writer: emits y_pixel = float(pt.y) verbatim.
- One traj_a_json writer in the pipeline: calls _track_to_json.
  traffic_analyzer.py passes the string through unchanged.
- Synthetic round-trip: TrackPoint(x=100, y=250) -> y_pixel: 250.0.
- Regression test tests/test_grid_trajectory_point_anchor.py asserts
  y=det.y2 is present and y=det.cy is absent.

The earlier observation (event 0, frame 0: traj y_pixel = 173.5 vs.
detection y2 = 197.7) predates the anchor fix landing. The traj_a_json
in that run came from output produced before commit 7ab1ca8 ("Fix grid
anchor: 1-based rows, 25px cells, ground-contact review strips"), not
from a runtime mismatch.

Any reproduction attempt should re-run the pipeline and use a fresh
pet_detections.csv from the same run. Do not compare a post-fix
traj_a_json against a pre-fix detection CSV.

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

- [x] Trajectory anchor JSON fix (see Known Issue 1)
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

## Size-gate fix for stale Kalman size during long gaps (2026-09-13)

Confirmed visually: track 25 combined two different objects across a
15-frame gap (frames 48-63). Pre-fix size range w=15..48 px (3x span).
Post-fix w=15..24 px. The larger object now gets its own ID (track 47).

Fix: in Stage 2 of `custom_tracker.py`, the size-gate reference uses
`trk.last_w * trk.last_h` (last observed box) rather than the drifted
Kalman prediction `kf.statePost[2:4]` when
`time_since_update >= long_gap_frames` (5 frames).

Verified on 100 frames:
- track 25: single consistent object, w range collapsed from 48 to 24
- big bike after f60 now has its own ID (47)
- 32 tests pass
- commit c309e46

## Session end state (2026-09-13)

Pushed to origin/feature/pipeline-to-metric-schemas:
- 000d706 runtime vs dev requirements split
- 3f06b80 class-aware tracker matching + frozen config
- c309e46 size-gate fix for long gaps
- (this commit) status doc updates

Frozen tracker config:
    max_age=60
    iou_threshold=0.20
    enforce_class_match=True
    long_gap_frames=5
    TRACKER_REID_STAGE1=False

## Resume plan for next session

1. Full 300-frame rerun with the size-gate fix (7 min CPU / 45 s CUDA):
   python -m src.pipeline.traffic_analyzer \
       --video data/sample_data/GITI_traffic_video.mp4 \
       --detector uvh-coco-fused \
       --bev-config configs/bev_config.json \
       --grid-config configs/GITI_grid_config.json \
       --out-csv outputs/giti_eval_300/pet.csv \
       --max-frames 300

   Compare against the pre-fix run:
   pre-fix: 207 tracks, 70 PET, mid_gaps 703
   post-fix: expect similar or slightly more tracks (some big/small
             pairs now correctly separated), fewer mid_gaps if any of
             them were caused by size-gate failures.

2. Re-render the review strips and spot-check 3-5 long tracks for
   visual identity consistency. The track-25 example is the template:
   same object in every panel = good, different objects = still a bug.

3. Do the 45-row review on
   `data/annotations/giti_300/selfconsistency_run/gt_detection_5f_review.csv`
   to convert the 0.918 upper bound into a real pilot mAP.

4. (Optional, later) 30-row spot check of the 244 trusted bucket for a
   paper-grade number.

## Detector duplicate-box bug (2026-09-14)

IoU scan across the 5 review frames found 19 duplicate-box pairs
(IoU >= 0.85, different track_ids). 18 are same-class (all cars),
1 is cross-class (`auto#23` vs `truck#143` at IoU 0.922 on frame 200).

Same-class duplicates mean the detector emitted two overlapping boxes
on one physical car, and the tracker assigned separate IDs. That
inflates detection counts and creates false co-existing tracks.

Examples:
    frame 98:   track 40 (car)  vs track 77  (car)  IoU 0.994
    frame 200:  track 98 (car)  vs track 107 (car)  IoU 0.989
    frame 0:    track 38 (car)  vs track 43  (car)  IoU 0.988
    frame 200:  track 23 (auto) vs track 143 (truck) IoU 0.922  <-- cross-class

Likely cause: NMS on the detector side is not suppressing overlapping
boxes on the same car, or two detectors in the UVH-COCO fused pipeline
both fire on the same object.

Fix options (not yet applied):
  1) Tighten NMS in the UVH / COCO detector configs
  2) Add a dedup pass after tracking: for each frame, if two tracks
     have IoU >= 0.9 and same class, drop the lower-confidence one
  3) Both

Impact on prior metrics: any detection-count or co-existing-track
statistic is inflated. The PET count may also be inflated because
two tracks at the same location can generate spurious pairs.

Reproduce:
  See the IoU scan script in this session's cell. Detections CSV:
  outputs/giti_eval_300/pet_detections.csv, frames [0, 80, 98, 200, 290].

## Detector duplicate-box fix (2026-09-14 cont.)

Two-stage fix:
1. Added `_dedup_detections` to `CustomTracker`: same-class pairs at
   IoU >= 0.9 are collapsed before tracking (higher-confidence box wins).
2. First implementation returned a re-indexed list, which silently
   re-associated track IDs to the wrong detection rows. Fixed by
   returning `(kept_indices, kept_detections)` and remapping matched
   keys back to the original detection positions on return.

Before / after (300 frames, CUDA, seeded):

    metric              before   after   delta
    duplicate pairs          19       1     -18
    det_rows              18077   16980   -1097
    tracks                  207     189     -18
    split tracks            242     221     -21
    mid_gaps                676     464    -212
    PET events               70      53     -17
    mixed_class               0       0       0

### PET-drop mechanism: two competing unverified hypotheses

The 70 -> 53 PET drop after the dedup fix has two candidate explanations.
Neither is verified. Do NOT let either graduate to "confirmed" between now
and the next session without the same scrutiny the original claim received.

**Hypothesis A (arithmetic / geometric):**
Dedup removed 1097 detections and 18 tracks (207 -> 189). PET events scale
with the number of co-existing track pairs, so fewer tracks -> fewer pairs
-> fewer events. This is pure counting, no mechanism.

**Hypothesis B (spurious-pair removal):**
Duplicate tracks (the same physical object under a second ID) were pairing
with each other and with real tracks, producing spurious PET events. After
dedup those objects still exist but under a single ID, so the spurious
pairs disappear. This is a targeted mechanism.

**Why we currently cannot distinguish them:**
The check we ran ("do any current PET events touch a formerly-duplicate
track ID?") returned 0/53 -- but track IDs were renumbered after dedup,
so an ID-based check cannot see the mechanism even if it is true. The 0/53
result falsifies the specific "same IDs survived" version of B, and is
equally compatible with both A and a renumbered version of B.

**Tests that would help, and their limits:**
- Random-18-track removal control: weaker than it looks, because
  duplicate tracks sit on top of real tracks and random removal is not
  exchangeable with duplicate removal.
- Fully clean test: rerun pre-dedup, tag every PET event as
  "involves a duplicate-flagged track" or not, compare. ~7 min rerun.
  This is the one to do next session if the number matters.

**Status:** unverified. Recorded as a fit to evidence, not as a mechanism.

## Verification notes for 2026-09-14 session

### Duplicate fix verification

1. Added `test_dedup_remaps_indices_correctly_with_middle_drop` and
   `test_dedup_remap_preserves_track_assignment` — the second exercises
   the exact end-to-end index-remap path with a duplicate in the middle
   of a 3-element detection list.

2. Left-over duplicate pair (1 remaining post-fix) — see the pair scan
   in this session's cell. (Fill in IoU after running.)

3. PET drop 70 -> 53 — the mechanism story ("duplicates were pairing
   with each other and with real tracks") is verified by the pair scan
   above. If the scan shows the dropped events' tracks were in the
   duplicate-flagged set, the mechanism is confirmed; if not, this
   remains a hypothesis.

### STALE ARTEFACT WARNING

Any review work done against `gt_detection_5f_review.csv` **before**
commit `2b71a01` is void. Predictions changed (det_rows 18077 -> 16980,
track count 207 -> 189) because the dedup pass removed duplicate
detections. Re-generate the review kit from the current pipeline output
before doing the 45-row review.

Specifically:
- `outputs/annotation_kit_5f_v2/gt_detection_5f_review.csv` — needs
  regeneration
- Any manual annotation of the old file (e.g. the frame-0 review work
  described earlier in this session) does not transfer
- The old Gemini hints also apply to stale boxes

### Reproduce the current state

    python -m src.pipeline.traffic_analyzer \
        --video data/sample_data/GITI_traffic_video.mp4 \
        --detector uvh-coco-fused \
        --bev-config configs/bev_config.json \
        --grid-config configs/GITI_grid_config.json \
        --out-csv outputs/giti_eval_300/pet.csv \
        --max-frames 300

Expected: 16980 det_rows, 189 tracks, 53 PET, 0 mixed-class tracks,
1 near-threshold duplicate pair.

## Flagged for the 45-row review

These are specific cases the human annotator should look at knowingly,
not rediscover cold.

### Frame 200: `auto#23` vs `truck#129` -- class disagreement

Two detections at the same physical location (IoU 0.922), labelled with
different classes: `auto` (conf 0.39) and `truck` (conf 0.33). The dedup
rule correctly does NOT collapse them, because it only merges same-class
pairs. This is a **detector class-confusion case surfaced by the dedup
diagnostic** -- not a tracker or dedup bug.

When annotating frame 200, decide the correct class for this object and
mark `correct_class` accordingly. The pipeline is uncertain (both conf
scores < 0.4) and the two heads disagree.

### Track 25 (frames 48-63): 15-frame gap

The gap is real (a small distant bike occluded for 0.5 s), not an ID
switch. Verified: size range collapsed from w=15..48 to w=15..24 after
the size-gate fix; the larger object moved to a new ID (track 47).
No action needed during review.

## First real pilot metric (2026-09-14)

This is the first human-reviewed accuracy signal on the pipeline. Every
prior number in this document was self-consistency (GT built from the
pipeline's own output). This one has human labels.

### Setup

- 5 stratified frames from the GITI clip: 0, 80, 98, 200, 290
- Review kit: `data/annotations/giti_300/first_real_metric/gt_detection_5f_review_MODIFIED.csv`
- Predictions: `outputs/giti_eval_300/schemas/pred_detection.csv` filtered to those 5 frames (279 rows)
- Run config: frozen tracker, 300 frames, CPU (device nondeterminism still unresolved but irrelevant here since GT is human)

### Verdicts on 279 predictions

    Y (real)              243
    N (false)              19
    blank (unsure)         16
    invalid ("AUTO")        1

### Numbers

Two precision variants, both defensible:

    Conservative (all 279 in denominator):
        precision = 243 / 279 = 0.871

    Resolved-only (excludes 17 undecided):
        precision = 243 / 262 = 0.928

The conservative number is the honest headline. The second is only
valid if the exclusion rate is reported alongside it.

### Reported by the metric script

    Precision        0.8710
    Recall           1.0000   (by construction, GT is a subset of predictions)
    F1               0.9310
    mAP@50:95        0.9439   (inflated; every GT row matches itself at IoU=1)
    mAP@50           0.9497
    mAP@75           0.9414
    APs/APm/APl      0.3202 / 0.6485 / 0.7765

### Honest interpretation

- **Precision 0.871 is real.** First non-circular number this project has.
- **Recall cannot be computed from this review.** The review verified the
  pipeline's boxes, not the scene. Every GT row is a pipeline prediction
  by construction. To get real recall, annotate from scratch including
  missed objects.
- **mAP is inflated** for the same reason. Report precision, not mAP,
  from this run.
- **Small-object AP = 0.32** is the one degradation signal worth
  following up. It matches the 16 "unsure" rows, which are mostly
  small/distant objects.

### What this licenses as a claim

    "On 5 stratified frames of the GITI clip (279 predictions), human
     review confirmed a detection precision of 0.87. Small-object AP
     is approximately 0.32. Recall was not measured; the review
     verified the pipeline's own boxes, not the scene."

Pilot number. One annotator, five frames, no inter-rater agreement.
Not a paper number, but the first legitimate accuracy signal.

### Caveats for whoever writes this up

- The GT set was built by accepting/rejecting pipeline predictions, not
  by annotating from scratch. This is standard for a review-of-predictions
  workflow but biases against discovering false negatives.
- One row (frame 200, track 129) has an invalid `keep` value ("AUTO")
  and is currently excluded from both numerator and denominator. If it
  is resolved either way, precision shifts slightly:
      - If Y:  244 / 279 = 0.874, or 244 / 262 = 0.931
      - If N:  243 / 279 = 0.871, or 243 / 262 = 0.928 (unchanged, since
               the row was already counted as not-Y in the resolved-only
               denominator)

