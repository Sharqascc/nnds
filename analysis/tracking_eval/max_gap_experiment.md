# max_frame_gap=12 experiment — null result

## What we tested

Ran `run_video_to_pet(max_frame_gap=12)` on the 100-frame clip.
Baseline used `max_frame_gap=5`.

## What we found

- Tracker output is byte-identical between the two runs:
  `md5(petevents_100f_detections.csv) == md5(petevents_gap12_detections.csv)`
  == `fe88601aa0d1429ea608d6af1d8422ab`
- 105 tracks in both. 19 short (1-3 frames), 49 long (61+) in both.
- Only the diagnostic splitter step differs: 113 segments vs 110.

## What max_frame_gap actually controls

`run_video_to_pet` has two layers:
1. Tracker — assigns track IDs to detections. Its gap tolerance is
   internal and not exposed via this argument.
2. Splitter — re-segments existing tracks using `max_gap` and `max_jump`.
   This is a diagnostic layer applied after tracking.

`max_frame_gap` only affects layer 2. The tracker keeps its own
config, which lives elsewhere in the pipeline.

## Why this matters

The 3 confirmed fragments in the 102-track stitched result (tracks
65/86/104, 87/100) are not caused by `max_frame_gap`. They come from
the tracker's own internal behaviour. Changing this parameter cannot
improve the tracker.

## Next step if tracker improvement is needed

Investigate `configs/tracktrack_reid.yaml` and the tracker source for
its actual gap/confirmation parameters. Not attempted in this session.
