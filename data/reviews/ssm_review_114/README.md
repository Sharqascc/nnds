# SSM review — 114 hand-labeled PET events

This directory holds the human review that underlies every PET/SSM
quality number in `docs/METRIC_STATUS.md`. It replaces the previously
reported 0.857 precision/recall/F1, whose derivation could not be
traced (`git log --all -- outputs/pet_pred_14.csv` returns nothing).

## Contents

| File | Description |
|---|---|
| `to_label.csv` | 114 events with reviewer verdicts (`Y` / `N`). 38 Y, 76 N. |
| `features.csv` | 13 features per event (angle_fl, consist_a, max_p95, ...) |
| `all_features.csv` | `features.csv` joined with `to_label.csv` and derived fields |
| `newfeatures.csv` | The two features from the post-review model: `decel_max`, `pred_min_sep` |
| `ev_jumpstats.csv` | Per-event tracker jump statistics (`max_p95`, `max_p99`, `max_frac`, `max_jump`) |
| `findings.md` | Review writeup with MCC, balanced accuracy, PR-AUC for each feature, cross-validated |

The source PET events (with trajectories) are not stored here —
they are ~5.7 MB and were already published as part of the pipeline
outputs. The `idx` column in each file indexes into
`outputs/giti_screened.csv`.

## Headline numbers (from `findings.md`)

- **55° angle gate**: MCC −0.106, CI [−0.307, +0.084] — worse than random
- **2-feature model** (`decel_max` + `pred_min_sep`): MCC +0.367, CI [+0.005, +0.723]
- **N-class precision** 0.814 ± 0.082, **N-class recall** 0.713 ± 0.124
- **Y-class precision** 0.548 ± 0.119, **Y-class recall** 0.660 ± 0.174

## Method

All scores computed with MCC (Matthews correlation coefficient)
rather than F1. The prior F1-based analysis was misleading: the
majority class (N = 76/114) gives a trivial F1 baseline of 0.800,
which the earlier report misread as signal. RepeatedStratifiedKFold
(5 folds, 100 repeats) was used for cross-validation.

## What this does and does not show

The review shows that (a) the previously reported 0.857 is
unsupported, (b) the shipped gate is worse than random on independent
labels, and (c) a 2-feature model built from post-hoc critiques has
weak but real signal. The CI is wide (±0.36) and the review does not
recommend deploying the 2-feature model.

## Adding labels

If you extend the review, update `to_label.csv` and re-run the
analysis in `findings.md`. Keep the file schema stable: other
artifacts reference it by `idx`.
