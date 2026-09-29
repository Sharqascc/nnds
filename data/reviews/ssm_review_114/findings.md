# SSM Review — final findings

## Headline

The 55° angle gate used in the SSM review document is anti-signal
(MCC −0.106). All 13 document-proposed features fail on 114 hand-labeled
PET events. A 2-feature model (post-event deceleration + projected
minimum separation) reaches weak out-of-sample MCC +0.367 (CI
[+0.005, +0.723]). The pipeline's reported 0.857 PET/SSM F1 has no
traceable ground truth.

## Method

114 labeled events (38 Y, 76 N). All scores computed with MCC (Matthews
correlation coefficient), balanced accuracy, and PR-AUC, replacing the
prior F1-based analysis whose majority-class baseline (F1 = 0.800) was
misleading. Cross-validation: RepeatedStratifiedKFold(5, 100).

## Current gate

`angle_fl > 55`:
- MCC −0.106, CI [−0.307, +0.084]
- Worse than random, worse than always-N

## Document-proposed features

| feature | thr | MCC | bal_acc | PR-AUC |
|---|---|---|---|---|
| angle_fl | 55 | −0.106 | 0.447 | 0.630 |
| angle_sg | 55 | −0.094 | 0.454 | 0.651 |
| angle_world | 55 | −0.088 | 0.461 | 0.690 |
| consist_a | 30 | −0.383 | 0.349 | 0.536 |
| consist_b | 30 | −0.142 | 0.434 | 0.620 |
| pet_wt_gap | 10 | −0.190 | 0.401 | 0.558 |
| max_gap_40 | 20 | +0.082 | 0.539 | 0.732 |
| max_frac | 0.05 | +0.137 | 0.559 | 0.766 |
| max_p95 | 20 | +0.204 | 0.579 | 0.774 |
| max_p99 | 50 | +0.231 | 0.605 | 0.751 |
| max_jump | 60 | +0.028 | 0.513 | 0.720 |

The PET-weighted gap ranked #1 by F1 and is anti-signal under MCC.
The F1 framing was the source of the prior "nothing works" conclusion.

## The 2-feature model

Logistic regression on `decel_max + pred_min_sep`:

- CV mean MCC: +0.367 ±0.175
- CV CI: [+0.005, +0.723]
- N precision: 0.814 ±0.082
- N recall: 0.713 ±0.124
- Y precision: 0.548 ±0.119
- Y recall: 0.660 ±0.174

Neither feature was proposed in the document. Both come from the
critiques that followed. Signal is weak but real, and out-of-sample.

## Prior validation is unverifiable

`docs/METRIC_STATUS.md` reports PET/SSM precision 0.857 against a
"53-event review (reconstructed)". The review file does not exist on
disk. `docs/VALIDATION.md` lists "PET / TTC accuracy" under Not
verified. `docs/STATUS.md` states that real ground-truth annotations
do not yet exist in the repository.

The 0.857 was introduced in git commit `abf6d9a`, replacing a 0.650
from PR #4 and a 0.264 from PR #3 — three gate iterations tuned
against the same (reconstructed) review.

Commit 35b8a3c (this branch) added a note to METRIC_STATUS.md
documenting that the referenced input files (`pet_pred_14.csv`,
`pet_gt_20.csv`, `pet_gt_y_only.csv`) were never committed.

## Recommendation

1. Retire the 55° gate. It is worse than random.
2. Do not deploy the 2-feature model. Weak signal, wide CI.
3. Correct the METRIC_STATUS.md "verified" tag on the PET/SSM row.
4. Benchmark the deployed velocity-strip gate against independent labels.
5. Collect more labels. 114 gives a CI of ±0.36 on the MCC.

## Reproducibility

| artifact | file |
|---|---|
| 114 events | `outputs/ssm_review_114_source.csv` |
| 114 verdicts | `outputs/ssm_review_114_to_label.csv` |
| features | `outputs/ssm_review_114_features.csv` |
| jump stats | `outputs/ssm_review_114_ev_jumpstats.csv` |
| new features | `outputs/ssm_review_114_newfeatures.csv` |
| merged | `outputs/ssm_review_114_all_features.csv` |

All numbers verified against the pre-loss session by exact match.
