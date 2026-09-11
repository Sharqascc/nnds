# Validation Status

This file records what has been verified about the NNDS evaluation framework,
and what has not been verified. It is deliberately explicit about the
difference between "the metric code is correct" and "the pipeline is accurate".

## Verified

- Every metric module in src/analysis/ (detection, tracking, BEV, trajectory,
  SSM, agreement, gold-standard) has unit tests and Hypothesis property tests.
- scripts/validate_gt.py correctly rejects malformed ground truth
  (missing columns, degenerate boxes, non-finite values, wrong PET sign).
- scripts/evaluate_*_metrics.py run end-to-end on synthetic data and emit
  well-formed JSON via --out-json.
- scripts/gold_standard_report.py assembles the 15-row table from those
  JSON files.
- scripts/demo_end_to_end.py exercises the full pipeline on synthetic GT
  and produces a complete report.
- ruff check . is clean. ruff format . is applied.
- mypy is clean on all new modules and scripts.
- pytest -q reports 1359 passed, 1 skipped, 0 failed.

## Not verified (and why)

- Detection accuracy. No annotated bounding-box ground truth for any real
  sequence exists in the repo. tests/fixtures/sample_detections.csv is a
  synthetic smoke-test file, not annotation.
- Tracking accuracy. Same reason; no ground-truth track IDs.
- BEV calibration accuracy. The current calibration is evaluated on its own
  fit points (see scripts/validate_bev.py). A held-out validation set is
  needed for an independent position-error number.
- Velocity / acceleration accuracy. Requires a reference trajectory
  (surveyed points or radar), not just pixel annotations.
- PET / TTC accuracy. Requires human-annotated conflict events on a real
  sequence.

## What this means in plain terms

Passing 1359 tests is evidence the metric library is well-built. It is not
evidence that NNDS achieves any particular mAP, HOTA, or PET error.

Any number reported in the paper must come from running the metric scripts
against real ground truth. The demo's output is a plumbing check, not a
result.

## Minimum path to real numbers

1. Pick a 30-60 s clip from data/sample_data/.
2. Annotate into the four CSVs per docs/ANNOTATION_GUIDE.md.
3. Run scripts/validate_gt.py and resolve every error.
4. Run scripts/run_pipeline.py on the same clip to produce predictions.
5. Run the four evaluate_*_metrics.py scripts with --out-json.
6. Run scripts/gold_standard_report.py --out-md outputs/validation_report.md.

That produces the real evaluation table for the paper.

## How to reproduce the smoke test

    python scripts/demo_end_to_end.py --out-dir outputs/e2e_demo
    cat outputs/e2e_demo/validation_report.md

Expected: a markdown table with 15 data rows, all numbers finite. The values
are meaningless as accuracy metrics (they come from synthetic predictions),
but a successful run proves the plumbing is intact.
