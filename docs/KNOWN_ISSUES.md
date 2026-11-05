# Known Issues

**Last updated:** 2026-11-05

This file lists review findings and other known limitations deferred from the current release. Each is tracked, sized, and can become its own PR. None block the release described in [`REPRODUCE.md`](../REPRODUCE.md).

## Summary

- **Entry-point findings:** all resolved. See PR #49's resolved threads and the fix PRs #60, #64, #65, #66, #67, #69, #70, #71.
- **Non-entry-point findings:** 59 open, tracked in **#72**.

## Non-entry-point findings (59)

These are findings on code not touched by the entry-point hardening work. Full tracking, per-file detail, and suggested order are in **#72**.

Files with open findings:

- `src/analysis/visualization/render_full_video.py` — 6 findings
- `paper/analysis/build_fp_clips.py` — 3 findings
- `paper/analysis/loose_candidates.py` — 3 findings
- `scripts/pipeline_to_metric_schemas.py` — 3 findings
- `src/diffusion/complete_ddpm.py` — 3 findings
- `tests/test_paper_analysis_pipeline_integration.py` — 3 findings
- `paper/analysis/gt_validation.py` — 2 findings
- `paper/analysis/tracklet_stitching.py` — 2 findings
- `src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py` — 2 findings
- `.githooks/pre-commit` — 1 finding
- `.github/workflows/agentic_auto_fix.yml` — 1 finding
- `.github/workflows/ollama_auto_fix.yml` — 1 finding
- `.github/workflows/paper-analysis.yml` — 1 finding
- `.pylintrc` — 1 finding
- `Makefile` — 1 finding
- `configs/sites/giti/bev_config.json` — 1 finding
- `configs/sites/giti/calibration_points.json` — 1 finding
- `docs/WORKFLOW.md` — 1 finding
- `docs/colab_api_helper.py` — 1 finding
- `paper/analysis/blind_windows.py` — 1 finding
- `paper/annotations/annotate.html` — 1 finding
- `pyproject.toml` — 1 finding
- `scripts/agentic_fix.py` — 1 finding
- `scripts/evaluate_ground_truth.py` — 1 finding
- `scripts/evaluate_position_ddpm.py` — 1 finding
- `scripts/evaluate_tracking_metrics.py` — 1 finding
- `scripts/pre_push_checks.sh` — 1 finding
- `src/analysis/gt_validation.py` — 1 finding
- `src/analysis/pipeline_outputs.py` — 1 finding
- `src/analysis/tracking_metrics.py` — 1 finding
- `src/pipeline/custom_tracker.py` — 1 finding
- `tests/test_bev_grid_pet_integration.py` — 1 finding
- `tests/test_classifier_summary_integration.py` — 1 finding
- `tests/test_core_roundtrip_integration.py` — 1 finding
- `tests/test_cross_site_integration.py` — 1 finding
- `tests/test_metrics_chain_integration.py` — 1 finding
- `tests/test_preflight_property.py` — 1 finding
- `tests/test_safety_eval_diffusion_notebook_property.py` — 1 finding
- `tests/test_ssm_review_chain_integration.py` — 1 finding
- `tests/test_tracker_health_property.py` — 1 finding
- `tests/test_tracker_pet_integration.py` — 1 finding


## Why these are deferred

PR #49's review feedback spans two very different surfaces:

1. **The entry point** — `nnds`, `nnds --self-check`, `--out-video`, and the CI workflows that exercise them. This is the interface a reviewer hits first. All findings here have been fixed and verified in the entry-point hardening series.
2. **The wider codebase** — `paper/analysis/`, `src/diffusion/`, integration tests, docs. These are research code, training scripts, and cross-module tests. None are on the release path; all are catalogued in the tracking issue.

Deferring #2 keeps the release honest: the reviewer sees exactly what is fixed, and **#72** is where the work continues. Merging a release that pretends the deferred findings do not exist would be worse; blocking the release on weeks of orthogonal cleanup would be unreasonable.

## Verification

The entry-point fixes were verified against source before resolving their threads. See `docs/ENTRYPOINT_TODO.md` for the full verification table. The seven most-scrutinised threads (MPS device check, detector in push.paths, NNDS_VIDEO_DIR, max_frames in filenames, out_csv default, coverage --append, path traversal) each pass a source-code check on the release branch tip.

## Related PRs

- #49 — this release (feature → main)
- #60, #64, #65, #66, #67, #69, #70, #71 — entry-point hardening series
- #72 — tracking for the 59 non-entry-point findings above
