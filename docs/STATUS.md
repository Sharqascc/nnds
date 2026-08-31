# Repository Status (Authoritative)

**Last updated:** 2026-08-31
**Git tag:** v1.6-paper

## Current Results

| Site | Screened Events | Median PET | Min PET | Max PET |
|------|----------------|------------|---------|---------|
| GITI | 153 | 1.566 s | 0.167 s | 2.999 s |
| MRC  | 34  | 1.600 s | 0.133 s | 2.999 s |

## Event Count Explanation

| Version | Count | Reason |
|---------|-------|--------|
| 156 | Pre-fix | BEV mapper bug (null world coords), PET computed incorrectly |
| 168 | First fix | BEV mapper fixed, PET recomputed (no screening yet) |
| 164 | Current raw | After same-origin exclusion (orig_a != orig_b) |
| **153** | **Current screened** | Raw 164 minus 11 events with PET < 0.10s |

## Event Screening Rules

1. **Same-origin exclusion:** Events where `orig_track_a == orig_track_b` are removed (same vehicle segments should not be compared).
2. **Temporal resolution:** Events with PET < 0.10s (3 frames) are removed as tracking artifacts.
3. **Temporal duplicate rule:** Same vehicle pair in same grid cell must be separated by >= 10 frames to be distinct episodes.

## Canonical Pipeline

See `MODULE_MANIFEST.md` for the exact path from video to PET events.

## Key Documents

- `MODULE_MANIFEST.md` — Canonical pipeline definition
- `scientific_audit.md` — Claim-to-code traceability
- `EXPERIMENTAL_MODULES.md` — Modules not used in paper
- `calibration_provenance.md` — Calibration details
- `DEPRECATED_CONFIGS.md` — Deprecated configs

## Deprecated Documents

The following documents are historical and may contain outdated numbers.
Do NOT use them for manuscript claims:

- `final_submission_summary.md` (mentions 156 events)
- `final_assessment_report.md`
- `comprehensive_assessment_report.md`
- `detection_system_report.md`
- `tracking_system_report.md`
- `repository_assessment.md`
- `repo_full_details.md`
- `CLEANUP_SUMMARY.md`
- `MIGRATION_GUIDE.md`
- `PUBLICATION_READINESS.md` (70 bytes - incomplete)
- `DEBUGGING.md` (debug log)
