# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] — freeze/class-colors (2026-11-09)

### Added
- Six-stage progress relabeling for the uvh grid-PET backend: `Setup`, `Detect + Track (fused per frame)`, `Index & Split Tracks`, `Grid Assignment`, `Gate + PET Event Computation`, `Write Outputs` (#76).
- `CANONICAL_STAGES` tuple in `src/analysis/_stages.py` as the single reference for stage names (#76).
- MEP formalization: `src/pipeline/traffic_analyzer.py` (console script `nnds`) documented as the main entry point, with a guard test against accidental rename or reassignment (#76).
- `track_b_exit_time_sec` column now populated in the CLI PET CSV (#78).
- Regression tests:
  - `tests/test_stages_ordered.py`: static AST checks for stage numbers and names
  - `tests/test_uhv_output_invariance.py`: AST-diff proving the relabel changed
    only stage-call arguments (output invariance)
  - `tests/test_pet_output_schema.py`: runtime + AST checks that
    `track_b_exit_time_sec` flows through to the DataFrame
  - `tests/test_entrypoint_hygiene.py::test_mep_is_traffic_analyzer`:
    subprocess guard against MEP drift
  - `tests/test_class_colors.py::test_class_name_to_id_keys_all_mapped_in_class_colors`:
    guard that every `CLASS_NAME_TO_ID` key has a colour

### Fixed
- Pedestrian detections previously rendered grey because `pedestrian` was
  missing from `CLASS_COLORS`; now yellow, matching `person`.
- `track_b_exit_time_sec` was always `NaN` in the CLI PET CSV — the uvh
  backend never wrote the field, so `_events_to_dataframe` fell back to
  `None` (#78).
- Stale `[1/5]..[5/5]` comment in `traffic_analyzer.py` updated to
  `[1/6]..[6/6]`.
- Inherited `ruff format` debt in `class_colors.py` and `test_class_colors.py`.

### Changed
- `uvh_coco_fused_grid_pet.py` now emits `begin(6)` and six
  `stages.stage(N, ...)` calls matching `CANONICAL_STAGES`.
- Old combined stage 4 (`Grid, Gate & PET`) split into `Grid Assignment`
  and `Gate + PET Event Computation` (measured honestly — gate attribution
  happens inside the PET stage).
- `pyproject.toml` and `docs/ENTRYPOINT_TODO.md` updated to name
  `traffic_analyzer.py` as the MEP; `scripts/run_pipeline.py` documented
  as a preflight/dispatch wrapper.

### Deferred (tracked)
- `yolo_cpu_grid_pet.py` and `sam3_grid_pet.py` not relabelled — Phase 3 of
  `docs/ENTRYPOINT_TODO.md`.
- Fake-detector smoke test superseded by `test_uhv_output_invariance`'s
  AST-diff, which is strictly stronger.
- PET conflict-zone geometry study — see `docs/PET_ZONE_TODO_2026-11-09.md`.
- SonarCloud external 403 from `api.sonarcloud.io/analysis/jres` — issue #77.
## [Unreleased]
### Added
- Industry‑standard files (LICENSE, CITATION.cff, CONTRIBUTING.md, CHANGELOG.md)
- GitHub Actions CI workflow
- Pre‑commit hooks configuration
- Code formatting with black/isort

### Changed
- Repository structure cleaned up for publication.

## [1.0.0] - 2026-08-11
- Initial release of the NNDS pipeline.


## [Unreleased] - 2026-08-25
### Added
- Paired t-test script for method comparison (`scripts/paired_ttest.py`)
- Sample ground truth annotation fixture (`tests/fixtures/ground_truth_sample.csv`)
- Anonymized sample video (`data/sample_data/anonymized_traffic_video_50f.mp4`)
- CI upload artifact for experiment log
- Baselines: constant acceleration, social force
- Ground truth evaluation script
- Model cards, seed management, data privacy docs
- Docker reproducibility files
- Statistical confidence intervals (bootstrap) in validation report

### Changed
- Minimal requirements.txt
- Removed stub modules
- Normalized BEV condition number reporting
- Improved PET output schema with explicit track/split columns and time columns
