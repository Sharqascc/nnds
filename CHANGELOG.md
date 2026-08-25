# Changelog

All notable changes to this project will be documented in this file.

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
