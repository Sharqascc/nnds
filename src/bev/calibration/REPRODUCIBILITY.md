# Reproducibility

## Environment
- Python version: record the interpreter used.
- Dependencies: `requirements.txt` and/or `pyproject.toml`.

## Steps
1. Start from a clean clone.
2. Place source calibration inputs in `calibration/raw/`.
3. Run the calibration scripts.
4. Save outputs to `calibration/derived/` and publication figures to `calibration/reports/`.
5. Capture a file manifest and commit hash.

## Recommended commands
- `python calibration/grid_validation_calibration.py`
- `python calibration/monte_carlo_calibration_benchmark.py`

## Verification
Re-run the scripts from scratch and compare outputs with the saved manifest.
