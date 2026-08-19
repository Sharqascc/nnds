# Provenance

## Origin
Calibration inputs are derived from the project’s calibration points, configuration files, and validation scripts.

## Transformations
- Grid validation calibration is run by `grid_validation_calibration.py`.
- Monte Carlo calibration benchmarking is run by `monte_carlo_calibration_benchmark.py`.

## Tracking
For each generated artifact, record:
- source input files,
- script version or commit hash,
- parameter values,
- output file names,
- timestamp of generation.
