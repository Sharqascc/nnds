# NNDS Testing Roadmap

This document is the permanent plan for strengthening the NNDS test suite.
It distinguishes software correctness from scientific validity.

The objective is not merely to reach a coverage percentage. The objective is to
establish that the traffic-safety research pipeline produces mathematically
correct, reproducible, and scientifically defensible results.

## Current CI gate

The required GitHub Actions workflow is `CI`. Its final required check is `test`.

The current gate requires:

- Static analysis: `compileall`, Ruff lint, Ruff format check, MyPy, and Bandit.
- Fast unit tests.
- Hypothesis property tests.
- Integration, smoke, and full/slow test lanes.
- Combined test coverage of at least 70 percent.
- The final `test` check must succeed only when every upstream lane succeeds.

Do not weaken or bypass these checks to make new tests pass.

## Implementation principles

1. Inspect the repository before adding tests.
2. Avoid duplicating existing tests.
3. Do not modify scientific algorithms merely to make tests pass.
4. Use analytically verifiable synthetic datasets for mathematical tests.
5. Include normal, boundary, extreme, and invalid-input cases.
6. Use Hypothesis for appropriate mathematical invariants.
7. Keep pull-request tests deterministic and reasonably fast.
8. Move expensive research experiments to scheduled or dedicated workflows.
9. Preserve the existing CI interface and required `test` check name.
10. Ensure every new test is discovered and executed by CI.
11. Do not claim that a test passes unless it has actually been executed successfully.

## Priority 0: Scientific correctness

| Test module | Purpose |
|---|---|
| `tests/test_safety_physics.py` | Validate TTC, DCPA, TCPA, closing speed, and PET calculations |
| `tests/test_conflict_geometry.py` | Validate conflict-zone membership, trajectory intersection, headings, and coordinate consistency |
| `tests/test_bev_units.py` | Validate pixel-to-world transforms, homography round trips, and unit consistency |
| `tests/test_temporal_integrity.py` | Validate monotonic timestamps, time-window boundaries, temporal ordering, and absence of future-information leakage |
| `tests/test_trajectory_integrity.py` | Validate handling of missing frames, duplicate IDs, ID switches, out-of-order points, and short trajectories |
| `tests/test_data_units.py` | Detect NaN/inf values, invalid bounding boxes, impossible speeds or accelerations, and unit inconsistencies |
| `tests/test_density_confound.py` | Ensure conflict measures are not merely proxies for traffic volume or exposure |

### Safety physics requirements

The safety-physics tests must include:

- TTC for approaching vehicles.
- TTC must not predict a future collision when vehicles are already close but moving apart.
- DCPA and TCPA for controlled two-vehicle scenarios.
- PET for controlled crossing, rear-end, and side-swipe scenarios.
- Relative speed and acceleration calculations.
- Collision-zone identification.
- Correct handling of zero velocity, parallel motion, and diverging motion.

Use analytically known synthetic trajectories. For example, two vehicles on
known straight-line paths should produce a calculable closest-approach distance
and time.

### Geometry requirements

The geometry tests must include:

- Homography forward and inverse transformations.
- Pixel-to-meter and world-coordinate consistency.
- Point-in-polygon tests.
- Vehicle-trajectory intersection.
- Conflict-zone boundary behavior.
- Heading and angle calculations.
- Coordinate-system consistency between pixel, BEV, and world representations.

### Temporal integrity requirements

The temporal tests must include:

- Timestamps and frame indices remain monotonic where required.
- PET remains consistent with vehicle entry and exit times.
- Sustained-conflict detection respects configured durations.
- Overlapping time windows do not create duplicate observations.
- Future information must not leak into past predictions.
- Boundary conditions at the start and end of trajectories are handled safely.

### Density confounding

This is a mandatory scientific validation test.

The test must demonstrate that conflict rate or severity is not explained solely
by the number of vehicles in a scene.

Use controlled synthetic scenarios:

1. Low traffic volume with several close conflicts.
2. High traffic volume with well-separated vehicles.
3. High traffic volume with the same conflict geometry as the low-volume case.

The test should show that traffic density alone does not automatically create
high-severity conflicts. Conflict measures should reflect interaction geometry
and exposure-adjusted risk, not merely vehicle count.

## Priority 1: Research reliability

| Test module | Purpose |
|---|---|
| `tests/test_conflict_labels.py` | Validate conflict-label rules and stability under small perturbations |
| `tests/test_threshold_sensitivity.py` | Ensure conclusions are not overly sensitive to arbitrary PET or TTC thresholds |
| `tests/test_grouped_splits.py` | Prevent leakage of trajectories, videos, sites, or time windows between splits |
| `tests/test_feature_leakage.py` | Prevent target or future information from appearing in predictor columns |
| `tests/test_reproducibility_artifacts.py` | Validate seed determinism, configuration recording, and artifact equivalence |
| `tests/test_pipeline_dry_run.py` | Ensure dry-run mode does not write outputs and missing inputs fail clearly |
| `tests/test_artifact_validation.py` | Validate CSV, JSON, Markdown, and figure artifacts |

## Priority 2: Robustness and generalization

These tests are important but should generally run in nightly or dedicated
research-validation workflows, not every pull request.

| Area | Purpose |
|---|---|
| Noise sensitivity | Add trajectory measurement noise and verify graceful degradation |
| Missing observations | Remove frames and verify safe behavior |
| Detection perturbations | Perturb detection positions and test stability |
| Traffic-density stress tests | Test low, normal, and high-density scenarios |
| Model calibration | Validate probability calibration and error metrics |
| Ablation testing | Identify which features contribute to predictions |
| Baseline comparison | Compare advanced models against simpler baselines |
| Cross-site validation | Evaluate generalization across intersections |
| Cross-video validation | Evaluate generalization across unseen recordings |
| GPU/CPU parity | Compare numerical behavior across devices when applicable |

## Existing tests to audit

Before creating a new test, inspect the existing suite. The repository already
contains tests related to many of these areas, including:

- `tests/test_pet_computation.py`
- `tests/test_pet_output_schema.py`
- `tests/test_pet_logic.py`
- `tests/test_pet_interval.py`
- `tests/test_bev_error.py`
- `tests/test_bev_calibration.py`
- `tests/test_bev_mapper.py`
- `tests/test_giti_bev_calib.py`
- `tests/test_track_stability.py`
- `tests/test_tracking_metrics.py`
- `tests/test_scientific_invariants.py`
- `tests/test_reproducibility_audit.py`
- `tests/test_determinism.py`
- `tests/test_pipeline_outputs.py`
- `tests/test_artifact_consistency_integration.py`
- `tests/test_validate_outputs.py`
- `tests/test_conflict_metrics.py`
- `tests/test_conflict_classifier.py`
- `tests/test_grid_validation_calibration.py`

The audit must determine whether existing tests already cover a proposed
scenario. Extend an existing test when appropriate instead of creating a
duplicate module.

## CI integration plan

After the current CI lanes pass, add a `test-scientific` lane to
`.github/workflows/ci.yml`.

The lane should run the Priority 0 scientific tests:

```yaml
  test-scientific:
    name: Scientific correctness tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          persist-credentials: false
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-dev.txt
      - name: Run scientific correctness tests
        run: |
          pytest tests/test_safety_physics.py \
                 tests/test_conflict_geometry.py \
                 tests/test_bev_units.py \
                 tests/test_temporal_integrity.py \
                 tests/test_trajectory_integrity.py \
                 tests/test_data_units.py \
                 tests/test_density_confound.py \
                 -q
```

Then update the required `test` job:

```yaml
  test:
    name: test
    needs:
      [
        static,
        test-fast,
        test-property,
        test-full,
        coverage,
        test-scientific,
      ]
```

This preserves the existing required check name, `test`, while making
scientific correctness mandatory.

## Empty-lane guard

Add a collection guard so a marker typo cannot create a green-but-empty CI lane.

For each lane, verify that pytest selects at least one test:

```bash
pytest tests/ --collect-only -q -m "<lane-marker>" | tail -1
```

If the output is `no tests ran`, fail the CI job.

Apply this guard to:

- Fast tests.
- Property tests.
- Integration tests.
- Smoke tests.
- Full or slow tests.
- Scientific tests.

## Deferred improvements

Do not add these to the required pull-request gate initially:

| Improvement | Recommended location |
|---|---|
| Dependency vulnerability scanning with `pip-audit` or `safety` | Weekly security workflow |
| Dead-code detection with `vulture` | Scheduled quality workflow |
| Cyclomatic-complexity checks with `radon` | Scheduled quality workflow |
| Notebook execution | Nightly or manual research workflow |
| Mutation testing | Scheduled quality audit |
| Performance benchmarking | Scheduled performance workflow |
| GPU/CPU parity tests | Nightly workflow with GPU runners |
| Cross-site and cross-video generalization | Dedicated research-validation workflow |

## Implementation sequence

1. Run and fix the existing static-analysis lane.
2. Run and fix the existing fast-test lane.
3. Run property, integration, smoke, and full/slow lanes.
4. Confirm the combined coverage gate.
5. Produce a repository-wide test-gap report.
6. Implement the Priority 0 scientific tests.
7. Add the `test-scientific` CI lane.
8. Include `test-scientific` in the required `test` check.
9. Implement Priority 1 tests.
10. Move expensive research validation to scheduled workflows.

## Test-gap report

Before implementing new tests, produce a repository-wide audit containing:

1. Existing test modules and the scientific or engineering concern they cover.
2. Gaps in mathematical correctness.
3. Gaps in geometry and coordinate handling.
4. Gaps in temporal integrity and leakage prevention.
5. Gaps in tracking reliability.
6. Gaps in data integrity and unit consistency.
7. Gaps in reproducibility and artifact validation.
8. Gaps in density confounding and exposure adjustment.
9. A prioritized implementation plan.
10. Test execution results and coverage report.

Do not begin implementation until this audit is complete.
