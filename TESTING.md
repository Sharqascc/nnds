# Testing Strategy

## Coverage Overview
- Current unit test coverage on core testable modules: ~99% statement coverage, ~95% branch coverage.
- Heavy modules (diffusion training, VLM, video processing) are covered by smoke tests and integration tests, not full unit tests.

## Excluded/Untestable Modules
| Module | Reason |
|--------|--------|
| `src/diffusion/*` | Requires GPU and long training runs |
| `src/vlm/*` | Requires API keys or large models |
| `src/analysis/grid_trajectory/sam3_grid_pet.py` | Requires SAM3 model and video |
| `src/analysis/grid_trajectory/yolo_cpu_grid_pet.py` | Requires YOLO model and video |
| `src/utils/interactive.py` | Interactive display functions |
| `src/utils/debug_helpers.py` | Debug printing and debugging helpers |

## Mutation Testing
The installed `mutmut` version (3.7.0) does not support `--paths-to-mutate`.
A manual mutation test can be performed as follows:

1. Copy a core file, e.g., `src/analysis/conflict_classifier.py`.
2. Introduce a small mutation (e.g., change `<` to `>`).
3. Run the relevant test file.
4. Confirm tests fail; revert the mutation.

Example manual mutation:
- Original: `if pet < 1.0: return ConflictSeverity.CRITICAL`
- Mutated: `if pet > 1.0: return ConflictSeverity.CRITICAL`
- Expected: tests should fail.

## CI
GitHub Actions runs the fast unit suite (excluding integration/slow tests) on every push and PR with:
```bash
pytest tests/ -m "not integration and not slow" --cov=src --cov-branch --cov-fail-under=84
```

## Property-Based Testing
Uses Hypothesis to test invariants in PET computation and severity classification. See `tests/test_property_based.py`.
