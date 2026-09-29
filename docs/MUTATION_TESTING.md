# Mutation testing

Mutation testing measures how strong a test suite is by injecting small
faults into source code and checking whether any test notices.

## What it catches that coverage doesn't

Branch coverage tells you a line *executed*. Mutation testing tells you
an assertion *would have failed* if the line were wrong. A module with
100% branch coverage can still have weak assertions - mutation testing
finds the gaps.

On this repo, mutation testing has already paid off: an accidental leak
of a cosmic-ray mutant (`<` changed to `==` in `critical_conflict_recall`)
was caught by three existing tests, confirming the suite has real
detection power on that function.

## Why it is not in CI

Runtime. Measured on this repo:

| | value |
|---|---|
| per-mutant wall time | ~4.3 s |
| mutants per source line | ~2.05 |
| `src/analysis/ssm_error.py` (163 lines) | 335 mutants, ~24 min |
| full `src/` (~17,500 lines) | ~35,000 mutants, **~43 hours** |

Not viable as a PR gate, heavy even for nightly. Mutation testing here
is a **targeted, on-demand tool** - run it per module when you want to
strengthen that module's tests.

## Safety: the `local` distributor mutates in place

cosmic-ray's `local` distributor edits the source file, runs tests, and
restores. If the process is killed mid-run - timeout, Ctrl-C, OOM - the
mutation stays in the working tree. That is how the leak above happened.

Mitigations:

- Run inside a git worktree: `git worktree add ../mutation-check HEAD`
- Or, always `git checkout -- <module>` after an aborted run
- `mutants/` (cosmic-ray's scratch directory) is gitignored
- Prefer the `multiprocessing` distributor over `local` - it isolates
  mutation into worker processes and cannot leak into the working tree

## Procedure

### 1. Find the covering tests

Mutation tools do **not** infer which tests exercise a module. You
supply the list. If a covering test file is omitted, mutants will
falsely survive.

```bash
scripts/find_covering_tests.sh src/analysis/ssm_error.py
```

The script searches `tests/` for direct imports and indirect references
(bare module name). Review the output - direct matches should not be
missed; indirect matches may include false positives.

### 2. Write a cosmic-ray config

```toml
[cosmic-ray]
module-path = "src/analysis/ssm_error.py"
timeout = 60.0
excluded-modules = []
test-command = "python -m pytest tests/test_ssm_error.py tests/test_ssm_error_property.py tests/test_conflict_metrics.py tests/test_conflict_metrics_property.py tests/test_ssm_agreement.py tests/test_ssm_agreement_property.py -x -q -o addopts= -p no:cacheprovider --timeout=30"

[cosmic-ray.distributor]
name = "multiprocessing"
```

### 3. Run

```bash
cosmic-ray init   cosmic-ray-config.toml  session.sqlite
cosmic-ray exec   cosmic-ray-config.toml  session.sqlite
cr-report         session.sqlite
cr-rate           session.sqlite
```

`cr-rate` prints a survival percentage. Lower is better.

### 4. Interpret survivors

A surviving mutant is one of two things:

- **A real test gap.** No test pins the behavior that changed. Add an
  assertion.
- **An equivalent mutant.** The change does not alter observable
  behavior (e.g. `x + 0` vs `x * 1`). Ignore.

Inspect each survivor's diff before deciding:

```bash
cosmic-ray dump session.sqlite | python -m json.tool | less
```

## Recovery after an aborted run

If the run is killed, the working tree may hold a leaked mutation:

```bash
git status                      # look for unexpected modified files
git checkout -- <changed files> # restore
rm -rf mutants/                 # cosmic-ray scratch dir
```

## Scope guidance

| Target | Mutants | Est. time | Recommended |
|---|---|---|---|
| one module (~150 lines) | ~300 | ~25 min | yes, on-demand |
| one package (~1,000 lines) | ~2,000 | ~2.5 hr | yes, quarterly |
| full `src/` (~17,500 lines) | ~35,000 | ~43 hr | no, unless distributed |

The metric modules (`src/analysis/ssm_error.py`,
`src/analysis/tracking_metrics.py`, `src/analysis/detection_metrics.py`,
`src/analysis/traj_error.py`, `src/analysis/bev_error.py`) are the
highest-value targets - they compute the numbers the repo reports.

## Alternative tools

- **mutmut** - does not handle `src/`-layout packages: its trampoline
  rejects modules whose import name starts with `src.`
- **pytest-gremlins** - mutation switching; keeps the test process hot
  rather than reloading numpy per mutant. Worth trying if cosmic-ray's
  per-mutant cost becomes the bottleneck.
