# Paper scope

## In scope (this paper)

Traffic conflict detection using post-encroachment time (PET) on
fixed-camera intersections with heterogeneous traffic.

Modules covered by the paper's tests and coverage floor:

- `src/pipeline/*` — detection, tracking, PET event extraction
- `src/analysis/grid_trajectory/*` — grid-based PET computation
- `src/analysis/pet_*`, `src/analysis/ssm_*`, `src/analysis/conflict_*`
- `src/analysis/track_*`, `src/analysis/detection_*`, `src/analysis/traj_*`
- `src/analysis/verification/*`, `src/analysis/visualization/*`
- `src/bev/*` — bird's-eye projection and calibration
- `src/core/*` — shared types and contracts
- `paper/analysis/*` — paper-specific statistical and validation modules

## Out of scope (separate research lines)

Present in the repository from prior work but not part of this paper's
scope. Their coverage and quality are not tracked against this paper's
floor.

- `src/diffusion/*` — trajectory diffusion models. Separate paper.
- `src/vlm/*` — vision-language model gate validation. Separate paper.
- `src/utils/*` — developer utilities (interactive shells, debug helpers,
  duration parsing). Used interactively, not in the pipeline.
- `scripts/agentic_fix*.py`, `scripts/benchmark_ollama_patch.py` —
  developer tooling.

## Why this matters

Mixing scopes inflates the denominator and hides the paper-relevant
coverage signal. Coverage on the in-scope set is the number a reviewer
should look at; coverage on the whole repo includes code the paper does
not use.

## Coverage policy

- CI enforces the 70% floor on the **in-scope set** (see `.coveragerc`).
- Repo-total coverage is reported but not enforced.
- SonarCloud is configured with the same exclusion (see
  `sonar-project.properties`).
