# Contribution rule (non-negotiable)

## Every code change flows through GitHub CI. No local-only verification.

1. **Write the code** as a file under `src/` or `paper/analysis/`.
2. **Commit and push** to a feature branch. Do not run ruff or pytest locally.
3. **CI runs automatically** on every push (`ci.yml`: ruff check,
   ruff format --check, mypy, bandit, pytest with coverage floor,
   property tests).
4. **If CI fails**, read the failing step in the Actions tab, fix the
   specific line, push again. Loop until green.
5. **Merging to `main` requires the CI job `test` to pass** — enforced by
   branch protection with `enforce_admins: true`, so no one, including the
   owner, can bypass it.

## Why

Local lint and test runs duplicate CI and drift from it (different ruff
version, missing deps, cached results). CI is the single source of truth.
The only acceptable local commands are:

- `git status` / `git diff` — to see what will be committed
- Reading output files — to inspect results, not to validate code

## Enforced by

- `.github/workflows/ci.yml` — runs on every push
- Branch protection on `main` — requires the `test` job to pass, admins
  cannot override
