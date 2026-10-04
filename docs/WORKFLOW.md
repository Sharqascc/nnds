# Working rule — GitHub is the only executor

## Non-negotiable

No code runs on your local machine or in Colab.

Every code change and every analysis is executed by GitHub Actions on a
GitHub-hosted runner. Colab is a thin client: it uploads content to GitHub
and downloads results.

## Why

- One executor. No drift between "works on my machine" and CI.
- Every change is linted, type-checked, tested, and — for analyses — run
  and archived as a downloadable artifact.
- No local state to lose when the Colab runtime restarts.
- Reproducible: the same commit, the same environment, the same result.

## The three operations

### 1. Write a file (creates a commit directly on GitHub)

Use the Contents API. Do not use `git clone`, `git add`, `git commit`,
`git push` from Colab. Do not use `Path.write_text`.

See `docs/colab_api_helper.py` for a ready-to-paste helper.

### 2. Run code

Two workflows do this:

- `.github/workflows/ci.yml` runs on every push. Lint, type-check, bandit,
  pytest with coverage floor, property tests.
- `.github/workflows/paper-analysis.yml` runs any module under
  `paper.analysis.*`. Triggered by `workflow_dispatch` or by pushing a
  change under `paper/analysis/`.

To trigger paper analysis on a specific module, call the
`workflow_dispatch` endpoint via API (see helper), or open the Actions
tab and click Run workflow.

### 3. Read results

Results are attached to the workflow run as artifacts. Download via the
GitHub API:

- `GET /repos/{owner}/{repo}/actions/runs/{run_id}/artifacts` lists them
- `GET /repos/{owner}/{repo}/actions/artifacts/{artifact_id}/zip` returns
  the zip

Unzip in memory and inspect. Never write to disk in Colab.

## What Colab is allowed to do

- Call GitHub API endpoints with a PAT
- Print JSON from responses
- Hold data in Python variables during the session
- Optionally cache artifacts in memory for the current session only

## What Colab is not allowed to do

- `git clone`, `git pull`, `git add`, `git commit`, `git push`
- `Path.write_text` on project files
- `subprocess.run("python -m paper.analysis...")`
- `ruff`, `mypy`, `pytest`, `bandit` — these run only in CI
- Any command that produces a file on the local filesystem

## Branch rules

- `main` is protected. Merging requires PR + green CI (`test` job).
  Enforced with `enforce_admins: true`, so even the owner cannot bypass.
- Feature branches are free to iterate on. Push directly.
- All new work goes on a feature branch. No direct commits to `main`.

## Making a change

1. Open a Colab cell.
2. Paste the helper from `docs/colab_api_helper.py`.
3. Set `PATH`, `CONTENT`, `MESSAGE`.
4. Call `put_file(PATH, CONTENT, MESSAGE)`.
5. Watch CI at `https://github.com/Sharqascc/nnds/actions`.
6. If CI fails, read the log via API (see helper `fetch_log`).
7. Fix and push again.
8. When green, open a PR to `main`.

## Fetching a workflow log

If CI fails, the exact error is in the run log. The helper has a
`fetch_log(run_id, filename_filter)` function that returns the decoded
log text so you can paste it into a chat without touching disk.
