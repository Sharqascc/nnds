# Contribution rule (non-negotiable)

**Every change goes through GitHub. Nothing runs on a local machine or in Colab.**

Full details: `docs/WORKFLOW.md`.
Helper for Colab: `docs/colab_api_helper.py`.

## TL;DR

1. Write files to the repo via the GitHub Contents API. No `git clone`, no
   `git add`, no `Path.write_text`.
2. CI runs automatically on push (`ci.yml`). Lint, type-check, bandit,
   pytest with coverage floor, property tests.
3. Analysis modules run through `paper-analysis.yml`. Results are
   downloadable artifacts, not local files.
4. If CI fails, read the log via the API, patch, push again. Loop until
   green.
5. `main` is protected. Merging requires green CI. Admins cannot bypass.

## Why

One executor (GitHub Actions) means no drift between environments. Every
change is reproducible from its commit SHA alone. Nothing is lost when
the Colab runtime restarts.

<!-- coderabbit-retrigger -->
