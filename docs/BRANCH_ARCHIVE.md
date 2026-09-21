# Branch archive index

Long-lived branches whose work is merged, superseded, or abandoned
are preserved as **`archive/*` tags** instead of branches. The tags
keep the commit graphs fully reachable from any clone; only the
branch refs themselves are gone.

## Browse an archive (works everywhere)

```bash
git fetch --tags
git tag -l 'archive/*'                  # list all archives
git log --oneline archive/<name>        # commit history
git show archive/<name>:path/to/file    # read any file
git diff HEAD...archive/<name>          # diff against a branch
git checkout archive/<name>             # detached HEAD for review
```

On GitHub's web UI: **Code → Tags** — https://github.com/Sharqascc/nnds/tags

## Restore an archive to a live branch

```bash
scripts/restore_archived_branch.sh <tag-suffix> [new-branch-name]
```

Example:

```bash
scripts/restore_archived_branch.sh cpu-yolo-debug
# recreates branch cpu-yolo-debug at archive/cpu-yolo-debug

scripts/restore_archived_branch.sh cpu-yolo-debug debug-restored
# uses a different branch name if the original is taken
```

**Restore caveat.** If the archived tree modifies files under
`.github/workflows/`, GitHub rejects a `git push` that creates the
branch unless the token has the `workflow` scope. The script prints
three workarounds in that case:

1. Create the branch from the tag on GitHub's Tags page.
2. Push over SSH (`git@github.com:...`) from a clone with an SSH key.
3. Regenerate the token with the `workflow` scope and retry.

This is a push-permission limitation, not data loss — the commits
remain readable from every clone with no special scope.

## Current archives

| Tag | Tip sha | Description |
|---|---|---|
| `archive/agentic-auto-fixes` | `498b0053e5a2` | 2026-09-07 — 39 commits: agentic LLM fixer workflow experiment. Superseded by agentic.yml on trunk (now disabled). |
| `archive/cleanup-system-reorganization` | `2892564fe4a1` | 2026-09-19 — 2 commits: property tests (with regressions — markers removed). Two net-new assertions salvaged via PR #25. |
| `archive/cleanup-vlm` | `3eb692235145` | 2026-08-17 — 2 commits: VLM gate validation pipeline. 428 commits behind trunk at archive time. |
| `archive/cpu-yolo-debug` | `6e3f0445f806` | 2026-07-29 — 15 patch-novel commits: CPU YOLO debug pipeline. 715 commits behind trunk. |
| `archive/feat-ssm-review-recovery` | `275584340d45` | 2026-09-19 — 10 commits: SSM review dataset reconstruction. Superset of tracker-cleanup. Evidence artifacts pending issue #22. |
| `archive/feat-video-to-pet-pipeline` | `032aa466e7e9` | 2026-04-25 — 20 patch-novel commits: early video-to-PET pipeline (PR #1 merged). 715 behind. |
| `archive/master` | `15a9de778dc0` | 2025-12-05 — legacy snapshot pre-restructure. Superseded by main and by feature/pipeline-to-metric-schemas. |
| `archive/pet-window-contract` | `e81661ea0ec8` | 2026-09-12 — 15 commits: UVH detector docs, BEV calibration export, SAM3 stub fixes. Clean parts salvaged via PR #23. |
| `archive/tracker-cleanup-and-eval` | `482327ba2dc2` | 2026-09-19 — 9 commits: tracker quality evidence + LLM review fixes. Code fixes salvaged via PR #24. |

## Policy

- A branch is archived (tagged) **before** deletion. The two
  operations are never separated.
- Archive tags are permanent. Deleting one is deliberate cleanup,
  not routine.
- A branch held by live automation is **not** archived or deleted
  until the automation that depends on it is removed or redirected.

## Add a new archive

```bash
BR=some/feature-branch
TAG=archive/$(echo "$BR" | tr '/' '-')
SHA=$(git rev-parse origin/$BR)
git tag "$TAG" "$SHA"
git push --no-verify origin "$TAG"
gh api -X DELETE "/repos/Sharqascc/nnds/git/refs/heads/$BR"
```

Add a row to the table above so the archive stays discoverable.

