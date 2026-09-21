#!/usr/bin/env bash
# Restore an archived branch from its archive/* tag.
#
# Usage:
#   scripts/restore_archived_branch.sh <tag-suffix> [new-branch-name]
#
# tag-suffix is the part after "archive/" (e.g. "cpu-yolo-debug").
# new-branch-name defaults to the tag suffix.
#
# SAFETY:
#   - Never overwrites an existing branch on the remote.
#   - Never force-pushes.
#   - If the restore fails because the archived tree modifies files under
#     .github/workflows/, the script prints instructions for the two
#     workarounds (GitHub UI, or SSH/workflow-scoped token).
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <tag-suffix> [new-branch-name]"
    echo
    echo "available archives:"
    git ls-remote --tags origin 'refs/tags/archive/*' \
        | sed 's|.*refs/tags/archive/||' \
        | sed 's|\^{}||' \
        | sort -u \
        | sed 's/^/  archive\//'
    exit 2
fi

TAG="archive/$1"
TARGET="${2:-$1}"

# --- check the tag exists on origin ---
if ! git ls-remote --tags origin "refs/tags/$TAG" | grep -q "refs/tags/$TAG"; then
    echo "error: tag not found on origin: $TAG"
    exit 1
fi

# --- check the target branch does not already exist ---
if git ls-remote --heads origin "refs/heads/$TARGET" | grep -q "refs/heads/$TARGET"; then
    echo "error: branch already exists on origin: $TARGET"
    echo "choose a different name: $0 $1 <other-name>"
    exit 1
fi

# --- fetch the tag locally ---
git fetch origin "refs/tags/$TAG:refs/tags/$TAG" 2>/dev/null || true

# --- attempt the push ---
set +e
OUTPUT=$(git push --no-verify origin "refs/tags/$TAG:refs/heads/$TARGET" 2>&1)
RC=$?
set -e

if [ $RC -eq 0 ]; then
    echo "restored: $TAG -> $TARGET"
    exit 0
fi

# --- diagnose common failure ---
echo "$OUTPUT" >&2
echo >&2

if echo "$OUTPUT" | grep -qi 'workflow'; then
    cat >&2 <<'EOF'
========================================================================
Restore blocked: the archived tree modifies files under
.github/workflows/, and the current git token does not have the
"workflow" scope required to push such refs.

Workarounds:

  1. GitHub web UI:
     https://github.com/Sharqascc/nnds/tags
     -> find the archive/* tag
     -> "..." menu -> "Create branch from tag..."

  2. SSH remote (if you have an SSH key registered):
     git push git@github.com:Sharqascc/nnds.git \
         "refs/tags/archive/<suffix>:refs/heads/<branch>"

  3. Regenerate the PAT with "workflow" scope, then re-run this script.

The commits themselves are intact — this is a push-permission issue, not
a data-loss issue. Read-only access via `git log archive/<name>` and
`git show archive/<name>:<path>` works without any special scope.
========================================================================
EOF
fi

exit $RC
