#!/usr/bin/env bash
# Curated pre-push. Ruff + mypy + fast subset + metamorphic + determinism.
# Full property/differential suite lives in CI.
set -e
cd "$(git rev-parse --show-toplevel)"

CACHE_FILE=".last_quality_checks"
MAX_AGE=300

# Cache is keyed on the tree hash, not just file mtime. A commit within
# the TTL window only skips checks if it does not change any tracked
# source. Prevents the prior behaviour where editing a file and pushing
# again reused a stale pass.
CACHE_KEY="$(git rev-parse HEAD)-$(git diff HEAD --name-only | sort | sha256sum | cut -c1-12)"
if [ -f "$CACHE_FILE" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$CACHE_FILE" 2>/dev/null || echo 0) ))
    stored="$(head -n1 "$CACHE_FILE" 2>/dev/null || true)"
    if [ "$age" -lt "$MAX_AGE" ] && [ "$stored" = "$CACHE_KEY" ]; then
        echo "Recent checks for this tree (${age}s old). Skipping."
        exit 0
    fi
fi

echo "=== ruff ==="
ruff check .
ruff format --check .

echo "=== import guardrail ==="
python scripts/check_imports.py

echo "=== mypy (analysis) ==="
mypy --config-file mypy.ini src/

echo "=== bandit (SAST) ==="
bandit -c bandit.yaml -r src/ -q

echo "=== shell syntax ==="
shopt -s nullglob
for f in scripts/*.sh .githooks/pre-commit .githooks/pre-push; do
    bash -n "$f" || { echo "shell syntax error: $f"; exit 1; }
done

echo "=== fast subset ==="
pytest tests/ -q -o addopts="" -n auto \
    -m "not property and not integration and not slow and not differential and not metamorphic" \
    --ignore=tests/test_snapshot_bev_mapper.py \
    --ignore=tests/test_snapshot_pet_summary.py \
    --timeout=60

echo "=== property subset (fast)" ===
pytest tests/test_pet_summary_property.py \
       tests/test_traffic_analyzer_property.py \
    -q -o addopts="" -n auto --timeout=60

echo "=== metamorphic + determinism ==="
pytest tests/test_metamorphic_ssm.py tests/test_determinism.py \
    -q -o addopts="" --timeout=60

echo "$CACHE_KEY" > "$CACHE_FILE"
echo "pre-push OK"
