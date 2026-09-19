#!/usr/bin/env bash
# Curated pre-push. Fast subset + metamorphic + determinism.
# Full property/differential suite lives in CI.
set -e
cd "$(git rev-parse --show-toplevel)"

CACHE_FILE=".last_quality_checks"
MAX_AGE=300

if [ -f "$CACHE_FILE" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$CACHE_FILE" 2>/dev/null || echo 0) ))
    if [ "$age" -lt "$MAX_AGE" ]; then
        echo "Recent checks (${age}s old). Skipping."
        exit 0
    fi
fi

echo "=== ruff ==="
ruff check src tests scripts
ruff format --check src tests scripts

echo "=== fast subset ==="
pytest tests/ -q -o addopts="" -n auto \
    -m "not property and not integration and not slow and not differential and not metamorphic" \
    --ignore=tests/test_snapshot_bev_mapper.py \
    --ignore=tests/test_snapshot_pet_summary.py \
    --timeout=60

echo "=== metamorphic + determinism ==="
pytest tests/test_metamorphic_ssm.py tests/test_determinism.py \
    -q -o addopts="" --timeout=60

touch "$CACHE_FILE"
echo "pre-push OK"
