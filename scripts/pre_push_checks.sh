#!/bin/bash
# Pre-push checks with caching to avoid duplicate Ruff/Pytest runs.
# If a recent successful check exists (<= 600 seconds), skip rerun.
set -e

CACHE_FILE=".last_quality_checks"
MAX_AGE_SECONDS=600

# Check if cache file exists and is recent
if [ -f "$CACHE_FILE" ]; then
    now=$(date +%s)
    mtime=$(stat -c %Y "$CACHE_FILE" 2>/dev/null || echo 0)
    age=$(( now - mtime ))
    if [ "$age" -lt "$MAX_AGE_SECONDS" ]; then
        echo "✅ Recent quality checks found ($age seconds old). Skipping rerun."
        exit 0
    fi
fi

echo "=== Running Ruff checks ==="
ruff check src tests scripts

echo "=== Running Pytest ==="
pytest -q --timeout=120 -o addopts=""   --ignore=tests/test_snapshot_bev_mapper.py   --ignore=tests/test_snapshot_pet_summary.py

# Create/update cache file on success
touch "$CACHE_FILE"
echo "✅ All pre-push checks passed."
