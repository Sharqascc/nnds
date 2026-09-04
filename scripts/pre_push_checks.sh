#!/bin/bash
# Pre-push checks: run Ruff and pytest before allowing a push.
set -e

echo "=== Running Ruff checks ==="
ruff check src tests scripts

echo "=== Running Pytest ==="
pytest -q --timeout=120 -o addopts="" \
  --ignore=tests/test_snapshot_bev_mapper.py \
  --ignore=tests/test_snapshot_pet_summary.py

echo "✅ All pre-push checks passed."
