#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="$ROOT/weights"
TARGET="$WEIGHTS_DIR/uvh26.pt"
SOURCE_URL="https://raw.githubusercontent.com/Sharqascc/nnds/cleanup/system-reorganization/uvh26.pt"
EXPECTED_SIZE=57142241

mkdir -p "$WEIGHTS_DIR"

if [ -f "$TARGET" ]; then
  actual_size=$(stat -c%s "$TARGET")
  if [ "$actual_size" -eq "$EXPECTED_SIZE" ]; then
    echo "[OK] Existing file is already correct: $TARGET"
    exit 0
  fi
fi

if command -v git-lfs >/dev/null 2>&1; then
  git lfs install
  git lfs pull
  if [ -f "$TARGET" ]; then
    actual_size=$(stat -c%s "$TARGET")
    if [ "$actual_size" -eq "$EXPECTED_SIZE" ]; then
      echo "[OK] Pulled via Git LFS: $TARGET"
      exit 0
    fi
  fi
fi

if command -v curl >/dev/null 2>&1; then
  curl -L "$SOURCE_URL" -o "$TARGET"
else
  wget -O "$TARGET" "$SOURCE_URL"
fi

actual_size=$(stat -c%s "$TARGET")
if [ "$actual_size" -ne "$EXPECTED_SIZE" ]; then
  echo "[ERROR] Size mismatch: expected $EXPECTED_SIZE, got $actual_size"
  exit 1
fi

echo "[OK] Downloaded and verified $TARGET"
