#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Download pre-trained models required by the NNDS pipeline.
#
# Default detector is uvh-coco-fused, which needs:
#   - UVH-26 model (uvh26.pt)
#   - YOLO11n model (yolo11n.pt)
#
# Usage:
#   bash scripts/download_models.sh
# ---------------------------------------------------------------------------

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$ROOT/data/models"
mkdir -p "$MODELS_DIR"

# --------------------------------------------
# 1. UVH-26 (primary detection model)
# --------------------------------------------
UVH_URL="https://raw.githubusercontent.com/Sharqascc/nnds/cleanup/system-reorganization/uvh26.pt"
UVH_DEST="$MODELS_DIR/uvh26.pt"

if [ -f "$UVH_DEST" ] && [ -s "$UVH_DEST" ]; then
    echo "[OK] UVH model already exists: $UVH_DEST"
else
    echo "Downloading UVH model..."
    curl -L --fail --progress-bar "$UVH_URL" -o "$UVH_DEST"
    echo "[OK] UVH model saved to $UVH_DEST"
fi

# --------------------------------------------
# 2. YOLO11n (person fallback / COCO detection)
# --------------------------------------------
YOLO_URL="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
YOLO_DEST="$MODELS_DIR/yolo11n.pt"

if [ -f "$YOLO_DEST" ] && [ -s "$YOLO_DEST" ]; then
    echo "[OK] YOLO model already exists: $YOLO_DEST"
else
    echo "Downloading YOLO model..."
    curl -L --fail --progress-bar "$YOLO_URL" -o "$YOLO_DEST"
    echo "[OK] YOLO model saved to $YOLO_DEST"
fi

echo "Done! Models are in $MODELS_DIR"
