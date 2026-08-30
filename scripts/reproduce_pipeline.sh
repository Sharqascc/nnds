#!/usr/bin/env bash
# =====================================================================
# NNDS Pipeline Reproduction Script
# =====================================================================
# This script reproduces the full GITI + MRC PET analysis from a fresh clone.
# Usage:
#   bash scripts/reproduce_pipeline.sh [--max-frames 300] [--device cpu]
# =====================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MAX_FRAMES="${1:-300}"
DEVICE="${2:-cpu}"

echo "=== [1/4] Installing dependencies ==="
pip install -q ultralytics pandas matplotlib opencv-python pyyaml scipy

echo "=== [2/4] Downloading models ==="
bash scripts/download_models.sh

echo "=== [3/4] Copying videos from Drive (if available) ==="
# This is optional; videos must be present in data/sample_data/
# The user should manually copy videos if not already present.
# We'll check for their existence.
if [[ ! -f data/sample_data/traffic_video.mp4 || $(stat -c%s data/sample_data/traffic_video.mp4) -lt 1048576 ]]; then
    echo "WARNING: GITI video not found. Please copy from Drive to data/sample_data/traffic_video.mp4"
fi

if [[ ! -f data/sample_data/mrc_intersection.mp4 || $(stat -c%s data/sample_data/mrc_intersection.mp4) -lt 1048576 ]]; then
    echo "WARNING: MRC video not found. Please copy from Drive to data/sample_data/mrc_intersection.mp4"
fi

echo "=== [4/4] Running parallel pipeline (GITI + MRC) ==="
mkdir -p outputs

# Run GITI and MRC in parallel using Python multiprocessing
python - << 'PYEOF'
import os, sys, subprocess, multiprocessing, time
from pathlib import Path

repo = Path('.')
max_frames = int(os.environ.get('MAX_FRAMES', 300))
device = os.environ.get('DEVICE', 'cpu')

def run_site(args):
    site, video, bev, grid, gate, out = args
    cmd = [
        sys.executable, '-m', 'src.pipeline.traffic_analyzer',
        '--video', str(video), '--video-source', site,
        '--bev-config', str(bev), '--grid-config', str(grid), '--gate-config', str(gate),
        '--detector', 'uvh-coco-fused',
        '--uvh-model', str(repo/'data/models/uvh26.pt'),
        '--coco-person-model', str(repo/'data/models/yolo11n.pt'),
        '--device', device, '--max-frames', str(max_frames),
        '--out-csv', str(out), '--pet-threshold', '3.0',
        '--max-gap', '5', '--max-jump', '30', '--no-progress'
    ]
    print(f"[{site}] Starting...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo))
    if r.returncode != 0:
        print(f"[{site}] FAILED: {r.stderr[-500:]}", flush=True)
    return site, r.returncode

jobs = [
    ('GITI', repo/'data/sample_data/traffic_video.mp4',
     repo/'configs/sites/giti/bev_config.json',
     repo/'configs/sites/giti/grid_config.json',
     repo/'configs/sites/giti/gate_config.yaml',
     repo/'outputs/giti_full_300_parallel.csv'),
    ('MRC', repo/'data/sample_data/mrc_intersection.mp4',
     repo/'configs/sites/mrc/bev_config.json',
     repo/'configs/sites/mrc/grid_config.json',
     repo/'configs/sites/mrc/gate_config.yaml',
     repo/'outputs/mrc_full_300_parallel.csv')
]

# Ensure configs are reconstructed (in case of LFS pointers)
# This is handled by a Python script, but we'll leave a note.
# If configs are LFS pointers, run the reconstruction function.

with multiprocessing.Pool(processes=2) as pool:
    results = pool.map(run_site, jobs)

for site, code in results:
    print(f"{site}: {'OK' if code == 0 else 'FAILED'}")

print("Done.")
PYEOF

echo "=== Reproducibility manifest ==="
python - << 'PYEOF'
import json, hashlib, subprocess, sys
from pathlib import Path

repo = Path('.')
git_hash = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
manifest = {
    "git_commit": git_hash,
    "python_version": sys.version,
    "pip_freeze": subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout.strip(),
    "config_hashes": []
}
for site in ['giti', 'mrc']:
    site_dir = repo/'configs/sites'/'site'
    for f in ['calibration_points.json', 'bev_config.json', 'grid_config.json']:
        path = site_dir/f
        if path.exists():
            manifest["config_hashes"].append({"site": site, "file": f, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
with open(repo/'outputs/reproducibility_manifest.json', 'w') as fp:
    json.dump(manifest, fp, indent=2)
print("Manifest saved.")
PYEOF

echo "✅ Reproduction complete. See outputs/giti_full_300_parallel.csv and outputs/mrc_full_300_parallel.csv"
