#!/usr/bin/env bash
# =====================================================================
# NNDS Pipeline Reproduction Script
# =====================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MAX_FRAMES=300
DEVICE=cpu
GITI_ONLY=0
PREFLIGHT_ONLY=0
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-frames)     MAX_FRAMES="$2"; shift 2 ;;
        --device)         DEVICE="$2"; shift 2 ;;
        --giti-only)      GITI_ONLY=1; shift ;;
        --preflight-only) PREFLIGHT_ONLY=1; shift ;;
        -h|--help)
            sed -n '3,15p' "${BASH_SOURCE[0]}"
            exit 0 ;;
        --*)
            echo "Unknown option: $1" >&2; exit 2 ;;
        *)
            POSITIONAL+=("$1"); shift ;;
    esac
done
if (( ${#POSITIONAL[@]} >= 1 )); then MAX_FRAMES="${POSITIONAL[0]}"; fi
if (( ${#POSITIONAL[@]} >= 2 )); then DEVICE="${POSITIONAL[1]}"; fi

export MAX_FRAMES DEVICE GITI_ONLY

VIDEO_DIR="${NNDS_VIDEO_DIR:-$ROOT/data/sample_data}"
GITI_VIDEO="$VIDEO_DIR/GITI_traffic_video.mp4"
MRC_VIDEO="$VIDEO_DIR/MRC_traffic_video.mp4"
MIN_VIDEO_BYTES=1048576

check_video() {
    local path="$1"
    if [[ ! -f "$path" ]]; then
        echo "ERROR: missing $path" >&2
        echo "       run: git lfs pull" >&2
        return 1
    fi
    local size
    size=$(stat -c%s "$path" 2>/dev/null || stat -f%z "$path")
    if (( size < MIN_VIDEO_BYTES )); then
        echo "ERROR: $path is ${size} bytes — looks like an LFS pointer, not a video" >&2
        echo "       run: git lfs pull" >&2
        return 1
    fi
    return 0
}

echo "=== [0/4] Preflight: verifying input videos ==="
PREFLIGHT_FAILED=0
check_video "$GITI_VIDEO" || PREFLIGHT_FAILED=1
if (( GITI_ONLY == 0 )); then
    check_video "$MRC_VIDEO" || PREFLIGHT_FAILED=1
else
    echo "NOTE: --giti-only set; skipping MRC preflight"
fi
if (( PREFLIGHT_FAILED )); then
    echo "Preflight failed. Aborting." >&2
    exit 1
fi
if (( PREFLIGHT_ONLY )); then
    echo "Preflight OK."
    exit 0
fi

echo "=== [1/4] Installing dependencies ==="
pip install -q ultralytics pandas matplotlib opencv-python pyyaml scipy

echo "=== [2/4] Downloading models ==="
bash scripts/download_models.sh

echo "=== [3/4] Running pipeline(s) ==="
mkdir -p outputs

python - << 'PYEOF'
import os, sys, subprocess, multiprocessing
from pathlib import Path

repo = Path('.')
max_frames = int(os.environ.get('MAX_FRAMES', '300'))
device = os.environ.get('DEVICE', 'cpu')
giti_only = os.environ.get('GITI_ONLY', '0') == '1'

def run_site(args):
    site, video, bev, grid, gate, out = args
    if not Path(video).exists():
        print(f"[{site}] SKIPPED: video missing ({video})", flush=True)
        return site, 127, out
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
        print(f"[{site}] FAILED (rc={r.returncode}): {r.stderr[-500:]}", flush=True)
    return site, r.returncode, out

jobs = [
    ('GITI', repo/'data/sample_data/GITI_traffic_video.mp4',
     repo/'configs/sites/giti/bev_config.json',
     repo/'configs/sites/giti/grid_config.json',
     repo/'configs/sites/giti/gate_config.yaml',
     repo/'outputs/giti_full_300_parallel.csv'),
]
if not giti_only:
    jobs.append(
        ('MRC', repo/'data/sample_data/MRC_traffic_video.mp4',
         repo/'configs/sites/mrc/bev_config.json',
         repo/'configs/sites/mrc/grid_config.json',
         repo/'configs/sites/mrc/gate_config.yaml',
         repo/'outputs/mrc_full_300_parallel.csv')
    )

with multiprocessing.Pool(processes=len(jobs)) as pool:
    results = pool.map(run_site, jobs)

failed = []
missing = []
for site, code, out in results:
    if code != 0:
        failed.append(site)
    if not Path(out).exists():
        missing.append(f"{site} ({out})")
    print(f"{site}: {'OK' if code == 0 else 'FAILED'}", flush=True)

if failed:
    print(f"\nERROR: sub-pipelines failed: {failed}", file=sys.stderr, flush=True)
    sys.exit(1)
if missing:
    print(f"\nERROR: expected outputs missing: {missing}", file=sys.stderr, flush=True)
    sys.exit(1)

print("All requested pipelines succeeded.", flush=True)
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
    site_dir = repo/'configs/sites'/site
    for f in ['calibration_points.json', 'bev_config.json', 'grid_config.json']:
        path = site_dir/f
        if path.exists():
            manifest["config_hashes"].append({"site": site, "file": f, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
with open(repo/'outputs/reproducibility_manifest.json', 'w') as fp:
    json.dump(manifest, fp, indent=2)
print("Manifest saved.", flush=True)
PYEOF

echo "✅ Reproduction complete."
for out in outputs/giti_full_300_parallel.csv outputs/mrc_full_300_parallel.csv; do
    if [[ -f "$out" ]]; then
        echo "   - $out"
    fi
done
