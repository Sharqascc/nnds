# =====================================================================
# NNDS: FULL PIPELINE IN ONE CELL (Colab, with token placeholder)
# =====================================================================

import os
import sys
import subprocess
from pathlib import Path

# -----------------------------
# 0. CONFIG
# -----------------------------
# Set GitHub token if needed (for private / rate-limited access); else leave ""
GITHUB_TOKEN = "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"  # <-- put your token here, or "" if repo is public

REPO_OWNER = "Sharqascc"
REPO_NAME = "nnds"
BRANCH = "main"

VIDEO_URL = (
    "https://huggingface.co/datasets/sharqascc/traffic-video-dataset/"
    "resolve/main/videos/traffic_video.mp4"
)
SAM3_URL = (
    "https://huggingface.co/sharqascc/sam3-traffic-model/"
    "resolve/main/sam3.pt"
)

MAX_FRAMES = 100          # number of frames for quick run
PET_THRESHOLD = 2.0       # PET threshold in seconds
CRITICAL_PET = 1.0        # critical PET threshold for summary
MODERATE_PET = 3.0        # moderate PET threshold for summary
DIFFUSION_EPOCHS = 100    # diffusion training epochs

# -----------------------------
# Helper to run shell commands
# -----------------------------
def run(cmd, cwd=None, check=True):
    print(">>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=cwd, check=check)

# -----------------------------
# 1. Clone NNDS repo
# -----------------------------
os.chdir("/content")
repo_dir = Path("nnds")

if repo_dir.exists():
    print("Removing existing /content/nnds ...")
    run(["rm", "-rf", "nnds"])

if GITHUB_TOKEN:
    repo_url = f"https://{GITHUB_TOKEN}:x-oauth-basic@github.com/{REPO_OWNER}/{REPO_NAME}.git"
else:
    repo_url = f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git"

print(f"Cloning {repo_url} ...")
run(["git", "clone", repo_url, "nnds"])
os.chdir(repo_dir)
run(["git", "checkout", BRANCH])

print("\n[INFO] Repo cloned at:", os.getcwd())
run(["git", "status"])

# -----------------------------
# 2. Install Python dependencies
# -----------------------------
print("\n[INFO] Installing Python dependencies from requirements.txt ...")
run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# -----------------------------
# 3. Ensure demo video + SAM3
# -----------------------------
print("\n[INFO] Ensuring demo video and SAM3 weights exist ...")
os.makedirs("videos", exist_ok=True)
video_path = Path("videos/traffic_video.mp4")
sam3_path = Path("sam3.pt")

if not video_path.exists():
    print("[INFO] Downloading demo video ...")
    run(["wget", "-O", str(video_path), VIDEO_URL])
else:
    print("[INFO] Video already present:", video_path)

if not sam3_path.exists():
    print("[INFO] Downloading SAM3 checkpoint ...")
    run(["wget", "-O", str(sam3_path), SAM3_URL])
else:
    print("[INFO] SAM3 checkpoint already present:", sam3_path)

print("[INFO] Video exists:", video_path.exists())
print("[INFO] SAM3 exists:", sam3_path.exists())

# -----------------------------
# 4. Verify Ultralytics + SAM3
# -----------------------------
print("\n[INFO] Checking Ultralytics + SAM3 availability ...")
import ultralytics
print("Ultralytics version:", ultralytics.__version__)

try:
    from ultralytics.models.sam import SAM3VideoSemanticPredictor
    print("[INFO] SAM3VideoSemanticPredictor import OK")
except Exception as e:
    print("[WARN] Could not import SAM3VideoSemanticPredictor:", e)
    print("       Downstream video->PET may fail if SAM3 is not available.")

# -----------------------------
# 5. Stage 1: Video -> PET (100 frames)
# -----------------------------
print("\n[INFO] === Stage 1: Video -> PET (max-frames = {}) ===".format(MAX_FRAMES))
out_csv = f"outputs/petevents_bev_traffic_video_{MAX_FRAMES}f_pet{str(PET_THRESHOLD).replace('.', 'p')}.csv"
os.makedirs("outputs", exist_ok=True)

traffic_cmd = [
    sys.executable,
    "traffic_analyzer.py",
    "--video", str(video_path),
    "--sam3-weights", str(sam3_path),
    "--out-csv", out_csv,
    "--pet-threshold", str(PET_THRESHOLD),
    "--max-frames", str(MAX_FRAMES),
]

print("[INFO] Output PET CSV will be:", out_csv)
run(traffic_cmd)

print("\n[INFO] Checking PET CSV exists ...")
print("PET CSV exists:", Path(out_csv).exists())

# -----------------------------
# 6. Stage 2: PET summary & risk analysis
# -----------------------------
print("\n[INFO] === Stage 2: PET summary & risk analysis ===")
analysis_out_dir = Path("analysis_results_100")
analysis_out_dir.mkdir(exist_ok=True)

pet_summary_cmd = [
    sys.executable,
    "analysis/pet_summary.py",
    "--csv-path", out_csv,
    "--critical", str(CRITICAL_PET),
    "--moderate", str(MODERATE_PET),
    "--export",
    "--output-dir", str(analysis_out_dir),
]
run(pet_summary_cmd)

print("\n[INFO] PET summary outputs written to:", analysis_out_dir)

# -----------------------------
# 7. Stage 3: Diffusion training
# -----------------------------
print("\n[INFO] === Stage 3: Diffusion training ===")
diff_train_cmd = [
    sys.executable,
    "traffic_diffusion/train_trajectory_diffusion.py",
    "--csv-path", out_csv,
    "--epochs", str(DIFFUSION_EPOCHS),
]
run(diff_train_cmd)

# -----------------------------
# 8. Stage 4: Diffusion safety evaluation
# -----------------------------
print("\n[INFO] === Stage 4: Diffusion safety evaluation ===")
diff_eval_cmd = [
    sys.executable,
    "analysis/safety_eval_diffusion.py",
]
run(diff_eval_cmd)

print("\n✅ All stages completed.")
print("   PET CSV:", out_csv)
print("   PET summary dir:", analysis_out_dir)
print("   Diffusion evaluation CSVs in: outputs/")
