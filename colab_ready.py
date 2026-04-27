# === NNDS Colab bootstrap (no HF token needed for public use) ===
import os
import sys
import subprocess
from pathlib import Path
from urllib.parse import quote

# ------------------------------------------------------------------
# CONFIG (can be overridden via environment variables)
# ------------------------------------------------------------------
# Optional GitHub PAT with access to Sharqascc/nnds (for private/ratelimited cases)
GITHUB_TOKEN = os.environ.get("NNDS_GH_TOKEN", "").strip()

# Branch to checkout; defaults to main to match README / Quick Start
BRANCH = os.environ.get("NNDS_BRANCH", "main")

REPO_OWNER = "Sharqascc"
REPO_NAME = "nnds"

# ------------------------------------------------------------------
# Helper to run shell commands with basic logging
# ------------------------------------------------------------------
def run(cmd):
    print(">>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def get_clone_url() -> str:
    base = f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git"
    if not GITHUB_TOKEN:
        # Public HTTPS clone (no authentication)
        return base
    token_enc = quote(GITHUB_TOKEN, safe="")
    return f"https://{token_enc}:x-oauth-basic@github.com/{REPO_OWNER}/{REPO_NAME}.git"


# ------------------------------------------------------------------
# 1) Clone or update repo on target branch
# ------------------------------------------------------------------
os.chdir("/content")
repo_dir = Path(REPO_NAME)

if repo_dir.exists():
    print(f"Repository {REPO_NAME} already exists, updating...")
    os.chdir(repo_dir)
    run(["git", "fetch"])
    run(["git", "checkout", BRANCH])
    run(["git", "pull"])
else:
    print(f"Cloning repository {REPO_OWNER}/{REPO_NAME} (branch: {BRANCH})...")
    clone_url = get_clone_url()
    run(["git", "clone", clone_url, REPO_NAME])
    os.chdir(repo_dir)
    run(["git", "checkout", BRANCH])

print("Repo:", os.getcwd())
run(["git", "status"])

# ------------------------------------------------------------------
# 2) Install Python dependencies
# ------------------------------------------------------------------
print("\nInstalling Python dependencies from requirements.txt ...")
run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# ------------------------------------------------------------------
# 3) Ensure videos/traffic_video.mp4 from HF dataset (public, no token)
# ------------------------------------------------------------------
os.makedirs("videos", exist_ok=True)
video_path = Path("videos/traffic_video.mp4")
if not video_path.exists():
    print("\nDownloading demo video from Hugging Face dataset...")
    video_url = (
        "https://huggingface.co/datasets/sharqascc/traffic-video-dataset/"
        "resolve/main/videos/traffic_video.mp4"
    )
    run(["wget", "-O", str(video_path), video_url])
else:
    print("\nVideo already present:", video_path)

# ------------------------------------------------------------------
# 4) Download SAM3 weights to sam3.pt from HF model repo (public, no token)
# ------------------------------------------------------------------
sam3_path = Path("sam3.pt")
if not sam3_path.exists():
    print("\nDownloading SAM3 checkpoint from Hugging Face model repo...")
    sam3_url = (
        "https://huggingface.co/sharqascc/sam3-traffic-model/"
        "resolve/main/sam3.pt"
    )
    run(["wget", "-O", str(sam3_path), sam3_url])
else:
    print("\nSAM3 checkpoint already present:", sam3_path)

# ------------------------------------------------------------------
# 5) Sanity prints and next steps
# ------------------------------------------------------------------
print("\n=== Ready ===")
print(
    "Video:",
    video_path.resolve(),
    "exists:",
    video_path.exists(),
    "size:",
    video_path.stat().st_size if video_path.exists() else None,
)
print(
    "SAM3:",
    sam3_path.resolve(),
    "exists:",
    sam3_path.exists(),
    "size:",
    sam3_path.stat().st_size if sam3_path.exists() else None,
)

print("\nTo run the main NNDS pipeline now (in a Colab cell):")
print("!cd /content/nnds && PYTHONPATH=. python traffic_analyzer.py --video videos/traffic_video.mp4")
print(
    "\nBy default, PET events will be written to: outputs/petevents_bev.csv "
    "(or as specified by --out-csv)."
)
