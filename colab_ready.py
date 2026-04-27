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
# Helpers
# ------------------------------------------------------------------
def run(cmd):
    """Run a shell command with basic logging."""
    print(">>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def run_with_progress(cmd, description="Downloading"):
    """Run wget with progress bar if available."""
    # For now, we just ensure --show-progress for wget; tqdm is overkill here.
    if cmd and cmd[0] == "wget" and "--show-progress" not in cmd:
        cmd.insert(1, "--show-progress")
    print(f">> {description}: {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True)


def download_with_retry(url, output_path, max_retries=3, description="Downloading"):
    """Download with retry logic for flaky connections."""
    for attempt in range(max_retries):
        try:
            run_with_progress(
                ["wget", "-O", str(output_path), url],
                description=description,
            )
            return True
        except subprocess.CalledProcessError as e:
            if attempt < max_retries - 1:
                print(f"Retry {attempt + 1}/{max_retries} after error: {e}")
                continue
            print("❌ Download failed after maximum retries.")
            raise
    return False


def get_clone_url() -> str:
    base = f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git"
    if not GITHUB_TOKEN:
        # Public HTTPS clone (no authentication)
        return base
    token_enc = quote(GITHUB_TOKEN, safe="")
    return f"https://{token_enc}:x-oauth-basic@github.com/{REPO_OWNER}/{REPO_NAME}.git"


def check_disk_space(required_gb=5):
    """Ensure enough disk space for models and video."""
    import shutil

    usage = shutil.disk_usage("/content")
    free_gb = usage.free / (1024**3)
    if free_gb < required_gb:
        print(
            f"⚠️ Low disk space: {free_gb:.1f} GB free, "
            f"{required_gb} GB recommended for video + models"
        )
    else:
        print(f"✅ Disk space: {free_gb:.1f} GB free")


def verify_imports():
    """Check that critical imports work."""
    critical_imports = ["cv2", "numpy", "pandas", "torch", "ultralytics", "yaml"]
    missing = []
    for module in critical_imports:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)

    if missing:
        print(f"⚠️ Missing imports: {missing}")
        return False
    print("✅ All critical imports verified")
    return True


def check_gpu():
    """Report PyTorch GPU availability."""
    print("\n🔍 Checking PyTorch GPU availability...")
    try:
        import torch

        if torch.cuda.is_available():
            print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print("⚠️ No GPU detected. Using CPU (slower for large videos).")
    except ImportError:
        print("⚠️ PyTorch not yet importable (check installation above).")


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
# 2) Check disk space before heavy downloads
# ------------------------------------------------------------------
print("\nChecking disk space...")
check_disk_space(required_gb=5)

# ------------------------------------------------------------------
# 3) Install Python dependencies
# ------------------------------------------------------------------
print("\nInstalling Python dependencies from requirements.txt ...")
run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Verify critical imports
print("\nVerifying critical imports...")
verify_imports()

# Check GPU availability for PyTorch
check_gpu()

# ------------------------------------------------------------------
# 4) Ensure videos/traffic_video.mp4 from HF dataset (public, no token)
# ------------------------------------------------------------------
os.makedirs("videos", exist_ok=True)
video_path = Path("videos/traffic_video.mp4")
if not video_path.exists():
    print("\nDownloading demo video from Hugging Face dataset...")
    video_url = (
        "https://huggingface.co/datasets/sharqascc/traffic-video-dataset/"
        "resolve/main/videos/traffic_video.mp4"
    )
    download_with_retry(
        video_url,
        video_path,
        max_retries=3,
        description="Downloading demo video",
    )
else:
    print("\nVideo already present:", video_path)

# ------------------------------------------------------------------
# 5) Download SAM3 weights to sam3.pt from HF model repo (public, no token)
# ------------------------------------------------------------------
sam3_path = Path("sam3.pt")
if not sam3_path.exists():
    print("\nDownloading SAM3 checkpoint from Hugging Face model repo...")
    sam3_url = (
        "https://huggingface.co/sharqascc/sam3-traffic-model/"
        "resolve/main/sam3.pt"
    )
    download_with_retry(
        sam3_url,
        sam3_path,
        max_retries=3,
        description="Downloading SAM3 checkpoint",
    )
else:
    print("\nSAM3 checkpoint already present:", sam3_path)

# ------------------------------------------------------------------
# 6) Sanity prints and next steps
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
print(
    "!cd /content/nnds && PYTHONPATH=. python traffic_analyzer.py "
    "--video videos/traffic_video.mp4"
)
print(
    "\nBy default, PET events will be written to: outputs/petevents_bev.csv "
    "(or as specified by --out-csv)."
)
