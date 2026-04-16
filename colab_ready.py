# === NNDS Colab bootstrap (no HF token needed) ===
import os, sys, subprocess
from pathlib import Path
from urllib.parse import quote

# 0) CONFIG – EDIT THIS ONLY
GITHUB_TOKEN = "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"        # GitHub PAT with access to Sharqascc/nnds
assert GITHUB_TOKEN != "YOUR_GITHUB_PAT_HERE", "Set GITHUB_TOKEN"

# 1) Clone or update repo on feat/video-to-pet-pipeline
os.chdir("/content")
repo_dir = Path("nnds")

if repo_dir.exists():
    os.chdir(repo_dir)
    subprocess.run(["git", "fetch"], check=True)
    subprocess.run(["git", "checkout", "feat/video-to-pet-pipeline"], check=True)
    subprocess.run(["git", "pull"], check=True)
else:
    token_enc = quote(GITHUB_TOKEN, safe="")
    clone_url = f"https://{token_enc}:x-oauth-basic@github.com/Sharqascc/nnds.git"
    subprocess.run(["git", "clone", clone_url, "nnds"], check=True)
    os.chdir(repo_dir)
    subprocess.run(["git", "checkout", "feat/video-to-pet-pipeline"], check=True)

print("Repo:", os.getcwd())
subprocess.run(["git", "status"], check=True)

# 2) Install Python dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)

# 3) Ensure videos/traffic_video.mp4 from HF dataset (public, no token)
os.makedirs("videos", exist_ok=True)
video_path = Path("videos/traffic_video.mp4")
if not video_path.exists():
    print("Downloading demo video from Hugging Face dataset...")
    video_url = (
        "https://huggingface.co/datasets/sharqascc/traffic-video-dataset/"
        "resolve/main/videos/traffic_video.mp4"
    )
    subprocess.run(
        ["wget", "-O", str(video_path), video_url],
        check=True,
    )
else:
    print("Video already present:", video_path)

# 4) Download SAM3 weights to sam3.pt from HF model repo (public, no token)
sam3_path = Path("sam3.pt")
if not sam3_path.exists():
    print("Downloading SAM3 checkpoint from Hugging Face model repo...")
    sam3_url = (
        "https://huggingface.co/sharqascc/sam3-traffic-model/"
        "resolve/main/sam3.pt"
    )
    subprocess.run(
        ["wget", "-O", str(sam3_path), sam3_url],
        check=True,
    )
else:
    print("SAM3 checkpoint already present:", sam3_path)

# 5) Sanity prints
print("\n=== Ready ===")
print("Video:", video_path.resolve(), "exists:", video_path.exists(), "size:", video_path.stat().st_size if video_path.exists() else None)
print("SAM3:", sam3_path.resolve(), "exists:", sam3_path.exists(), "size:", sam3_path.stat().st_size if sam3_path.exists() else None)

print("\nTo run the pipeline now:")
print("!cd /content/nnds && PYTHONPATH=. python traffic_analyzer.py --video videos/traffic_video.mp4")
