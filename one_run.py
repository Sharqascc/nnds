#!/usr/bin/env python3
"""
one_run.py

End-to-end NNDS run for Colab / local:

- Stage 0: clone + install + download video + SAM3 (if needed)
- Stage 1: Video -> PET (if SAM3VideoSemanticPredictor available)
            otherwise, fall back to a PET CSV (demo or user-provided)
- Stage 2: PET summary & risk analysis
- Stage 3: Diffusion training
- Stage 4: Diffusion safety evaluation
"""

import os
import sys
import subprocess
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
GITHUB_TOKEN = os.environ.get("NNDS_GH_TOKEN", "").strip()  # optional; for cloning, if needed

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

MAX_FRAMES = 100
PET_THRESHOLD = 2.0
CRITICAL_PET = 1.0
MODERATE_PET = 3.0
DIFFUSION_EPOCHS = 100

# Fallback PET CSV (demo) if video->PET fails or SAM3 is unavailable
DEMO_PET_CSV = "docs/data_samples/petevents_bev_demo.csv"


def run(cmd, cwd=None, check=True):
    print(">>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=cwd, check=check)


def ensure_repo():
    """Ensure we're in an NNDS repo; clone only in Colab when needed."""
    cwd = Path.cwd()

    # Case 1: already inside a repo checkout
    if (cwd / "traffic_analyzer.py").exists() and (cwd / "pyproject.toml").exists():
        print("[INFO] Already in NNDS repository:", cwd)
        return

    # Case 2: Colab root, clone into /content/nnds
    if is_colab():
        os.chdir("/content")
        repo_dir = Path("nnds")
        if not repo_dir.exists():
            if GITHUB_TOKEN:
                repo_url = f"https://{GITHUB_TOKEN}:x-oauth-basic@github.com/{REPO_OWNER}/{REPO_NAME}.git"
            else:
                repo_url = f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git"
            print(f"[INFO] Cloning {repo_url} ...")
            run(["git", "clone", repo_url, "nnds"])
        os.chdir(repo_dir)
        run(["git", "checkout", BRANCH])
        print("[INFO] Repo at:", os.getcwd())
        run(["git", "status"])
        return

    # Case 3: local, not in repo and not Colab
    raise RuntimeError(
        "Not in NNDS repo and not in Colab. "
        "Please clone the repo first.\n"
        "  git clone https://github.com/{REPO_OWNER}/{REPO_NAME}.git\n"
        "then run one_run.py from inside that directory."
    )


def ensure_deps_and_data():
    """Install requirements and ensure demo video + SAM3 exist."""
    print("\n[INFO] Installing Python dependencies from requirements.txt ...")
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

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
    return video_path, sam3_path


def check_sam3_available():
    """Return True if SAM3VideoSemanticPredictor is importable, else False."""
    print("\n[INFO] Checking Ultralytics + SAM3 availability ...")
    try:
        import ultralytics  # noqa: F401
        print("Ultralytics version:", ultralytics.__version__)
        from ultralytics.models.sam import SAM3VideoSemanticPredictor  # noqa: F401
        print("[INFO] SAM3VideoSemanticPredictor import OK")
        return True
    except Exception as e:
        print("[WARN] SAM3VideoSemanticPredictor not available:", e)
        print("      Will fall back to PET CSV only (no video->PET in this run).")
        return False


def stage1_video_to_pet(video_path: Path, sam3_path: Path) -> Path:
    """Run video -> PET via traffic_analyzer.py if SAM3 is available."""
    os.makedirs("outputs", exist_ok=True)
    out_csv = Path(
        f"outputs/petevents_bev_traffic_video_{MAX_FRAMES}f_pet{str(PET_THRESHOLD).replace('.', 'p')}.csv"
    )

    print("\n[INFO] === Stage 1: Video -> PET (max-frames = {}) ===".format(MAX_FRAMES))
    print("[INFO] Output PET CSV will be:", out_csv)

    traffic_cmd = [
        sys.executable,
        "traffic_analyzer.py",
        "--video",
        str(video_path),
        "--sam3-weights",
        str(sam3_path),
        "--out-csv",
        str(out_csv),
        "--pet-threshold",
        str(PET_THRESHOLD),
        "--max-frames",
        str(MAX_FRAMES),
    ]

    try:
        run(traffic_cmd)
        if not out_csv.exists():
            raise FileNotFoundError(out_csv)
        print("[INFO] Stage 1 completed, PET CSV:", out_csv)
        return out_csv
    except Exception as e:
        print("[ERROR] Stage 1 (video->PET) failed:", e)
        print("[WARN] Falling back to demo PET CSV:", DEMO_PET_CSV)
        demo_path = Path(DEMO_PET_CSV)
        if not demo_path.exists():
            raise FileNotFoundError(
                f"Demo PET CSV not found at {demo_path}. "
                "Ensure docs/data_samples/petevents_bev_demo.csv exists."
            )
        return demo_path


def stage2_pet_summary(pet_csv: Path):
    print("\n[INFO] === Stage 2: PET summary & risk analysis ===")
    analysis_out_dir = Path("analysis_results_100")
    analysis_out_dir.mkdir(exist_ok=True)

    pet_summary_cmd = [
        sys.executable,
        "analysis/pet_summary.py",
        "--csv-path",
        str(pet_csv),
        "--critical",
        str(CRITICAL_PET),
        "--moderate",
        str(MODERATE_PET),
        "--export",
        "--output-dir",
        str(analysis_out_dir),
    ]
    run(pet_summary_cmd)
    print("[INFO] PET summary outputs written to:", analysis_out_dir)
    return analysis_out_dir


def stage3_diffusion_train(pet_csv: Path):
    print("\n[INFO] === Stage 3: Diffusion training ===")
    diff_train_cmd = [
        sys.executable,
        "traffic_diffusion/train_trajectory_diffusion.py",
        "--csv-path",
        str(pet_csv),
        "--epochs",
        str(DIFFUSION_EPOCHS),
    ]
    run(diff_train_cmd)


def stage4_diffusion_eval():
    print("\n[INFO] === Stage 4: Diffusion safety evaluation ===")
    diff_eval_cmd = [
        sys.executable,
        "analysis/safety_eval_diffusion.py",
    ]
    run(diff_eval_cmd)


def main():
    ensure_repo()
    video_path, sam3_path = ensure_deps_and_data()
    sam3_ok = check_sam3_available()

    if sam3_ok:
        pet_csv = stage1_video_to_pet(video_path, sam3_path)
    else:
        # Skip video extraction; use demo PET CSV directly
        pet_csv = Path(DEMO_PET_CSV)
        print("\n[INFO] Skipping Stage 1. Using PET CSV:", pet_csv)

    stage2_pet_summary(pet_csv)
    stage3_diffusion_train(pet_csv)
    stage4_diffusion_eval()

    print("\n✅ All stages completed.")
    print("   PET CSV used:", pet_csv)
    print("   PET summary dir: analysis_results_100")
    print("   Diffusion evaluation CSVs are in: outputs/")


if __name__ == "__main__":
    main()
