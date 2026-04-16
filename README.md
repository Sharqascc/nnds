# NNDS: Non-motorized and Heterogeneous Traffic Safety Analysis

AI-powered system for analyzing vehicle behavior and surrogate safety metrics at unsignalized intersections.

> Note: The versioned PET sample and Colab bootstrap described below are introduced on the `feat/video-to-pet-pipeline` branch and will appear on `main` after this branch is merged.

---

## Quickstart

Use the repository with the Make targets or with direct Python entry points.

```bash
git clone https://github.com/Sharqascc/nnds.git
cd nnds
make install
```

### Run video to PET pipeline

```bash
make grid
# or
PYTHONPATH=. python traffic_analyzer.py --video videos/traffic_video.mp4
```

This runs the end-to-end grid/PET pipeline on a video, computes PET events, and writes them to `outputs/petevents_bev.csv` by default.

A canonical 30-frame PET events sample, derived from the HF demo video, is versioned at:

- `docs/data_samples/petevents_bev_demo.csv`

You can use this sample directly for quick diffusion experiments without recomputing PETs. [page:57]

### Train diffusion model

```bash
make diffusion-train
# or
PYTHONPATH=. python traffic_diffusion/train_trajectory_diffusion.py
```

By default, the diffusion training script expects PET events in `outputs/petevents_bev.csv`; you may also point it to the canonical sample:

```bash
PYTHONPATH=. python traffic_diffusion/train_trajectory_diffusion.py \
  --csv-path docs/data_samples/petevents_bev_demo.csv
```

### Evaluate diffusion safety

```bash
make diffusion-eval
# or
PYTHONPATH=. python analysis/safety_eval_diffusion.py
```

### Run notebook-style end-to-end diffusion evaluation

```bash
make diffusion-notebook
# or
PYTHONPATH=. python analysis/safety_eval_diffusion_notebook.py
```

---

## Colab setup (recommended)

For Google Colab, use a single bootstrap script to prepare the environment. This:

- clones or updates the `nnds` repository
- switches to the desired branch
- installs Python dependencies
- downloads the default demo video into `videos/traffic_video.mp4`
- downloads `sam3.pt` into the repo root only if it is missing

Both the demo video and SAM3 weights are hosted publicly on Hugging Face; no HF token is required. [page:32]

### One-cell Colab bootstrap

Paste this into a fresh Colab cell:

```python
# NNDS Colab bootstrap: clone, install, download demo video + SAM3

import os
import sys
import subprocess
from pathlib import Path
from urllib.parse import quote

# 0) CONFIG – EDIT THIS ONLY IF YOU NEED A PRIVATE FORK
GITHUB_TOKEN = "YOUR_GITHUB_PAT_HERE"  # GitHub PAT with access to Sharqascc/nnds

assert GITHUB_TOKEN != "YOUR_GITHUB_PAT_HERE", "Set GITHUB_TOKEN"

# 1) Clone or update repo (main branch or your preferred branch)
os.chdir("/content")
repo_dir = Path("nnds")

if repo_dir.exists():
    os.chdir(repo_dir)
    subprocess.run(["git", "fetch"], check=True)
    # Switch to main by default; change to your feature branch if needed
    subprocess.run(["git", "checkout", "main"], check=True)
    subprocess.run(["git", "pull"], check=True)
else:
    token_enc = quote(GITHUB_TOKEN, safe="")
    clone_url = f"https://{token_enc}:x-oauth-basic@github.com/Sharqascc/nnds.git"
    subprocess.run(["git", "clone", clone_url, "nnds"], check=True)
    os.chdir(repo_dir)
    subprocess.run(["git", "checkout", "main"], check=True)

print("Repo:", os.getcwd())
subprocess.run(["git", "status"], check=True)

# 2) Install Python dependencies
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
    check=True,
)

# 3) Ensure demo video at videos/traffic_video.mp4 (public HF dataset)
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

# 4) Ensure SAM3 weights at sam3.pt (public HF model)
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
if video_path.exists():
    print("Video:", video_path.resolve(), "size:", video_path.stat().st_size)
if sam3_path.exists():
    print("SAM3:", sam3_path.resolve(), "size:", sam3_path.stat().st_size)

print("\nTo run the pipeline now:")
print("!cd /content/nnds && PYTHONPATH=. python traffic_analyzer.py "
      "--video videos/traffic_video.mp4")
```

Then run the pipeline:

```bash
!cd /content/nnds && PYTHONPATH=. python traffic_analyzer.py --video videos/traffic_video.mp4
```

---

## Project Structure

- `grid_trajectory/` – Spatial grid mapping and PET computation
- `analysis/` – Evaluation and research scripts
- `calibration/` – Camera calibration files
- `configs/` – Grid and calibration configurations
- `outputs/` – Generated experiment artifacts and evaluation CSV files (ignored by git)
- `docs/data_samples/` – Versioned sample artifacts (e.g., `petevents_bev_demo.csv`)
- `traffic_diffusion/` – Diffusion-based trajectory and safety modules
- `bev_mapper.py` – Bird's Eye View transformation
- `giti_bev_calib.py` – Homography calibration
- `traffic_analyzer.py` – Traffic analysis and conflict processing pipeline
- `traj_diffusion_normalized.py` – Normalized diffusion experiment script [page:32][page:57]

---

## Entry points

### Grid / PET extraction

- `traffic_analyzer.py` – End-to-end traffic analysis on video, including detection, BEV transformation, grid construction, conflict extraction, and PET computation.
- `grid_trajectory/` – Core grid and trajectory logic used by `traffic_analyzer.py`. [page:32]

### Diffusion training

- `traffic_diffusion/train_trajectory_diffusion.py` – Trains the conditional trajectory diffusion model on PET events from `outputs/petevents_bev.csv` or a sample file like `docs/data_samples/petevents_bev_demo.csv`. [page:32][page:57]
- `traffic_diffusion/training_utils.py` – Reusable helpers for data cleaning, loader creation, and training loops.

### Diffusion safety evaluation

- `analysis/safety_eval_diffusion.py` – Batch PET/TTC evaluation using a saved diffusion checkpoint and writes `outputs/safety_eval_diffusion.csv`.
- `analysis/safety_eval_diffusion_notebook.py` – Notebook-friendly variant that retrains, samples futures, and produces:
  - `outputs/safety_events_diffusion_model.csv`
  - `outputs/safety_eval_diffusion_summary.csv`

---

## Key Features

- SAM3 video segmentation
- Spatial grid zone analysis
- Bird's Eye View world-coordinate mapping
- PET computation
- Conflict detection
- Diffusion-based trajectory modeling
- PET and TTC safety evaluation [page:32]

---

## Development setup

Recommended environment:

- Python 3.10+
- Google Colab for experiments, or a local Python environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

Typical developer workflow:

```bash
git clone https://github.com/Sharqascc/nnds.git
cd nnds
pip install -r requirements.txt
python -m pytest
```

---

## Repository conventions

- `grid_trajectory/` is the canonical location for grid and PET logic.
- `traffic_analyzer.py` is the main end-to-end entry point for video-to-PET processing.
- `analysis/` contains evaluation-oriented scripts.
- `traffic_diffusion/` contains reusable model, sampling, and safety modules.
- `outputs/` stores generated experiment artifacts and evaluation CSV files and is ignored by git.
- `docs/data_samples/` contains small, versioned sample artifacts (like `petevents_bev_demo.csv`) for reproducible experiments. [page:57]

For new research work:

1. Prefer reusable logic inside `traffic_diffusion/` or `grid_trajectory/`.
2. Keep one-off experiment runners inside `analysis/`.
3. Write generated artifacts into `outputs/` with stable, descriptive filenames.

---

## Usage

Code and configs are maintained on GitHub, while the default public demo video and SAM3 weights are hosted on Hugging Face. [page:32]

### Default demo video (Hugging Face)

The canonical example video for `traffic_analyzer.py` lives in a dataset repo:

- Dataset: <https://huggingface.co/datasets/sharqascc/traffic-video-dataset>
- Video file (web view):  
  <https://huggingface.co/datasets/sharqascc/traffic-video-dataset/blob/main/videos/traffic_video.mp4>

For scripts and Colab, use the `resolve` URL so the file is downloaded directly:

```bash
cd /content/nnds
mkdir -p videos
wget -O videos/traffic_video.mp4 \
  "https://huggingface.co/datasets/sharqascc/traffic-video-dataset/resolve/main/videos/traffic_video.mp4"
PYTHONPATH=. python traffic_analyzer.py --video videos/traffic_video.mp4
```

Larger private or experimental videos can still be stored on Google Drive, but all public examples in this repo are expected to work with the Hugging Face–hosted demo video by default. [page:32]

### SAM3 model weights

The SAM3 video segmentation model used by `traffic_analyzer.py` is stored in a Hugging Face model repo and is not committed to this repo. [page:32]

For Colab users, the Colab bootstrap script above downloads `sam3.pt` automatically into the repository root if it is missing.

If you want to download it manually:

```bash
cd /content/nnds
wget -O sam3.pt \
  "https://huggingface.co/sharqascc/sam3-traffic-model/resolve/main/sam3.pt"
```

Verify:

```bash
ls -lh sam3.pt
```

Then run:

```bash
PYTHONPATH=. python traffic_analyzer.py --video videos/traffic_video.mp4
```

---

## Diffusion-based Safety Evaluation

This repository includes a trajectory diffusion model and PET/TTC-based safety evaluation on PET events extracted from the grid pipeline. [page:32]

### Components

- `traffic_diffusion/train_trajectory_diffusion.py` – Trains a conditional trajectory diffusion model on PET-event futures using `outputs/petevents_bev.csv` or a sample file such as `docs/data_samples/petevents_bev_demo.csv`. [page:32][page:57]
- `traffic_diffusion/model_and_sampler.py` – Loads diffusion checkpoints and samples counterfactual futures given past trajectories.
- `analysis/safety_eval_diffusion.py` – Iterates over PET events, samples multiple futures per event, and computes PET and TTC statistics.

### Running the original safety evaluation

```bash
cd /content
git clone https://github.com/Sharqascc/nnds.git
cd nnds
PYTHONPATH=. python analysis/safety_eval_diffusion.py
```

### Notebook-friendly diffusion pipeline

For iterative experiments and Colab runs, this repository also includes a notebook-style pipeline:

- `traffic_diffusion/training_utils.py` – Data cleaning, normalization, loader creation, and reusable training helpers
- `traffic_diffusion/sampling_utils.py` – Utilities to load a trained checkpoint and sample counterfactual futures
- `analysis/safety_eval_diffusion_notebook.py` – End-to-end notebook-oriented script that:
  - builds cleaned train and eval loaders
  - trains the diffusion model and saves a checkpoint
  - samples future trajectories for evaluation events
  - constructs an event-level PET and risk table
  - summarizes safety using `traffic_diffusion/pet_safety_metrics.py`

Run it in Colab with:

```bash
cd /content
git clone https://github.com/Sharqascc/nnds.git
cd nnds
PYTHONPATH=. python analysis/safety_eval_diffusion_notebook.py
```

---

## License

MIT
