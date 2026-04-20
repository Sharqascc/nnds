NNDS: Non-motorized and Heterogeneous Traffic Safety Analysis
AI-powered system for analyzing vehicle behavior and surrogate safety metrics at unsignalized intersections.

Note: The --max-frames speed/debug option and the Colab bootstrap script are now available on main and support the video→PET pipeline used in the diffusion experiments.



## Pipeline Architecture

The NNDS system implements a multi-stage pipeline that transforms raw intersection video into quantitative safety metrics and diffusion-based trajectory predictions. The system is designed for research reproducibility, with each phase producing intermediate artifacts that can be independently validated.

### Pipeline Overview

| Phase | Stage | Key Output | File |
|-------|-------|------------|------|
| 1 | Video Input & Preprocessing | Raw frames | `videos/` |
| 2 | SAM3 Video Segmentation | Actor masks + track IDs | `traffic_analyzer.py` |
| 3 | BEV Transformation | World-coordinate trajectories | `giti_bev_calib.py` |
| 4 | Grid Mapping & Trajectory | Grid cell assignments | `grid_trajectory/` |
| 5 | PET Conflict Extraction | Conflict events + PETs | `outputs/petevents_bev.csv` |
| 6 | Analysis & Visualization | Safety statistics + plots | `analysis/` |
| 7 | Diffusion-Based Modeling | Counterfactual trajectories | `traffic_diffusion/` |

### Research Questions Addressed

1. **Detection**: Can SAM3 segment heterogeneous actors (vehicles, NMTs) in complex intersection scenes?
2. **Calibration**: How accurately does homography transform image-plane detections to world coordinates?
3. **Trajectory Construction**: Do grid-based trajectory representations capture meaningful conflict patterns?
4. **Safety Metrics**: What PET/TTC distributions emerge from real intersection data?
5. **Counterfactual Modeling**: Can diffusion models generate plausible alternative trajectories for safety evaluation?

---

### Phase 1: Video Input & Preprocessing

| Component | File | Description |
|-----------|------|-------------|
| Input video | `videos/` | Raw traffic video (e.g., `traffic_video.mp4`) |
| Demo video | HF dataset | Public demo hosted at `sharqascc/traffic-video-dataset` |
| Frame limiting | `--max-frames` | Debug/fast-test mode for rapid iteration |

### Phase 2: SAM3 Video Segmentation

| Component | File | Description |
|-----------|------|-------------|
| SAM3 weights | `sam3.pt` | Downloaded from HF model repo `sharqascc/sam3-traffic-model` |
| Segmentation | `traffic_analyzer.py` | SAM3 segments actors (vehicles, NMTs) per frame |
| Actor tracking | -- | Track IDs assigned to segmented actors across frames |

### Phase 3: BEV Transformation & Calibration

| Component | File | Description |
|-----------|------|-------------|
| Homography calibration | `giti_bev_calib.py` | Computes camera-to-world homography matrix |
| BEV mapper | `bev_mapper.py` | Transforms image-plane detections to world/BEV coordinates |
| World coordinates | -- | Outputs (t, x, y) trajectories in meters |

### Phase 4: Grid Mapping & Trajectory Construction

| Component | File | Description |
|-----------|------|-------------|
| Spatial grid | `grid_trajectory/spatial_grid.py` | Defines intersection grid zones (CELL_* identifiers) |
| PET grid logic | `grid_trajectory/pet_grid.py` | Computes grid cell assignments per actor per frame |
| SAM3-grid integration | `grid_trajectory/sam3_grid_pet.py` | Combines SAM3 output with grid + PET logic |
| Trajectory dataset | `traffic_diffusion/data/` | `trajdiff_inputs.npy`, `trajdiff_targets.npy`, `trajdiff_meta.parquet` |

### Phase 5: PET Conflict Extraction

| Component | File | Description |
|-----------|------|-------------|
| End-to-end pipeline | `traffic_analyzer.py` | Orchestrates SAM3 \u2192 BEV \u2192 Grid \u2192 PET |
| PET computation | `grid_trajectory/` | Computes Post Encroachment Time per actor pair |
| Output CSV | `outputs/petevents_bev.csv` | Conflict events with PET, trajectories, grid cells |
| Gate counter | `gate_counter.py` | Counts actors passing through grid gates |
| Research runner | `analysis/research_run.py` | Batch video processing with max-frames support |

### Phase 6: Analysis & Visualization

| Component | File | Description |
|-----------|------|-------------|
| PET summary | `analysis/pet_summary.py` | PET statistics, percentiles, grid hotspot counts |
| Conflict plots | `analysis/visualization/pet_event_plots.py` | BEV conflict visualizations with trajectories |
| Video overlays | `analysis/visualization/video_overlays.py` | Grid overlay + conflict cell highlights on video frames |
| Safety eval (diffusion) | `analysis/safety_eval_diffusion.py` | Batch PET/TTC evaluation with diffusion checkpoint |
| Notebook pipeline | `analysis/safety_eval_diffusion_notebook.py` | Train \u2192 Sample \u2192 Evaluate in one script |

### Phase 7: Diffusion-Based Trajectory Modeling

| Component | File | Description |
|-----------|------|-------------|
| Diffusion model | `traffic_diffusion/trajectory_diffusion.py` | Conditional trajectory diffusion model |
| Training script | `traffic_diffusion/train_trajectory_diffusion.py` | Trains on PET-event futures |
| Model & sampler | `traffic_diffusion/model_and_sampler.py` | Checkpoint loading + counterfactual future sampling |
| Training utils | `traffic_diffusion/training_utils.py` | Data cleaning, loaders, training loop helpers |
| Sampling utils | `traffic_diffusion/sampling_utils.py` | Load checkpoint and sample futures for eval events |
| PET safety metrics | `traffic_diffusion/pet_safety_metrics.py` | Compute PET/TTC from sampled trajectories |
| Episode reward | `traffic_diffusion/episode_reward.py` | Reward functions for trajectory quality assessment |
| PET diffusion analysis | `analysis/pet_diffusion_analysis.py` | Compare real PET vs PET-like from diffusion |
| Diffusion plots | `analysis/visualization/pet_diffusion_plots.py` | Histograms, true-vs-predicted, delta-step plots |

### CLI Flags & Reproducibility

```bash
# Full pipeline run
python traffic_analyzer.py --video videos/traffic_video.mp4 --max-frames 300

# Grid + PET extraction only
make grid VIDEO=videos/traffic_video.mp4

# Diffusion training
python traffic_diffusion/train_trajectory_diffusion.py --data data/ --epochs 100

# Safety evaluation with diffusion
python analysis/safety_eval_diffusion.py --checkpoint checkpoints/diffusion_ckpt.pt
```

### Pipeline Data Flow

```text
[videos/traffic_video.mp4]
        |
        v
[Phase 1: Video Input] --> frames (H, W, 3)
        |
        v
[Phase 2: SAM3 Segmentation] --> actor masks + track IDs
        |
        v
[Phase 3: BEV Transform] --> giti_bev_calib.py, bev_mapper.py
(image -> world coordinates)
        |
        v
[Phase 4: Grid + Trajectory] --> grid_trajectory/
(grid cells + (t,x,y) traj)
        |
        v
[Phase 5: PET Extraction] --> outputs/petevents_bev.csv
(conflict events + PETs)
        |
        +----> [Phase 6: Analysis & Viz]
        |        pet_summary, video_overlay,
        |        conflict_plot, safety_eval
        |
        +----> [Phase 7: Diffusion Modeling]
                 (train + sample + eval)
                 train_trajectory_diffusion.py
                 model_and_sampler.py
                 pet_safety_metrics.py
```


Quickstart
Use the repository with the Make targets or with direct Python entry points.

bash
git clone https://github.com/Sharqascc/nnds.git
cd nnds
make install
Run video-to-PET pipeline
bash
make grid
# or
PYTHONPATH=. python traffic_analyzer.py --video videos/traffic_video.mp4
This runs the end‑to‑end grid/PET pipeline on a video, computes PET events, and writes them to outputs/petevents_bev.csv by default.

A canonical 30‑frame PET events sample, derived from the HF demo video, is versioned at:

docs/data_samples/petevents_bev_demo.csv

You can use this sample directly for quick diffusion experiments without recomputing PETs.

Run video with frame limit (debug / fast tests)
bash
PYTHONPATH=. python traffic_analyzer.py \
  --video videos/traffic_video.mp4 \
  --out-csv outputs/petevents_bev_test.csv \
  --pet-threshold 2.0 \
  --max-frames 30
This processes only the first N frames, which is useful for rapid iteration and debugging.

PET CSV columns
The default PET CSV written by traffic_analyzer.py contains:

column	description
event_id	Integer conflict index
pet	Post Encroachment Time (seconds)
frame	Approximate frame index (optional, may be NaN)
track_a	Track ID of actor i (from SAM3/grid pipeline)
track_b	Track ID of actor j
conflict_type	Grid cell ID where conflict is detected (e.g. CELL_C_1)
world_traj_i	BEV/world trajectory for actor i as (t, x, y) list
world_traj_j	BEV/world trajectory for actor j as (t, x, y) list
These PET CSVs are used as input to downstream diffusion training and safety evaluation.

PET summary and grid hotspot analysis
After generating a PET CSV (e.g. outputs/petevents_bev.csv or outputs/petevents_bev_test.csv), you can print a quick PET distribution and grid hotspot counts with:

bash
PYTHONPATH=. python analysis/pet_summary.py \
  --csv-path outputs/petevents_bev.csv
This script reports:

PET statistics (count, mean, percentiles, etc.).

Counts of events per conflict_type grid cell.

This provides a research‑friendly description of temporal and spatial risk for a given video.

Visualization utilities
The repository includes small, reusable visualization modules under analysis/visualization/ for PET conflicts, grid overlays, and diffusion diagnostics.

BEV PET conflict plots
analysis/visualization/pet_event_plots.py provides helpers to visualize PET events in BEV/world coordinates.

Typical Colab usage:

python
import os
os.chdir("/content/nnds")

from analysis.visualization.pet_event_plots import (
    load_pet_csv,
    compute_timing_from_traj,
    plot_conflict_event,
)

df = load_pet_csv("outputs/petevents_bev_test.csv")
df = compute_timing_from_traj(df)

# Show a single conflict
plot_conflict_event(df, event_id=0)

# Save a gallery of the first few conflicts
import os
os.makedirs("outputs/visualizations", exist_ok=True)

for eid in df["event_id"].head(10):
    plot_conflict_event(
        df,
        event_id=int(eid),
        save_path=f"outputs/visualizations/conflict_{int(eid):03d}.png",
    )
These plots show both actors’ trajectories, closest-approach point, grid cell, and PET value in seconds.

Video grid overlays and conflict cells
analysis/visualization/video_overlays.py provides utilities to draw the grid overlay and highlight conflict cells directly on raw video frames using SpatialGrid.

Example:

python
from analysis.visualization.video_overlays import save_conflict_frame

video_path = "videos/traffic_video.mp4"
grid_config_path = "configs/GITI_grid_config.json"

# Example cell_id and frame index (e.g., from a PET event and fps * t_exit_i)
cell_id = "CELL_T_2"
frame_idx = 50

save_conflict_frame(
    video_path,
    grid_config_path,
    cell_id,
    frame_idx,
    out_path="outputs/visualizations/conflict_frame_000.png",
)
This produces a frame with the neon grid overlay and the selected conflict cell highlighted, useful for qualitative inspection and paper figures.

Diffusion PET visualization
analysis/visualization/pet_diffusion_plots.py works together with analysis/pet_diffusion_analysis.py to visualize PET-like metrics for diffusion-based trajectory modeling.

Core analysis functions (existing):

analysis/pet_diffusion_analysis.compute_pet_like_metrics(batch, sample_future_fn, scale, noise_scale, d_thresh)

analysis/pet_diffusion_analysis.compare_realPET_samplePET(df_pet_path, batch, sample_future_fn, scale, noise_scale, d_thresh)

Visualization helpers:

python
from analysis.pet_diffusion_analysis import (
    compute_pet_like_metrics,
    compare_realPET_samplePET,
)

from analysis.visualization.pet_diffusion_plots import (
    plot_pet_like_histogram,
    plot_true_vs_pet_like,
    plot_true_vs_sample_delta,
)

# Assuming batch, sample_future_fn, and scale are defined by the diffusion code
pet_pairs = compute_pet_like_metrics(batch, sample_future_fn, scale)

records = compare_realPET_samplePET(
    df_pet_path="outputs/petevents_bev_test.csv",
    batch=batch,
    sample_future_fn=sample_future_fn,
    scale=scale,
)

plot_pet_like_histogram(
    pet_pairs,
    out_path="outputs/visualizations/diffusion_pet_like_hist.png",
)

plot_true_vs_pet_like(
    records,
    out_path="outputs/visualizations/diffusion_true_vs_pet_like.png",
)

plot_true_vs_sample_delta(
    records,
    out_path="outputs/visualizations/diffusion_true_vs_delta_steps.png",
)
These plots help compare true PET values with PET-like metrics derived from real and sampled trajectories, supporting diffusion-based safety evaluation.

Colab setup (recommended)
For Google Colab, use the single‑cell bootstrap script below to prepare the environment. It:

clones or updates the nnds repository

switches to the desired branch (defaults to main)

installs Python dependencies

downloads the default demo video into videos/traffic_video.mp4

downloads sam3.pt into the repo root if it is missing

Both the demo video and SAM3 weights are hosted publicly on Hugging Face; no HF token is required.

One-cell Colab bootstrap
Paste this into a fresh Colab cell:

python
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
Run the pipeline in Colab
bash
!cd /content/nnds && PYTHONPATH=. python traffic_analyzer.py --video videos/traffic_video.mp4
Use frame limiting for debugging:

bash
!cd /content/nnds && PYTHONPATH=. python traffic_analyzer.py \
  --video videos/traffic_video.mp4 \
  --out-csv outputs/petevents_bev_test.csv \
  --pet-threshold 2.0 \
  --max-frames 30
Summarize PETs:

bash
!cd /content/nnds && PYTHONPATH=. python analysis/pet_summary.py \
  --csv-path outputs/petevents_bev_test.csv
Project Structure

grid_trajectory/ – Spatial grid mapping and PET computation

analysis/ – Evaluation and research scripts (pet_summary.py, diffusion safety eval, PET visualization, etc.)

analysis/visualization/ – Visualization helpers for PET conflicts, video overlays, and diffusion PET plots

calibration/ – Camera calibration files

configs/ – Grid and calibration configurations

outputs/ – Generated experiment artifacts and evaluation CSV files (ignored by git)

docs/data_samples/ – Versioned sample artifacts (e.g., petevents_bev_demo.csv)

traffic_diffusion/ – Diffusion-based trajectory and safety modules

bev_mapper.py – Bird’s Eye View transformation

giti_bev_calib.py – Homography calibration

traffic_analyzer.py – End-to-end traffic analysis and conflict processing pipeline

traj_diffusion_normalized.py – Normalized diffusion experiment script

Entry points
Grid / PET extraction
traffic_analyzer.py – End-to-end traffic analysis on video, including detection, BEV transformation, grid construction, conflict extraction, and PET computation.

grid_trajectory/ – Core grid and trajectory logic used by traffic_analyzer.py.

Diffusion training
traffic_diffusion/train_trajectory_diffusion.py – Trains the conditional trajectory diffusion model on PET events from outputs/petevents_bev.csv or a sample file like docs/data_samples/petevents_bev_demo.csv.

traffic_diffusion/training_utils.py – Reusable helpers for data cleaning, loader creation, and training loops.

Diffusion safety evaluation
analysis/safety_eval_diffusion.py – Batch PET/TTC evaluation using a saved diffusion checkpoint and writes outputs/safety_eval_diffusion.csv.

analysis/safety_eval_diffusion_notebook.py – Notebook-friendly variant that retrains, samples futures, and produces:

outputs/safety_events_diffusion_model.csv

outputs/safety_eval_diffusion_summary.csv

Key Features

SAM3 video segmentation

Spatial grid zone analysis

Bird’s Eye View world-coordinate mapping

PET computation

Conflict detection

Diffusion-based trajectory modeling

PET and TTC safety evaluation

PET and diffusion visualization utilities (BEV plots, video overlays, PET-like diagnostics)

Development setup
Recommended environment:

Python 3.10+

Google Colab for experiments, or a local Python environment

Install dependencies with:

bash
pip install -r requirements.txt
Typical developer workflow:

bash
git clone https://github.com/Sharqascc/nnds.git
cd nnds
pip install -r requirements.txt
python -m pytest
Repository conventions

grid_trajectory/ is the canonical location for grid and PET logic.

traffic_analyzer.py is the main end-to-end entry point for video-to-PET processing.

analysis/ contains evaluation-oriented scripts (including pet_summary.py and diffusion safety eval).

analysis/visualization/ contains reusable plotting utilities for PET conflicts and diffusion analysis.

traffic_diffusion/ contains reusable model, sampling, and safety modules.

outputs/ stores generated experiment artifacts and evaluation CSV files and is ignored by git.

docs/data_samples/ contains small, versioned sample artifacts (like petevents_bev_demo.csv) for reproducible experiments.

For new research work:

Prefer reusable logic inside traffic_diffusion/ or grid_trajectory/.

Keep one-off experiment runners inside analysis/.

Write generated artifacts into outputs/ with stable, descriptive filenames.

Usage
Code and configs are maintained on GitHub, while the default public demo video and SAM3 weights are hosted on Hugging Face.

Default demo video (Hugging Face)
Dataset:
https://huggingface.co/datasets/sharqascc/traffic-video-dataset

Video file (web view):
https://huggingface.co/datasets/sharqascc/traffic-video-dataset/blob/main/videos/traffic_video.mp4

For scripts and Colab, use the resolve URL so the file is downloaded directly:

bash
cd /content/nnds
mkdir -p videos
wget -O videos/traffic_video.mp4 \
  "https://huggingface.co/datasets/sharqascc/traffic-video-dataset/resolve/main/videos/traffic_video.mp4"

PYTHONPATH=. python traffic_analyzer.py --video videos/traffic_video.mp4
Larger private or experimental videos can still be stored on Google Drive, but all public examples in this repo are expected to work with the Hugging Face–hosted demo video by default.

SAM3 model weights
The SAM3 video segmentation model used by traffic_analyzer.py is stored in a Hugging Face model repo and is not committed to this repo.

For Colab users, the Colab bootstrap script above downloads sam3.pt automatically into the repository root if it is missing.

Manual download:

bash
cd /content/nnds
wget -O sam3.pt \
  "https://huggingface.co/sharqascc/sam3-traffic-model/resolve/main/sam3.pt"
ls -lh sam3.pt

PYTHONPATH=. python traffic_analyzer.py --video videos/traffic_video.mp4
Diffusion-based Safety Evaluation
This repository includes a trajectory diffusion model and PET/TTC-based safety evaluation on PET events extracted from the grid pipeline.

Components:

traffic_diffusion/train_trajectory_diffusion.py – Trains a conditional trajectory diffusion model on PET-event futures using outputs/petevents_bev.csv or a sample file such as docs/data_samples/petevents_bev_demo.csv.

traffic_diffusion/model_and_sampler.py – Loads diffusion checkpoints and samples counterfactual futures given past trajectories.

analysis/safety_eval_diffusion.py – Iterates over PET events, samples multiple futures per event, and computes PET and TTC statistics.

Running the original safety evaluation
bash
cd /content
git clone https://github.com/Sharqascc/nnds.git
cd nnds
PYTHONPATH=. python analysis/safety_eval_diffusion.py
Notebook-friendly diffusion pipeline
For iterative experiments and Colab runs, this repository also includes a notebook-oriented pipeline:

traffic_diffusion/training_utils.py – Data cleaning, normalization, loader creation, and reusable training helpers

traffic_diffusion/sampling_utils.py – Utilities to load a trained checkpoint and sample counterfactual futures

analysis/safety_eval_diffusion_notebook.py – End-to-end notebook-oriented script that:

builds cleaned train and eval loaders

trains the diffusion model and saves a checkpoint

samples future trajectories for evaluation events

constructs an event-level PET and risk table

summarizes safety using traffic_diffusion/pet_safety_metrics.py

Run it in Colab with:

bash
cd /content
git clone https://github.com/Sharqascc/nnds.git
cd nnds
PYTHONPATH=. python analysis/safety_eval_diffusion_notebook.py
License
MIT.
