# NNDS: Non-motorized and Heterogeneous Traffic Safety Analysis

AI-powered system for analyzing vehicle behavior and surrogate safety metrics at unsignalized intersections.

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
PYTHONPATH=. python traffic_analyzer.py
```

### Train diffusion model

```bash
make diffusion-train
# or
PYTHONPATH=. python traffic_diffusion/train_trajectory_diffusion.py
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

## Project Structure

- `grid_trajectory/` – Spatial grid mapping and PET computation
- `analysis/` – Evaluation and research scripts
- `calibration/` – Camera calibration files
- `configs/` – Grid and calibration configurations
- `outputs/` – Generated experiment artifacts and evaluation CSV files
- `traffic_diffusion/` – Diffusion-based trajectory and safety modules
- `bev_mapper.py` – Bird's Eye View transformation
- `giti_bev_calib.py` – Homography calibration
- `traffic_analyzer.py` – Traffic analysis and conflict processing pipeline
- `traj_diffusion_normalized.py` – Normalized diffusion experiment script

## Entry points

### Grid / PET extraction

- `traffic_analyzer.py` – End-to-end traffic analysis on video, including detection, BEV transformation, grid construction, conflict extraction, and PET computation.
- `grid_trajectory/` – Core grid and trajectory logic used by `traffic_analyzer.py`.

### Diffusion training

- `traffic_diffusion/train_trajectory_diffusion.py` – Trains the conditional trajectory diffusion model on PET events from `outputs/petevents_bev.csv`.
- `traffic_diffusion/training_utils.py` – Reusable helpers for data cleaning, loader creation, and training loops.

### Diffusion safety evaluation

- `analysis/safety_eval_diffusion.py` – Batch PET/TTC evaluation using a saved diffusion checkpoint and writes `outputs/safety_eval_diffusion.csv`.
- `analysis/safety_eval_diffusion_notebook.py` – Notebook-friendly variant that retrains, samples futures, and produces:
  - `outputs/safety_events_diffusion_model.csv`
  - `outputs/safety_eval_diffusion_summary.csv`

## Key Features

- SAM3 video segmentation
- Spatial grid zone analysis
- Bird's Eye View world-coordinate mapping
- PET computation
- Conflict detection
- Diffusion-based trajectory modeling
- PET and TTC safety evaluation

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
```

## Repository conventions

- `grid_trajectory/` is the canonical location for grid and PET logic.
- `traffic_analyzer.py` is the main end-to-end entry point for video-to-PET processing.
- `analysis/` contains evaluation-oriented scripts.
- `traffic_diffusion/` contains reusable model, sampling, and safety modules.
- `outputs/` stores generated experiment artifacts and evaluation CSV files.

For new research work:

1. Prefer reusable logic inside `traffic_diffusion/` or `grid_trajectory/`.
2. Keep one-off experiment runners inside `analysis/`.
3. Write generated artifacts into `outputs/` with stable, descriptive filenames.

## Usage

Code and configs are maintained on GitHub, while the default public demo video is hosted on Hugging Face. [page:77]

### Default demo video (Hugging Face)

The canonical example video for `traffic_analyzer.py` lives in a dataset repo:

- Dataset: <https://huggingface.co/datasets/sharqascc/traffic-video-dataset>
- Video file (web view):  
  <https://huggingface.co/datasets/sharqascc/traffic-video-dataset/blob/main/videos/traffic_video.mp4>

For scripts and Colab, use the `resolve` URL so the file is downloaded directly:

```bash
mkdir -p videos
wget -O videos/traffic_video.mp4 \
  "https://huggingface.co/datasets/sharqascc/traffic-video-dataset/resolve/main/videos/traffic_video.mp4"

PYTHONPATH=. python traffic_analyzer.py --video videos/traffic_video.mp4
```

Larger private or experimental videos can still be stored on Google Drive, but all public examples in this repo are expected to work with the Hugging Face–hosted demo video by default. [page:77]

## Diffusion-based Safety Evaluation

This repository includes a trajectory diffusion model and PET/TTC-based safety evaluation on PET events extracted from the grid pipeline. [page:77]

### Components

- `traffic_diffusion/train_trajectory_diffusion.py` – Trains a conditional trajectory diffusion model on PET-event futures using `outputs/petevents_bev.csv`.
- `traffic_diffusion/model_and_sampler.py` – Loads diffusion checkpoints and samples counterfactual futures given past trajectories.
- `analysis/safety_eval_diffusion.py` – Iterates over PET events, samples multiple futures per event, and computes PET and TTC statistics. [page:77]

### Running the original safety evaluation

```bash
cd /content
git clone https://github.com/Sharqascc/nnds.git
cd nnds
PYTHONPATH=. python analysis/safety_eval_diffusion.py
```

### Notebook-friendly diffusion pipeline

For iterative experiments and Colab runs, this repository also includes a notebook-style pipeline: [page:77]

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

## License

MIT
