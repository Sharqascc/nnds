# NNDS: Non-motorized and Heterogeneous Traffic Safety Analysis

AI-powered system for analyzing vehicle behavior and surrogate safety metrics at unsignalized intersections.

## Project Structure

- `Grid_&_trajectory/` - Spatial grid mapping and PET computation
- `analysis/` - Evaluation and research scripts
- `calibration/` - Camera calibration files
- `code/Grid_&_trajectory/` - Legacy or compatibility grid code
- `configs/` - Grid and calibration configurations
- `outputs/` - Generated experiment artifacts and evaluation CSV files
- `traffic_diffusion/` - Diffusion-based trajectory and safety modules
- `bev_mapper.py` - Bird's Eye View transformation
- `giti_bev_calib.py` - Homography calibration
- `traffic_analyzer.py` - Traffic analysis and conflict processing pipeline
- `traj_diffusion_normalized.py` - Normalized diffusion experiment script

## Entry points

For developers, the main scripts to run the pipeline are:

### Grid / PET extraction

- `traffic_analyzer.py` - End-to-end traffic analysis on video, including detection, BEV, grid, conflict extraction, and PET computation.
- `Grid_&_trajectory/` - Core grid and trajectory logic used by `traffic_analyzer.py`. New work should prefer this path over `code/Grid_&_trajectory/`.

### Diffusion training

- `traffic_diffusion/train_trajectory_diffusion.py` - Trains the conditional trajectory diffusion model on PET events from `outputs/petevents_bev.csv` using the original `Tf = 9`, 2-agent setup.
- `traffic_diffusion/training_utils.py` - Reusable helpers for data cleaning, loader creation, and training loops, mainly for notebook and experimental runs.

### Diffusion safety evaluation

- `analysis/safety_eval_diffusion.py` - Batch PET/TTC evaluation using the saved diffusion checkpoint `checkpoints/traj_diffusion_best.pt`, writing `outputs/safety_eval_diffusion.csv`.
- `analysis/safety_eval_diffusion_notebook.py` - Notebook-friendly variant that retrains, samples futures, and produces:
  - `outputs/safety_events_diffusion_model.csv`
  - `outputs/safety_eval_diffusion_summary.csv`

## Key Features

- SAM3 video segmentation
- Spatial grid zone analysis
- BEV world coordinate mapping
- PET (Post-Encroachment Time) computation
- Conflict detection

## Development setup

Recommended environment:

- Python 3.10+
- Colab is fine for experiments

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

- `Grid_&_trajectory/` is the preferred path for current grid and PET logic.
- `code/Grid_&_trajectory/` should be treated as legacy or compatibility code unless a migration is explicitly being performed.
- `traffic_analyzer.py` is the main end-to-end entry point for video-to-PET processing.
- `analysis/` contains evaluation-oriented scripts.
- `traffic_diffusion/` contains reusable model, sampling, and safety modules.
- `outputs/` stores generated experiment artifacts and evaluation CSV files.

For new research work:

1. Prefer adding reusable logic inside `traffic_diffusion/` or `Grid_&_trajectory/`.
2. Keep one-off experiment runners inside `analysis/`.
3. Write generated artifacts into `outputs/` with stable, descriptive filenames.

## Usage

Videos and outputs are stored on Google Drive. Code and configs are stored on GitHub.

## License

MIT

## Diffusion-based Safety Evaluation

This repo includes a trajectory diffusion model and PET/TTC-based safety evaluation on PET events extracted from the grid pipeline.

### Components

- `traffic_diffusion/train_trajectory_diffusion.py`  
  Trains a conditional trajectory diffusion model on PET-event futures using `outputs/petevents_bev.csv`. Futures are normalized around the last past position and fixed to a horizon of `Tf = 9` steps for two agents (`traj_shape = (9, 2, 2)`).

- `traffic_diffusion/model_and_sampler.py`  
  Loads diffusion checkpoints such as `checkpoints/traj_diffusion_best.pt` and samples counterfactual futures given past trajectories.

- `analysis/safety_eval_diffusion.py`  
  Iterates over all PET events, samples multiple futures per event, and computes true PET, real minimum TTC, diffusion-based TTC statistics, and sampled-event fractions, writing results to `outputs/safety_eval_diffusion.csv`.

### Running the original safety evaluation

```bash
cd /content
git clone https://github.com/Sharqascc/nnds.git
cd nnds
PYTHONPATH=. python analysis/safety_eval_diffusion.py
```

### Notebook-friendly diffusion pipeline

For iterative experiments and Colab runs, this repo also includes a notebook-style pipeline:

- `traffic_diffusion/training_utils.py` - Data cleaning and normalization, loader creation, and reusable training helpers
- `traffic_diffusion/sampling_utils.py` - Utilities to load a trained checkpoint and sample counterfactual futures
- `analysis/safety_eval_diffusion_notebook.py` - End-to-end notebook-oriented script that:
  - builds cleaned train and eval loaders
  - trains the diffusion model and saves `checkpoints/traj_diffusion_best.pt`
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
