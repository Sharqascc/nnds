# NNDS: Non-motorized and Heterogeneous Traffic Safety Analysis

AI-powered system for analyzing vehicle behavior and surrogate safety metrics at unsignalized intersections.

## Project Structure


Entry points
------------

For developers, the main scripts to run the pipeline are:

- **Grid / PET extraction**
  - `traffic_analyzer.py` – end-to-end traffic analysis on video (detection, BEV, grid, conflict extraction, PET computation).
  - `Grid_&_trajectory/` – core grid and trajectory logic used by `traffic_analyzer.py` (new work should prefer this path over `code/Grid_&_trajectory/`).

- **Diffusion training**
  - `traffic_diffusion/train_trajectory_diffusion.py` – trains the conditional trajectory diffusion model on PET events (`outputs/petevents_bev.csv`) using the original Tf=9, 2-agent setup.
  - `traffic_diffusion/training_utils.py` – reusable helpers for data cleaning, loader creation, and training loops, primarily for notebook / experimental runs.

- **Diffusion safety evaluation**
  - `analysis/safety_eval_diffusion.py` – batch PET/TTC evaluation using the saved diffusion checkpoint (`checkpoints/traj_diffusion_best.pt`), writing `outputs/safety_eval_diffusion.csv`.
  - `analysis/safety_eval_diffusion_notebook.py` – notebook-friendly variant that retrains, samples futures, and produces `outputs/safety_events_diffusion_model.csv` and `outputs/safety_eval_diffusion_summary.csv`.

New experiments should generally:
1. Use `traffic_analyzer.py` to generate PET events and grid-based safety data.
2. Use the diffusion scripts above to train and evaluate counterfactual futures on those PET events.


- Grid_&_trajectory/ - Spatial grid mapping and PET computation
- calibration/ - Camera calibration files
- configs/ - Grid and calibration configurations
- traffic_diffusion/ - Diffusion-based trajectory and safety modules
  - trajectory_diffusion.py - Conditional trajectory diffusion model
  - pet_safety_metrics.py - PET/risk metric summarization
  - episode_reward.py - Episode reward calculation from safety metrics
- bev_mapper.py - Bird's Eye View transformation
- giti_bev_calib.py - Homography calibration
- traffic_analyzer.py - Traffic analysis and conflict processing pipeline
- README.md - Project documentation

## Key Features

- SAM3 Video Segmentation
- Spatial Grid Zone Analysis
- BEV World Coordinate Mapping
- PET (Post-Encroachment Time) Computation
- Conflict Detection

## 
Development setup
-----------------

Recommended environment:

- Python 3.10+ (Colab is fine for experiments)
- Install dependencies with:

```bash
pip install -r requirements.txt
```

Typical developer workflow:

```bash
git clone https://github.com/Sharqascc/nnds.git
cd nnds
pip install -r requirements.txt
```

Repository conventions
----------------------

- `Grid_&_trajectory/` is the preferred path for current grid / PET logic.
- `code/Grid_&_trajectory/` should be treated as legacy/compatibility code unless a migration is explicitly being performed.
- `traffic_analyzer.py` is the main end-to-end entry point for video-to-PET processing.
- `analysis/` contains evaluation-oriented scripts.
- `traffic_diffusion/` contains reusable model, sampling, and safety modules.
- `outputs/` stores generated experiment artifacts and evaluation CSV files.

For new research work:
1. Prefer adding reusable logic inside `traffic_diffusion/` or `Grid_&_trajectory/`.
2. Keep one-off experiment runners inside `analysis/`.
3. Write generated artifacts into `outputs/` with stable, descriptive filenames.


Usage

Videos and outputs stored on Google Drive.
Code and configs stored on GitHub.

## License

MIT

## Diffusion-based Safety Evaluation

This repo includes a trajectory diffusion model and PET/TTC-based safety evaluation on PET events extracted from the grid pipeline.

### Components

- `traffic_diffusion/train_trajectory_diffusion.py`  
  Trains a conditional trajectory diffusion model on PET-event futures using `outputs/petevents_bev.csv`. Futures are normalized around the last past position and fixed to a horizon of `Tf = 9` steps for two agents (`traj_shape = (9, 2, 2)`).

- `traffic_diffusion/model_and_sampler.py`  
  Helper to load diffusion checkpoints (`checkpoints/traj_diffusion_best.pt`) and sample counterfactual futures given past trajectories (`sample_future_denorm`).

- `analysis/safety_eval_diffusion.py`  
  Iterates over all PET events, samples multiple futures per event, and computes:
  - true PET and real minimum TTC,
  - diffusion-based minimum TTC statistics,
  - fraction of sampled trajectories that yield TTC and PET events (`sample_ttc_defined_frac`, `sample_pet_defined_frac`),  
  writing results to `outputs/safety_eval_diffusion.csv`.

### Running the safety evaluation

In Colab or a similar environment:

```bash
%cd /content
!git clone https://github.com/Sharqascc/nnds.git
%cd nnds
!PYTHONPATH=. python analysis/safety_eval_diffusion.py
```

The evaluation uses `checkpoints/traj_diffusion_best.pt` and `outputs/petevents_bev.csv` and produces `outputs/safety_eval_diffusion.csv` with per-event diffusion-based safety statistics.

### Notebook-friendly diffusion pipeline

For iterative experiments and Colab runs, this repo also includes a notebook-style pipeline:

- `traffic_diffusion/training_utils.py` – data cleaning/normalization, loader creation, and a reusable `train_diffusion_model` helper.
- `traffic_diffusion/sampling_utils.py` – utilities to load a trained diffusion checkpoint and sample counterfactual futures over the evaluation loader.
- `analysis/safety_eval_diffusion_notebook.py` – end-to-end script that:
  - builds cleaned train/eval loaders from the PET event dataset,
  - trains the conditional trajectory diffusion model and saves `checkpoints/traj_diffusion_best.pt`,
  - samples future trajectories for evaluation events,
  - constructs an event-level PET/risk table,
  - summarizes safety using `traffic_diffusion/pet_safety_metrics.py` (`compute_safety_metrics`) and writes:
    - `outputs/safety_events_diffusion_model.csv`
    - `outputs/safety_eval_diffusion_summary.csv`.

To run this pipeline in Colab:

```bash
%cd /content
!git clone https://github.com/Sharqascc/nnds.git
%cd nnds
!PYTHONPATH=. python analysis/safety_eval_diffusion_notebook.py
```

This workflow mirrors the original `analysis/safety_eval_diffusion.py` evaluation but is structured for quicker iteration and integration in notebook-based experiments.
