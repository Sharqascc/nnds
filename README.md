# NNDS: Non-motorized and Heterogeneous Traffic Safety Analysis

AI-powered system for analyzing vehicle behavior and surrogate safety metrics at unsignalized intersections.

## Project Structure

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

## Usage

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
