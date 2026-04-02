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
