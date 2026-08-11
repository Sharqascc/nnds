"""Publication-quality visualization for traffic safety analysis.

Provides industry-standard plots for Surrogate Safety Measures (SSM):
- Distribution plots (histogram + KDE)
- Time series analysis
- Severity scatter plots
- Comparative boxplots
- Cumulative distribution functions (ECDF)
- Correlation heatmaps
- Temporal density heatmaps
- Conflict density maps
- Diffusion model evaluation plots
- Individual conflict event visualization
- Video frame overlays with trajectories

Features:
- Journal-ready styling (300 DPI, serif fonts, proper sizing)
- Colorblind-safe palettes (Okabe-Ito)
- Vector graphics export (PDF + PNG)
- Statistical annotations (p-values, regression, effect sizes)
- Configurable safety thresholds
- APA/IEEE compliant formatting
- Video generation (MP4 output)

Compliant with:
- Transportation Research Board (TRB) standards
- IEEE Transactions on Intelligent Transportation Systems
- Accident Analysis & Prevention journal requirements
"""

# Import SSM visualization functions
from .industry_standard_viz import (
    SSMPlotter,
    plot_pet_distribution,
    plot_ttc_time_series,
    plot_severity_scatter,
    plot_conflict_density_map,
    plot_comparative_boxplot,
    plot_cumulative_distribution,
    plot_correlation_heatmap,
    plot_temporal_heatmap
)

# Import diffusion model evaluation functions
from .pet_diffusion_plots import (
    DiffusionPETPlotter,
    plot_pet_like_histogram,
    plot_true_vs_pet_like,
    plot_true_vs_sample_delta,
    plot_residual_analysis,
    plot_bland_altman
)

# Import conflict event visualization functions
from .pet_event_plots import (
    EventPlotter,
    load_pet_csv,
    compute_timing_from_traj,
    plot_conflict_event,
    plot_multiple_events,
    get_class_default
)

# Import video overlay functions
from .video_overlays import (
    VideoOverlayPlotter,
    overlay_conflict_frame,
    generate_conflict_video,
    create_before_during_after,
    save_conflict_frame
)

__all__ = [
    # SSM Analysis - Main plotter class
    'SSMPlotter',

    # SSM Analysis - Individual plot functions
    'plot_pet_distribution',
    'plot_ttc_time_series',
    'plot_severity_scatter',
    'plot_conflict_density_map',
    'plot_comparative_boxplot',
    'plot_cumulative_distribution',
    'plot_correlation_heatmap',
    'plot_temporal_heatmap',

    # Diffusion Model Evaluation
    'DiffusionPETPlotter',
    'plot_pet_like_histogram',
    'plot_true_vs_pet_like',
    'plot_true_vs_sample_delta',
    'plot_residual_analysis',
    'plot_bland_altman',

    # Conflict Event Visualization
    'EventPlotter',
    'load_pet_csv',
    'compute_timing_from_traj',
    'plot_conflict_event',
    'plot_multiple_events',
    'get_class_default',

    # Video Overlay Visualization
    'VideoOverlayPlotter',
    'overlay_conflict_frame',
    'generate_conflict_video',
    'create_before_during_after',
    'save_conflict_frame',
]

__version__ = '2.3.0'  # Updated for video overlays


# Quick usage examples
__doc_examples__ = """
Quick Start Examples
====================

SSM ANALYSIS PLOTS
==================

1. Basic PET Distribution
--------------------------
from analysis.visualization import plot_pet_distribution
import pandas as pd

df = pd.read_csv('outputs/petevents_bev_30frames.csv')
fig = plot_pet_distribution(
    df['pet'].values,
    style='journal',
    save_path='outputs/pet_dist.png'
)

2. Time Series with Critical Events
------------------------------------
from analysis.visualization import plot_ttc_time_series

fig = plot_ttc_time_series(
    timestamps=df['frame'].values / 30.0,  # Convert frames to seconds
    ttc_values=df['TTC'].values,
    highlight_critical=True,
    save_path='outputs/ttc_series.png'
)

3. Severity Correlation Scatter
--------------------------------
from analysis.visualization import plot_severity_scatter

fig = plot_severity_scatter(
    pet_values=df['pet'].values,
    ttc_values=df['TTC'].values,
    add_regression=True,
    save_path='outputs/severity.png'
)

4. Before/After Comparison
---------------------------
from analysis.visualization import plot_comparative_boxplot

before = df[df['period']=='before']['pet'].values
after = df[df['period']=='after']['pet'].values

fig = plot_comparative_boxplot(
    data_groups={'Before': before, 'After': after},
    metric_name='PET',
    show_stats=True,
    save_path='outputs/comparison.png'
)

5. Advanced: Custom Thresholds
-------------------------------
from analysis.visualization import SSMPlotter

plotter = SSMPlotter(
    style='journal',
    dpi=300,
    colorblind_safe=True,
    critical_pet=0.75,  # Custom threshold
    safe_pet=4.0
)

fig = plotter.plot_pet_distribution(
    df['pet'].values,
    save_path='outputs/custom_pet.png'
)

6. Multi-Metric Correlation
----------------------------
from analysis.visualization import plot_correlation_heatmap

fig = plot_correlation_heatmap(
    data_dict={
        'PET': df['pet'].values,
        'TTC': df['TTC'].values,
        'DRAC': df['DRAC'].values,
        'Speed': df['speed'].values
    },
    save_path='outputs/correlation.png'
)

7. Temporal Density Heatmap
----------------------------
from analysis.visualization import plot_temporal_heatmap

fig = plot_temporal_heatmap(
    timestamps=df['frame'].values / 30.0,
    pet_values=df['pet'].values,
    time_bins=30,
    severity_bins=15,
    save_path='outputs/temporal_density.png'
)

8. ECDF for Multiple Groups
----------------------------
from analysis.visualization import plot_cumulative_distribution

fig = plot_cumulative_distribution(
    data_groups={
        'Intersection A': df_a['pet'].values,
        'Intersection B': df_b['pet'].values,
        'Intersection C': df_c['pet'].values
    },
    metric_name='PET',
    save_path='outputs/ecdf.png'
)


DIFFUSION MODEL EVALUATION
===========================

9. PET-like Error Distribution
-------------------------------
from analysis.visualization import plot_pet_like_histogram

# pet_pairs = [(real_steps, sample_steps), ...]
fig = plot_pet_like_histogram(
    pet_pairs=pet_pairs,
    bins=30,
    show_stats=True,
    save_path='outputs/diffusion/error_histogram.png'
)

10. Ground Truth vs Generated
------------------------------
from analysis.visualization import plot_true_vs_pet_like

# records = [(idx, true_pet_sec, real_steps, sample_steps), ...]
fig = plot_true_vs_pet_like(
    records=records,
    add_regression=True,
    save_path='outputs/diffusion/scatter.png'
)

11. Generation Error Analysis
------------------------------
from analysis.visualization import plot_true_vs_sample_delta

fig = plot_true_vs_sample_delta(
    records=records,
    add_trend=True,
    save_path='outputs/diffusion/error_vs_truth.png'
)

12. Residual Diagnostics
-------------------------
from analysis.visualization import plot_residual_analysis

# Complete diagnostic suite (4 plots in one figure)
fig = plot_residual_analysis(
    records=records,
    save_path='outputs/diffusion/residuals.png'
)

13. Bland-Altman Agreement
---------------------------
from analysis.visualization import plot_bland_altman

# Agreement analysis between real and sampled
fig = plot_bland_altman(
    records=records,
    save_path='outputs/diffusion/bland_altman.png'
)

14. Complete Diffusion Evaluation Suite
----------------------------------------
from analysis.visualization import DiffusionPETPlotter

plotter = DiffusionPETPlotter(
    style='default',
    dpi=300,
    save_pdf=True  # Save both PNG and PDF
)

# Generate all plots at once
plotter.plot_all(
    pet_pairs=pet_pairs,
    records=records,
    out_dir='outputs/diffusion_eval'
)


CONFLICT EVENT VISUALIZATION
=============================

15. Single Conflict Event (Quick)
----------------------------------
from analysis.visualization import plot_conflict_event, load_pet_csv

df = load_pet_csv('outputs/petevents_bev_30frames.csv')
fig = plot_conflict_event(
    df,
    event_id=5,
    save_path='outputs/events/event_5.png'
)

16. Event with Velocity Vectors
--------------------------------
from analysis.visualization import EventPlotter, compute_timing_from_traj

df = load_pet_csv('outputs/petevents_bev_30frames.csv')
df = compute_timing_from_traj(df)  # Add timing info

plotter = EventPlotter(dpi=300)
fig = plotter.plot_conflict_event(
    df,
    event_id=5,
    show_velocities=True,      # Show velocity arrows
    show_conflict_zone=True,   # Show conflict zone circle
    save_path='outputs/events/event_5_detailed.png'
)

17. Batch Plot Critical Events
-------------------------------
from analysis.visualization import plot_multiple_events

# Get top 20 most critical events
critical_events = df.nsmallest(20, 'pet')['event_id'].tolist()

plot_multiple_events(
    df,
    event_ids=critical_events,
    save_dir='outputs/critical_events',
    dpi=300,
    save_pdf=True  # Save PNG + PDF for each event
)

18. Custom Severity Thresholds for Events
------------------------------------------
from analysis.visualization import EventPlotter

# Highway-specific thresholds
plotter = EventPlotter(
    dpi=300,
    conflict_zone_radius=3.5,  # Larger for highways
    thresholds={
        'critical': 1.0,   # < 1.0s = critical
        'serious': 2.0,    # 1.0-2.0s = serious
        'moderate': 3.0,   # 2.0-3.0s = moderate
        'safe': 7.0        # > 7.0s = safe
    }
)

fig = plotter.plot_conflict_event(
    df,
    event_id=10,
    show_velocities=True,
    save_path='outputs/highway_event_10.png'
)

19. Custom Vehicle Class Mapper
--------------------------------
from analysis.visualization import EventPlotter

def my_class_mapper(track_id):
    # Map track IDs to vehicle types
    if track_id < 100:
        return 'Car'
    elif track_id < 200:
        return 'Truck'
    else:
        return 'Motorcycle'

plotter = EventPlotter(dpi=300)
fig = plotter.plot_conflict_event(
    df,
    event_id=15,
    class_mapper=my_class_mapper,
    save_path='outputs/event_15_classified.png'
)

20. Generate Event Case Study for Paper
----------------------------------------
from analysis.visualization import EventPlotter, compute_timing_from_traj

# For Figure 3 in your paper
df = load_pet_csv('outputs/petevents_bev_30frames.csv')
df = compute_timing_from_traj(df)

plotter = EventPlotter(
    dpi=300,
    style='journal',
    font_size=10,
    conflict_zone_radius=2.0,
    arrow_scale=3.5
)

# Most critical event for case study
most_critical = df.nsmallest(1, 'pet')['event_id'].iloc[0]

fig = plotter.plot_conflict_event(
    df,
    event_id=most_critical,
    show_velocities=True,
    show_conflict_zone=True,
    save_path=f'paper_figures/figure3_critical_event_{most_critical}.png',
    save_pdf=True  # PDF for LaTeX
)


VIDEO OVERLAY VISUALIZATION
============================

21. Single Frame with Trajectories (Quick)
-------------------------------------------
from analysis.visualization import overlay_conflict_frame

# Simple trajectory overlay on one frame
trajectories = [
    [(0, 100, 200), (0.1, 105, 205), ...],  # Track 1
    [(0, 300, 400), (0.1, 295, 395), ...]   # Track 2
]

frame = overlay_conflict_frame(
    video_path='videos/traffic_video.mp4',
    frame_idx=1234,
    trajectories=trajectories,
    pet_value=0.8,
    save_path='outputs/video_frames/frame_1234.png'
)

22. Complete Video Frame with All Overlays
-------------------------------------------
from analysis.visualization import VideoOverlayPlotter
from grid_trajectory.spatial_grid import SpatialGrid

plotter = VideoOverlayPlotter(dpi=300, colorblind_safe=True)
grid = SpatialGrid('configs/GITI_grid_config.json')

frame = plotter.overlay_conflict_frame(
    video_path='videos/traffic_video.mp4',
    frame_idx=1234,
    trajectories=trajectories,
    track_ids=[42, 57],
    pet_value=0.75,
    conflict_center=(320, 240),  # Conflict location
    grid=grid,
    cell_id='G_A_3'
)

plotter.save_frame(frame, 'paper_figures/fig5_video_frame.png')

23. Generate Conflict Event Video (MP4)
----------------------------------------
from analysis.visualization import generate_conflict_video

# Create MP4 video of conflict event
generate_conflict_video(
    video_path='videos/traffic_video.mp4',
    frame_range=(1200, 1300),  # 100 frames (3.3 seconds at 30fps)
    trajectories=trajectories,
    output_path='paper_supplements/video_S1_conflict.mp4'
)

24. Before/During/After Sequence for Paper
-------------------------------------------
from analysis.visualization import create_before_during_after

# 3-panel figure showing conflict evolution
create_before_during_after(
    video_path='videos/traffic_video.mp4',
    before_idx=1200,   # 1 second before
    during_idx=1230,   # At closest approach
    after_idx=1260,    # 1 second after
    trajectories=trajectories,
    save_path='paper_figures/fig6_conflict_sequence.png'
)

25. Advanced: Custom Video Overlays
------------------------------------
from analysis.visualization import VideoOverlayPlotter

plotter = VideoOverlayPlotter(
    dpi=300,
    colorblind_safe=True,
    font_scale=0.8,
    line_thickness=3,
    show_grid=True,
    grid_alpha=0.4
)

# Overlay trajectories with bounding boxes
frame = plotter.overlay_trajectories(
    frame,
    trajectories=trajectories,
    track_ids=[42, 57],
    show_arrows=True
)

# Add bounding boxes
boxes = [(100, 200, 150, 250), (300, 400, 350, 450)]
frame = plotter.overlay_bounding_boxes(
    frame,
    boxes=boxes,
    track_ids=[42, 57]
)

# Add conflict statistics
frame = plotter.overlay_conflict_info(
    frame,
    pet_value=0.75,
    ttc_value=1.2,
    frame_number=1234,
    timestamp=41.13,
    position='top-left'
)

# Add conflict zone
frame = plotter.overlay_conflict_zone(
    frame,
    center=(320, 240),
    radius=50
)

plotter.save_frame(frame, 'outputs/complete_overlay.png')

26. Batch Process Video Frames
-------------------------------
from analysis.visualization import VideoOverlayPlotter
import pandas as pd

df = pd.read_csv('outputs/petevents_bev_30frames.csv')
plotter = VideoOverlayPlotter(dpi=300)

# Process top 10 critical events
for idx, row in df.nsmallest(10, 'pet').iterrows():
    frame = plotter.overlay_conflict_frame(
        video_path='videos/traffic_video.mp4',
        frame_idx=int(row['frame']),
        trajectories=[row['traj_i'], row['traj_j']],
        track_ids=[row['track_a'], row['track_b']],
        pet_value=row['pet']
    )

    plotter.save_frame(
        frame,
        f'outputs/video_frames/event_{row["event_id"]:04d}.png'
    )

27. Video Comparison (Real vs Diffusion)
-----------------------------------------
from analysis.visualization import VideoOverlayPlotter
import numpy as np

plotter = VideoOverlayPlotter(dpi=300)

# Real trajectories
real_frame = plotter.overlay_trajectories(
    frame.copy(),
    real_trajectories,
    track_ids=[1, 2]
)

# Diffusion-generated trajectories
sampled_frame = plotter.overlay_trajectories(
    frame.copy(),
    sampled_trajectories,
    track_ids=[1, 2]
)

# Combine side-by-side
comparison = np.hstack([real_frame, sampled_frame])
plotter.save_frame(comparison, 'paper_figures/fig7_real_vs_diffusion_video.png')

28. Generate Video Supplement with Statistics
----------------------------------------------
from analysis.visualization import VideoOverlayPlotter

plotter = VideoOverlayPlotter(dpi=300)

# Generate video with full overlays
plotter.generate_conflict_video(
    video_path='videos/traffic_video.mp4',
    frame_range=(1200, 1300),
    trajectories=trajectories,
    track_ids=[42, 57],
    pet_value=0.75,
    output_path='paper_supplements/video_S2_annotated.mp4',
    fps=30
)


Output Formats
==============
All plots automatically save in:
- PNG (high-res raster, 300 DPI)
- PDF (vector graphics for LaTeX/Word) [where applicable]
- MP4 (video output with H.264 codec)

All formats are journal-ready and submission-compliant.


Module Organization
===================
- industry_standard_viz.py: General SSM analysis plots
- pet_diffusion_plots.py: Diffusion model evaluation plots
- pet_event_plots.py: Individual conflict event visualization
- video_overlays.py: Video frame overlays and MP4 generation

All modules use:
- Colorblind-safe Okabe-Ito palette
- Publication-quality styling (300 DPI)
- Statistical annotations
- Configurable parameters
- Severity-based color coding
"""
