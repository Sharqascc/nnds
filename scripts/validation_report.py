#!/usr/bin/env python3
"""
Quantitative scientific validation report for NNDS pipeline.

Computes:
  - Detection quality metrics (count, confidence stats, class distribution)
  - Tracking stability metrics (track lengths, fragmentation, gaps/jumps)
  - BEV homography accuracy (reprojection error)
  - PET event metrics (count, PET distribution, validity)
  - Reproducibility checklist

Outputs a Markdown report to outputs/validation_report.md

Usage:
    python scripts/validation_report.py \
        --detections outputs/petevents_bev_final_detections.csv \
        --pet outputs/petevents_bev_final.csv \
        --bev-config configs/bev_config.json \
        --calib configs/giti_calibration_points.json
"""
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def bootstrap_ci(data, n_bootstrap=1000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for the median."""
    rng = np.random.default_rng(seed)
    medians = []
    data = np.asarray(data)
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        medians.append(np.median(sample))
    lower = np.percentile(medians, ((1-ci)/2)*100)
    upper = np.percentile(medians, (1-(1-ci)/2)*100)
    return lower, upper

REPO = Path(__file__).resolve().parents[1]


def check_df_columns(df, required):
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def detection_metrics(det_df):
    required = ['frame','track_id','class_name','conf','x1','y1','x2','y2']
    check_df_columns(det_df, required)
    metrics = {
        'total_detections': len(det_df),
        'unique_tracks': det_df['track_id'].nunique(),
        'mean_conf': float(det_df['conf'].mean()),
        'min_conf': float(det_df['conf'].min()),
        'max_conf': float(det_df['conf'].max()),
        'class_counts': det_df['class_name'].value_counts().to_dict(),
        'invalid_boxes': int(((det_df['x1'] >= det_df['x2']) | (det_df['y1'] >= det_df['y2'])).sum()),
    }
    if len(pet_values) >= 10:
        try:
            lower, upper = bootstrap_ci(pet_values)
            metrics['pet_median_ci_lower'] = float(lower)
            metrics['pet_median_ci_upper'] = float(upper)
        except Exception:
            pass
    return metrics


def tracking_metrics(det_df, max_gap=10, max_jump=50.0):
    required = ['track_id','frame','cx','cy']
    check_df_columns(det_df, required)
    track_lengths = []
    gaps_over = 0
    jumps_over = 0
    for track_id, group in det_df.groupby('track_id'):
        group = group.sort_values('frame')
        track_lengths.append(len(group))
        if len(group) >= 2:
            frames = group['frame'].values
            gaps = np.diff(frames)
            gaps_over += int((gaps > max_gap).sum())
            x = group['cx'].values
            y = group['cy'].values
            dx = np.diff(x)
            dy = np.diff(y)
            jumps = np.sqrt(dx**2 + dy**2)
            jumps_over += int((jumps > max_jump).sum())
    metrics = {
        'total_tracks': len(track_lengths),
        'mean_track_length': float(np.mean(track_lengths)) if track_lengths else 0,
        'median_track_length': float(np.median(track_lengths)) if track_lengths else 0,
        'max_track_length': int(np.max(track_lengths)) if track_lengths else 0,
        'tracks_with_gap_over': gaps_over,
        'tracks_with_jump_over': jumps_over,
        'fragmentation_score': float(gaps_over + jumps_over) / max(len(track_lengths), 1),
    }
    if len(pet_values) >= 10:
        try:
            lower, upper = bootstrap_ci(pet_values)
            metrics['pet_median_ci_lower'] = float(lower)
            metrics['pet_median_ci_upper'] = float(upper)
        except Exception:
            pass
    return metrics


def bev_metrics(bev_config_path, calib_path):
    with open(bev_config_path) as f:
        bev_cfg = json.load(f)
    with open(calib_path) as f:
        calib = json.load(f)
    H = np.array(bev_cfg['H_pixel_to_world'], dtype=float)
    pixel_pts = []
    world_pts = []
    for p in calib['calibration_points']:
        pixel_pts.append([p['pixel']['x'], p['pixel']['y']])
        world_pts.append([p['world']['easting'], p['world']['northing']])
    pixel_pts = np.array(pixel_pts, dtype=float)
    world_pts = np.array(world_pts, dtype=float)
    pixel_h = np.hstack([pixel_pts, np.ones((len(pixel_pts),1))])
    proj_world = H @ pixel_h.T
    proj_world = proj_world / proj_world[2,:]
    proj_world = proj_world[:2,:].T
    errors = np.linalg.norm(proj_world - world_pts, axis=1)
    metrics = {
        'rank': int(np.linalg.matrix_rank(H)),
        'condition_number_raw': float(np.linalg.cond(H)),
        'reprojection_error_mean': float(errors.mean()),
        'reprojection_error_max': float(errors.max()),
        'reprojection_error_std': float(errors.std()),
        'num_calibration_points': len(pixel_pts),
    }
    if len(pet_values) >= 10:
        try:
            lower, upper = bootstrap_ci(pet_values)
            metrics['pet_median_ci_lower'] = float(lower)
            metrics['pet_median_ci_upper'] = float(upper)
        except Exception:
            pass
    return metrics


def pet_metrics(pet_df):
    required = ['pet','frame','track_a','track_b']
    check_df_columns(pet_df, required)
    pet_values = pet_df['pet'].values
    non_positive = int((pet_values <= 0).sum())
    metrics = {
        'total_events': len(pet_df),
        'non_positive_pet': non_positive,
        'min_pet': float(pet_values.min()) if len(pet_values) else None,
        'max_pet': float(pet_values.max()) if len(pet_values) else None,
        'median_pet': float(np.median(pet_values)) if len(pet_values) else None,
        'mean_pet': float(np.mean(pet_values)) if len(pet_values) else None,
        'std_pet': float(np.std(pet_values)) if len(pet_values) else None,
            'pet_median_ci_lower': None,
        'pet_median_ci_upper': None,
    }
    if len(pet_values) >= 10:
        try:
            lower, upper = bootstrap_ci(pet_values)
            metrics['pet_median_ci_lower'] = float(lower)
            metrics['pet_median_ci_upper'] = float(upper)
        except Exception:
            pass
    return metrics


def generate_report(det_path, pet_path, bev_config_path, calib_path, output_path):
    det = pd.read_csv(det_path)
    pet = pd.read_csv(pet_path)
    det_metrics = detection_metrics(det)
    track_metrics = tracking_metrics(det)
    bev_metrics_dict = bev_metrics(bev_config_path, calib_path)
    pet_metrics_dict = pet_metrics(pet)

    report_lines = [
        "# NNDS Pipeline Validation Report",
        "",
        f"Generated: {pd.Timestamp.now().isoformat()}",
        "",
        "## Detection Quality",
        f"- Total detections: {det_metrics['total_detections']}",
        f"- Unique tracks: {det_metrics['unique_tracks']}",
        f"- Mean confidence: {det_metrics['mean_conf']:.3f}",
        f"- Confidence range: [{det_metrics['min_conf']:.3f}, {det_metrics['max_conf']:.3f}]",
        f"- Invalid bounding boxes: {det_metrics['invalid_boxes']}",
        f"- Class counts: {det_metrics['class_counts']}",
        "",
        "## Tracking Stability",
        f"- Total tracks: {track_metrics['total_tracks']}",
        f"- Mean track length: {track_metrics['mean_track_length']:.1f} frames",
        f"- Median track length: {track_metrics['median_track_length']:.1f} frames",
        f"- Max track length: {track_metrics['max_track_length']} frames",
        f"- Tracks with frame gap > 10: {track_metrics['tracks_with_gap_over']}",
        f"- Tracks with spatial jump > 50 px: {track_metrics['tracks_with_jump_over']}",
        f"- Fragmentation score: {track_metrics['fragmentation_score']:.3f} (lower is better)",
        "",
        "## BEV Homography Accuracy",
        f"- Rank: {bev_metrics_dict['rank']}",
        f"- Number of calibration points: {bev_metrics_dict['num_calibration_points']}",
        f"- Reprojection error (mean): {bev_metrics_dict['reprojection_error_mean']:.4f} ft",
        f"- Reprojection error (max): {bev_metrics_dict['reprojection_error_max']:.4f} ft",
        f"- Reprojection error (std): {bev_metrics_dict['reprojection_error_std']:.4f} ft",
        f"- Raw condition number: {bev_metrics_dict['condition_number_raw']:.2e} (not directly comparable due to scale)",
        "",
        "## PET Event Metrics",
        f"- Total events: {pet_metrics_dict['total_events']}",
        f"- Non-positive PET: {pet_metrics_dict['non_positive_pet']}",
        f"- PET range: [{pet_metrics_dict['min_pet']:.3f}, {pet_metrics_dict['max_pet']:.3f}] s",
        f"- PET median: {pet_metrics_dict['median_pet']:.3f} s",
        f"- PET median 95% CI: [{pet_metrics_dict['pet_median_ci_lower']:.3f}, {pet_metrics_dict['pet_median_ci_upper']:.3f}] s" if pet_metrics_dict['pet_median_ci_lower'] is not None else "",

        f"- PET mean: {pet_metrics_dict['mean_pet']:.3f} s",
        f"- PET std: {pet_metrics_dict['std_pet']:.3f} s",
        "",
        "## Interpretation",
        "- Low reprojection error (<0.1 ft) indicates accurate BEV mapping.",
        "- A low fragmentation score (<0.5) indicates stable tracking.",
        "- All PET values should be positive; median PET in typical intersection studies ranges from 0.5–3.0 s.",
        "",
    ]
    report = "\n".join(report_lines)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True)
    parser.add_argument("--pet", required=True)
    parser.add_argument("--bev-config", default="configs/bev_config.json")
    parser.add_argument("--calib", default="configs/giti_calibration_points.json")
    parser.add_argument("--output", default="outputs/validation_report.md")
    args = parser.parse_args()
    generate_report(args.detections, args.pet, args.bev_config, args.calib, args.output)


if __name__ == "__main__":
    main()