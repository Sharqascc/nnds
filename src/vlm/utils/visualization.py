"""
Visualization utilities for VLM analysis
=========================================

Functions for plotting and visualizing VLM results.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import json


def plot_vlm_results(
    results: List[Dict[str, Any]],
    output_path: str,
    title: str = "VLM Analysis Results"
) -> None:
    """
    Plot VLM analysis results.
    
    Args:
        results: List of VLM analysis result dictionaries
        output_path: Path to save plot
        title: Plot title
    """
    if not results:
        print("No results to plot")
        return
    
    # Extract data
    pets = [r.get("pet", 0) for r in results]
    severities = [r.get("severity", "UNKNOWN") for r in results]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # PET distribution
    ax1 = axes[0]
    ax1.hist(pets, bins=20, edgecolor='black', alpha=0.7)
    ax1.set_xlabel("PET (s)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("PET Distribution")
    ax1.axvline(x=1.0, color='r', linestyle='--', label='SEVERE threshold')
    ax1.axvline(x=1.5, color='orange', linestyle='--', label='HIGH threshold')
    ax1.legend()
    
    # Severity distribution
    ax2 = axes[1]
    severity_counts = {}
    for sev in severities:
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    
    colors = {'SEVERE': 'red', 'HIGH': 'orange', 'MODERATE': 'yellow', 'LOW': 'green'}
    labels = list(severity_counts.keys())
    values = [severity_counts[l] for l in labels]
    bar_colors = [colors.get(l, 'gray') for l in labels]
    
    ax2.bar(labels, values, color=bar_colors, edgecolor='black')
    ax2.set_ylabel("Count")
    ax2.set_title("Severity Distribution")
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {output_path}")


def create_heatmap(
    conflict_data: List[Dict[str, Any]],
    output_path: str,
    grid_size: int = 10
) -> None:
    """
    Create conflict heatmap from VLM analysis.
    
    Args:
        conflict_data: List of conflict data with location info
        output_path: Path to save heatmap
        grid_size: Size of grid cells
    """
    # Extract conflict locations (simplified - would need actual coordinates)
    # This is a placeholder for actual implementation
    print("Heatmap creation requires conflict location data")
