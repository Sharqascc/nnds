"""Industry-Standard Safety Metrics Visualization.

Provides publication-quality visualization for surrogate safety metrics
following transportation research best practices.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'


@dataclass
class SafetyMetricsData:
    """Container for safety metrics visualization data."""
    frame_id: int
    track_id: int
    x: float
    y: float
    vx: float
    vy: float
    bbox: Tuple[float, float, float, float]


class IndustryStandardSafetyViz:
    """Industry-standard visualization for surrogate safety metrics.
    
    Generates publication-quality plots following transportation
    research visualization conventions.
    """
    
    PALETTE = {
        'safe': '#2E8B57',
        'warning': '#FFA500',
        'critical': '#DC143C',
        'vehicle': '#4682B4',
        'pedestrian': '#9370DB',
        'trajectory': '#2F4F4F',
    }
    
    THRESHOLDS = {
        'pet_safe': 5.0,
        'pet_warning': 2.5,
        'ttc_safe': 5.0,
        'ttc_warning': 2.5,
    }
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6)):
        self.figsize = figsize
    
    def plot_pet_distribution(
        self,
        pet_values: np.ndarray,
        title: str = "PET Distribution",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot PET value distribution with threshold bands.
        
        Args:
            pet_values: Array of PET values
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        pet_values = np.asarray(pet_values)
        pet_values = pet_values[~np.isnan(pet_values)]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        n, bins, patches = ax.hist(
            pet_values, bins=30, edgecolor='black',
            alpha=0.7, color=self.PALETTE['vehicle']
        )
        
        for patch in patches:
            x = patch.get_x()
            if x >= self.THRESHOLDS['pet_safe']:
                patch.set_facecolor(self.PALETTE['safe'])
            elif x >= self.THRESHOLDS['pet_warning']:
                patch.set_facecolor(self.PALETTE['warning'])
            else:
                patch.set_facecolor(self.PALETTE['critical'])
        
        ax.axvline(self.THRESHOLDS['pet_safe'], color=self.PALETTE['safe'],
                   linestyle='--', linewidth=2, label='Safe threshold')
        ax.axvline(self.THRESHOLDS['pet_warning'], color=self.PALETTE['warning'],
                   linestyle='--', linewidth=2, label='Warning threshold')
        
        ax.set_xlabel('Post-Encroachment Time (s)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_ttc_heatmap(
        self,
        ttc_matrix: np.ndarray,
        title: str = "TTC Heatmap",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot TTC heatmap for conflict analysis.
        
        Args:
            ttc_matrix: 2D array of TTC values
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        im = ax.imshow(
            ttc_matrix, cmap='RdYlGn_r',
            vmin=0, vmax=self.THRESHOLDS['ttc_safe']
        )
        
        cbar = plt.colorbar(im, ax=ax, label='TTC (s)')
        
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5)
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5)
        
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Trajectory', fontsize=12)
        ax.set_title(title, fontsize=14)
        
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_conflict_timeline(
        self,
        events: List[Dict],
        title: str = "Conflict Timeline",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot conflict events on a timeline.
        
        Args:
            events: List of event dicts with 'time', 'type', 'severity'
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        times = [e.get('time', 0) for e in events]
        severities = [e.get('severity', 'safe') for e in events]
        
        colors = [self.PALETTE.get(s, self.PALETTE['vehicle']) for s in severities]
        sizes = [100 if s == 'critical' else 50 for s in severities]
        
        ax.scatter(times, [0] * len(times), s=sizes, c=colors,
                   alpha=0.7, edgecolors='black')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_yticks([])
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
