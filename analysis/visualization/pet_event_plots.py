"""
Publication-Quality PET Conflict Event Visualization

Individual conflict event plotting with:
- Colorblind-safe trajectory rendering
- Severity-based visualization (critical/serious/moderate/safe)
- High-resolution output (300 DPI + PDF)
- Batch plotting capabilities
- Conflict zone visualization (fixed radius)
- Statistical annotations
- Velocity vectors (corrected direction)

Compliant with:
- IEEE/TRB figure guidelines
- Accessibility standards (colorblind-safe)
- Journal submission requirements
"""

import os
import ast
from typing import Optional, List, Callable, Dict
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle


__all__ = [
    'EventPlotter',
    'load_pet_csv',
    'compute_timing_from_traj',
    'plot_conflict_event',
    'plot_multiple_events',
    'get_class_default'
]


# Colorblind-safe palette (Okabe-Ito - consistent with industry_standard_viz)
COLORS = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'green': '#009E73',
    'yellow': '#F0E442',
    'purple': '#CC79A7',
    'cyan': '#56B4E9',
    'red': '#D55E00',
    'black': '#000000'
}


# Safety thresholds (FHWA-based, configurable)
DEFAULT_THRESHOLDS = {
    'critical': 0.5,   # < 0.5s = critical
    'serious': 1.0,    # 0.5-1.0s = serious
    'moderate': 1.5,   # 1.0-1.5s = moderate
    'safe': 5.0        # > 5.0s = safe
}


def load_pet_csv(csv_path: str) -> pd.DataFrame:
    """
    Load PET CSV and parse world trajectories into Python lists.

    Args:
        csv_path: Path to PET events CSV file

    Returns:
        DataFrame with parsed trajectory columns
    """
    df = pd.read_csv(csv_path)
    df["traj_i"] = df["world_traj_i"].apply(ast.literal_eval)
    df["traj_j"] = df["world_traj_j"].apply(ast.literal_eval)
    return df


def compute_timing_from_traj(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add approximate timing info (for visualization only) to PET dataframe.

    Computes:
    - t_closest: Time of closest approach
    - t_leave_i: Time when track_i leaves conflict zone
    - t_enter_j: Time when track_j enters conflict zone
    - pet_approx: Approximate PET from trajectories
    - dist_min: Minimum distance between trajectories
    - k_closest: Index of closest approach

    Args:
        df: DataFrame with traj_i and traj_j columns

    Returns:
        DataFrame with added timing columns
    """
    def closest_approach_times(traj_i, traj_j):
        ti, xi, yi = zip(*traj_i)
        tj, xj, yj = zip(*traj_j)

        ti = np.array(ti); xi = np.array(xi); yi = np.array(yi)
        tj = np.array(tj); xj = np.array(xj); yj = np.array(yj)

        T = min(len(ti), len(tj))
        ti_c = ti[:T]; xi_c = xi[:T]; yi_c = yi[:T]
        tj_c = tj[:T]; xj_c = xj[:T]; yj_c = yj[:T]

        dist = np.hypot(xi_c - xj_c, yi_c - yj_c)
        k_min = int(np.argmin(dist))

        t_closest = float(ti_c[k_min])
        k_leave = max(0, k_min - 1)
        k_enter = min(T - 1, k_min + 1)

        t_leave_i = float(ti_c[k_leave])
        t_enter_j = float(tj_c[k_enter])

        return {
            "t_closest": t_closest,
            "t_leave_i": t_leave_i,
            "t_enter_j": t_enter_j,
            "pet_approx": float(t_enter_j - t_leave_i),
            "dist_min": float(dist[k_min]),
            "k_closest": k_min,
        }

    timing = df.apply(
        lambda r: closest_approach_times(r["traj_i"], r["traj_j"]),
        axis=1,
        result_type="expand",
    )
    return pd.concat([df, timing], axis=1)


def get_class_default(track_id: int) -> str:
    """Default class mapper: everything is just 'vehicle'."""
    return "vehicle"


class EventPlotter:
    """
    Publication-quality PET conflict event plotter.

    Features:
    - Colorblind-safe palettes
    - Severity-based coloring
    - High-resolution output (300 DPI)
    - PDF export for publications
    - Batch plotting
    - Conflict zone visualization with fixed radius
    - Velocity vectors with correct forward direction
    """

    def __init__(
        self,
        dpi: int = 300,
        style: str = 'journal',
        colorblind_safe: bool = True,
        thresholds: Optional[Dict[str, float]] = None,
        font_size: float = 10.0,
        conflict_zone_radius: float = 2.0,
        arrow_scale: float = 3.0
    ):
        """
        Args:
            dpi: Resolution for saved figures (300 for print)
            style: 'journal' or 'presentation'
            colorblind_safe: Use Okabe-Ito palette
            thresholds: Custom PET thresholds dict
            font_size: Base font size in points
            conflict_zone_radius: Radius of conflict zone circle (meters)
            arrow_scale: Scaling factor for velocity arrows
        """
        self.dpi = dpi
        self.style = style
        self.colorblind_safe = colorblind_safe
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.font_size = font_size
        self.conflict_zone_radius = conflict_zone_radius
        self.arrow_scale = arrow_scale

        self._setup_style()

    def _setup_style(self):
        """Configure matplotlib for publication quality."""
        if self.style == 'journal':
            plt.rcParams.update({
                'font.size': self.font_size,
                'axes.titlesize': self.font_size + 2,
                'axes.labelsize': self.font_size + 1,
                'xtick.labelsize': self.font_size - 1,
                'ytick.labelsize': self.font_size - 1,
                'legend.fontsize': self.font_size - 1,
                'figure.dpi': self.dpi,
                'savefig.dpi': self.dpi,
                'axes.grid': True,
                'grid.alpha': 0.3,
            })

    def _get_severity_color(self, pet_value: float) -> str:
        """Get color based on PET severity."""
        if pet_value < self.thresholds['critical']:
            return COLORS['red']
        elif pet_value < self.thresholds['serious']:
            return COLORS['orange']
        elif pet_value < self.thresholds['moderate']:
            return COLORS['yellow']
        elif pet_value < self.thresholds['safe']:
            return COLORS['green']
        else:
            return COLORS['blue']

    def _get_severity_label(self, pet_value: float) -> str:
        """Get severity category label."""
        if pet_value < self.thresholds['critical']:
            return 'Critical'
        elif pet_value < self.thresholds['serious']:
            return 'Serious'
        elif pet_value < self.thresholds['moderate']:
            return 'Moderate'
        elif pet_value < self.thresholds['safe']:
            return 'Slight'
        else:
            return 'Safe'

    def plot_conflict_event(
        self,
        df: pd.DataFrame,
        event_id: int,
        class_mapper: Optional[Callable] = None,
        save_path: Optional[str] = None,
        show_conflict_zone: bool = True,
        show_velocities: bool = False,
        save_pdf: bool = True
    ) -> plt.Figure:
        """
        Plot BEV trajectories and PET info for a single conflict event.

        Args:
            df: DataFrame with PET events
            event_id: Event ID to plot
            class_mapper: Function to map track_id -> class name
            save_path: Path to save figure (PNG + PDF)
            show_conflict_zone: Shade the conflict area
            show_velocities: Show velocity vectors (forward direction)
            save_pdf: Save PDF version alongside PNG

        Returns:
            Matplotlib figure
        """
        if class_mapper is None:
            class_mapper = get_class_default

        # Get event data
        row = df.loc[df["event_id"] == event_id].iloc[0]

        traj_i = row["traj_i"]
        traj_j = row["traj_j"]

        ti, xi, yi = zip(*traj_i)
        tj, xj, yj = zip(*traj_j)

        ti = np.array(ti); xi = np.array(xi); yi = np.array(yi)
        tj = np.array(tj); xj = np.array(xj); yj = np.array(yj)

        class_i = class_mapper(int(row["track_a"]))
        class_j = class_mapper(int(row["track_b"]))

        pet = float(row["pet"])
        pet_approx = float(row.get("pet_approx", pet))
        cell = row["conflict_type"]
        t_leave_i = float(row.get("t_leave_i", 0))
        t_enter_j = float(row.get("t_enter_j", 0))

        # Get severity info
        severity_color = self._get_severity_color(pet)
        severity_label = self._get_severity_label(pet)

        # Create figure
        fig, ax = plt.subplots(figsize=(7, 7), dpi=self.dpi)

        # Plot trajectories with colorblind-safe colors
        ax.plot(
            xi, yi, "-o",
            color=COLORS['blue'],
            linewidth=2,
            markersize=4,
            label=f"Track {int(row['track_a'])} ({class_i})",
            alpha=0.8
        )
        ax.plot(
            xj, yj, "-o",
            color=COLORS['orange'],
            linewidth=2,
            markersize=4,
            label=f"Track {int(row['track_b'])} ({class_j})",
            alpha=0.8
        )

        # Mark closest approach
        T = min(len(ti), len(tj))
        xi_c = xi[:T]; yi_c = yi[:T]
        xj_c = xj[:T]; yj_c = yj[:T]

        dist = np.hypot(xi_c - xj_c, yi_c - yj_c)
        k_closest = int(np.argmin(dist))

        ax.scatter(
            [xi_c[k_closest]], [yi_c[k_closest]],
            c=severity_color, marker="X", s=200,
            edgecolors='black', linewidths=2,
            label="Closest Approach", zorder=10
        )

        # FIXED: Conflict zone with fixed radius (not distance-based)
        if show_conflict_zone:
            conflict_x = xi_c[k_closest]
            conflict_y = yi_c[k_closest]

            circle = Circle(
                (conflict_x, conflict_y),
                self.conflict_zone_radius,  # Fixed radius in meters
                color=severity_color,
                alpha=0.15,
                label=f'Conflict Zone ({self.conflict_zone_radius}m)',
                zorder=1
            )
            ax.add_patch(circle)

        # FIXED: Velocity vectors with FORWARD direction (t+1 - t)
        if show_velocities:
            # Velocity of track i (forward direction)
            if k_closest < len(xi) - 1:
                vx_i = xi[k_closest + 1] - xi[k_closest]
                vy_i = yi[k_closest + 1] - yi[k_closest]

                # Normalize and scale
                vel_mag_i = np.hypot(vx_i, vy_i)
                if vel_mag_i > 1e-6:
                    scale_i = self.arrow_scale / vel_mag_i
                    ax.arrow(
                        xi_c[k_closest], yi_c[k_closest],
                        vx_i * scale_i, vy_i * scale_i,
                        head_width=0.3, head_length=0.2,
                        fc=COLORS['blue'], ec=COLORS['blue'],
                        alpha=0.6, linewidth=2, zorder=8
                    )

            # Velocity of track j (forward direction)
            if k_closest < len(xj) - 1:
                vx_j = xj[k_closest + 1] - xj[k_closest]
                vy_j = yj[k_closest + 1] - yj[k_closest]

                # Normalize and scale
                vel_mag_j = np.hypot(vx_j, vy_j)
                if vel_mag_j > 1e-6:
                    scale_j = self.arrow_scale / vel_mag_j
                    ax.arrow(
                        xj_c[k_closest], yj_c[k_closest],
                        vx_j * scale_j, vy_j * scale_j,
                        head_width=0.3, head_length=0.2,
                        fc=COLORS['orange'], ec=COLORS['orange'],
                        alpha=0.6, linewidth=2, zorder=8
                    )

        # Start/end markers
        ax.scatter([xi[0]], [yi[0]], c=COLORS['blue'], marker='s', s=100, 
                  edgecolors='black', linewidths=1.5, zorder=5, label='Start i')
        ax.scatter([xj[0]], [yj[0]], c=COLORS['orange'], marker='s', s=100,
                  edgecolors='black', linewidths=1.5, zorder=5, label='Start j')

        # Labels and styling
        ax.set_xlabel("X (world / BEV coordinates)", fontsize=self.font_size + 1)
        ax.set_ylabel("Y (world / BEV coordinates)", fontsize=self.font_size + 1)

        # Title with severity color coding
        title_text = (
            f"Conflict Event {event_id} – {class_i} vs {class_j}\n"
            f"Cell: {cell} | PET: {pet:.3f}s ({severity_label}) | "
            f"Min Distance: {dist[k_closest]:.2f}m"
        )
        ax.set_title(
            title_text,
            fontsize=self.font_size + 2,
            fontweight='bold',
            pad=15,
            color=severity_color
        )

        # Statistics box
        stats_text = (
            f"PET = {pet:.3f} s\n"
            f"Severity: {severity_label}\n"
            f"Min Dist = {dist[k_closest]:.2f} m\n"
            f"t_leave_i = {t_leave_i:.2f} s\n"
            f"t_enter_j = {t_enter_j:.2f} s"
        )

        ax.text(
            0.02, 0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=self.font_size - 1,
            verticalalignment='top',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='white',
                edgecolor=severity_color,
                linewidth=2,
                alpha=0.9
            )
        )

        ax.legend(loc='upper right', fontsize=self.font_size - 1, framealpha=0.9)
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", "box")

        plt.tight_layout()

        # Save
        if save_path:
            self._save_figure(fig, save_path, save_pdf)

        return fig

    def plot_multiple_events(
        self,
        df: pd.DataFrame,
        event_ids: List[int],
        class_mapper: Optional[Callable] = None,
        save_dir: str = 'outputs/conflict_events',
        save_pdf: bool = True
    ):
        """
        Batch plot multiple conflict events.

        Args:
            df: DataFrame with PET events
            event_ids: List of event IDs to plot
            class_mapper: Function to map track_id -> class name
            save_dir: Directory to save all plots
            save_pdf: Save PDF versions
        """
        os.makedirs(save_dir, exist_ok=True)

        print(f"🎨 Plotting {len(event_ids)} conflict events...")

        for i, event_id in enumerate(event_ids, 1):
            save_path = os.path.join(save_dir, f'event_{event_id:04d}.png')

            try:
                self.plot_conflict_event(
                    df, event_id,
                    class_mapper=class_mapper,
                    save_path=save_path,
                    save_pdf=save_pdf
                )
                plt.close()  # Close to free memory

                if i % 10 == 0:
                    print(f"  Progress: {i}/{len(event_ids)} events plotted")

            except Exception as e:
                warnings.warn(f"Failed to plot event {event_id}: {e}")

        print(f"✅ All plots saved to: {save_dir}/")
        if save_pdf:
            print(f"📄 PDF versions also saved")

    def _save_figure(self, fig: plt.Figure, save_path: str, save_pdf: bool = True):
        """Save figure in PNG and optionally PDF."""
        dirname = os.path.dirname(save_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        # Save PNG
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight", format='png')

        # Save PDF
        if save_pdf:
            pdf_path = save_path.replace('.png', '.pdf')
            if pdf_path != save_path:
                fig.savefig(pdf_path, bbox_inches="tight", format='pdf')


# ===================================================================
# CONVENIENCE FUNCTIONS
# ===================================================================

def plot_conflict_event(
    df: pd.DataFrame,
    event_id: int,
    class_mapper: Optional[Callable] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
    save_pdf: bool = True
) -> plt.Figure:
    """
    Standalone function to plot a single conflict event.

    Args:
        df: DataFrame with PET events
        event_id: Event ID to plot
        class_mapper: Function to map track_id -> class name
        save_path: Path to save figure
        dpi: Resolution
        save_pdf: Save PDF version

    Returns:
        Matplotlib figure
    """
    plotter = EventPlotter(dpi=dpi)
    return plotter.plot_conflict_event(
        df, event_id,
        class_mapper=class_mapper,
        save_path=save_path,
        save_pdf=save_pdf
    )


def plot_multiple_events(
    df: pd.DataFrame,
    event_ids: List[int],
    class_mapper: Optional[Callable] = None,
    save_dir: str = 'outputs/conflict_events',
    dpi: int = 300,
    save_pdf: bool = True
):
    """
    Batch plot multiple conflict events.

    Args:
        df: DataFrame with PET events
        event_ids: List of event IDs to plot
        class_mapper: Function to map track_id -> class name
        save_dir: Directory to save all plots
        dpi: Resolution
        save_pdf: Save PDF versions
    """
    plotter = EventPlotter(dpi=dpi)
    plotter.plot_multiple_events(
        df, event_ids,
        class_mapper=class_mapper,
        save_dir=save_dir,
        save_pdf=save_pdf
    )
