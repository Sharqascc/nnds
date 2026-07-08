"""
Canonical PET severity color/label scheme, shared across all visualization
modules so the same color always means the same thing in every figure that
might appear in the same paper. Previously industry_standard_viz.py,
pet_event_plots.py, and video_overlays.py each defined their own mapping,
and they disagreed with each other (e.g. green meant "safe" in one file and
"slight" in another). This module is now the single source of truth.
"""
from typing import Dict, Optional
import matplotlib.patches as mpatches

# Colorblind-safe palette (Okabe-Ito)
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "yellow": "#F0E442",
    "purple": "#CC79A7",
    "cyan": "#56B4E9",
    "red": "#D55E00",
    "black": "#000000",
}

DEFAULT_PET_THRESHOLDS = {
    "critical": 0.5,
    "serious": 1.0,
    "moderate": 1.5,
    "safe": 5.0,
}


def get_severity_color(pet_value: float, thresholds: Optional[Dict[str, float]] = None) -> str:
    """5-tier severity color for a PET value: Critical/Serious/Moderate/Slight/Safe."""
    t = thresholds or DEFAULT_PET_THRESHOLDS
    if pet_value < t["critical"]:
        return COLORS["red"]
    elif pet_value < t["serious"]:
        return COLORS["orange"]
    elif pet_value < t["moderate"]:
        return COLORS["yellow"]
    elif pet_value < t["safe"]:
        return COLORS["green"]
    else:
        return COLORS["blue"]


def get_severity_label(pet_value: float, thresholds: Optional[Dict[str, float]] = None) -> str:
    """5-tier severity label for a PET value, matching get_severity_color exactly."""
    t = thresholds or DEFAULT_PET_THRESHOLDS
    if pet_value < t["critical"]:
        return "Critical"
    elif pet_value < t["serious"]:
        return "Serious"
    elif pet_value < t["moderate"]:
        return "Moderate"
    elif pet_value < t["safe"]:
        return "Slight"
    else:
        return "Safe"


def severity_legend_patches(thresholds: Optional[Dict[str, float]] = None):
    """Legend patches guaranteed to match get_severity_color's actual output,
    so a figure's legend can never promise a color that isn't really used."""
    t = thresholds or DEFAULT_PET_THRESHOLDS
    return [
        mpatches.Patch(color=COLORS["red"], label=f'Critical (<{t["critical"]}s)'),
        mpatches.Patch(color=COLORS["orange"], label=f'Serious ({t["critical"]}-{t["serious"]}s)'),
        mpatches.Patch(color=COLORS["yellow"], label=f'Moderate ({t["serious"]}-{t["moderate"]}s)'),
        mpatches.Patch(color=COLORS["green"], label=f'Slight ({t["moderate"]}-{t["safe"]}s)'),
        mpatches.Patch(color=COLORS["blue"], label=f'Safe (>{t["safe"]}s)'),
    ]
