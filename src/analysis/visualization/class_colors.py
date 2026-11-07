"""Color palette for vehicle / road-user classes.

Single source of truth for how classes are colored in rendered video
and in any other visualisation that draws bounding boxes.

Palette is Okabe-Ito, which is designed to be distinguishable by
viewers with deuteranopia, protanopia, or tritanopia (the three most
common forms of colour-vision deficiency).

Contract: **each class gets one fixed colour, everywhere, forever.**
No track-ID dependence for known classes. Do not add per-call colour
logic elsewhere; add the class or alias here.

To change a colour: edit CLASS_COLORS below. The test suite
(tests/test_class_colors.py) will assert every pipeline class is
mapped and every colour is from the Okabe-Ito palette.

Usage:

    from src.analysis.visualization.class_colors import color_for, label_for
    bgr = color_for("car", track_id=42)          # -> (233, 180, 86)
    txt = label_for("car", track_id=42)          # -> "#42 car"
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Okabe-Ito colourblind-safe palette, in BGR (OpenCV convention).
# Reference: https://jfly.uni-koeln.de/color/
# ---------------------------------------------------------------------------
OKABE_ITO_BGR: dict[str, tuple[int, int, int]] = {
    "orange": (0, 159, 230),
    "sky_blue": (233, 180, 86),
    "bluish_green": (115, 158, 0),
    "yellow": (66, 228, 240),
    "blue": (178, 114, 0),
    "vermillion": (0, 94, 213),
    "reddish_purple": (167, 121, 204),
}

# ---------------------------------------------------------------------------
# Class -> BGR colour. This is the single source of truth.
# ---------------------------------------------------------------------------
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    # Normalised names (what the pipeline emits per CLASS_NAME_TO_ID)
    "car": OKABE_ITO_BGR["sky_blue"],
    "truck": OKABE_ITO_BGR["orange"],
    "bus": OKABE_ITO_BGR["bluish_green"],
    "bike": OKABE_ITO_BGR["vermillion"],
    "auto": OKABE_ITO_BGR["reddish_purple"],
    "person": OKABE_ITO_BGR["yellow"],
    # Other CLASS_NAME_TO_ID names — pipeline-visible class names that
    # are not covered by the raw-detector aliases below.
    "pedestrian": OKABE_ITO_BGR["yellow"],      # CLASS_NAME_TO_ID: 0 (same class as person)
    "motorcycle": OKABE_ITO_BGR["vermillion"],  # CLASS_NAME_TO_ID: 3 (same class as bike)
    "bicycle":    OKABE_ITO_BGR["blue"],        # CLASS_NAME_TO_ID: 1 (distinct class)
    # Raw detector names -> same colour as their normalised class
    "hatchback": OKABE_ITO_BGR["sky_blue"],
    "sedan": OKABE_ITO_BGR["sky_blue"],
    "suv": OKABE_ITO_BGR["sky_blue"],
    "muv": OKABE_ITO_BGR["sky_blue"],
    "van": OKABE_ITO_BGR["sky_blue"],
    "lcv": OKABE_ITO_BGR["orange"],
    "mini-bus": OKABE_ITO_BGR["bluish_green"],
    "tempo-traveller": OKABE_ITO_BGR["bluish_green"],
    "three-wheeler": OKABE_ITO_BGR["reddish_purple"],
    "two-wheeler": OKABE_ITO_BGR["vermillion"],
}

# ---------------------------------------------------------------------------
# Fallback for unknown classes. Track-id-based, distinct.
# ---------------------------------------------------------------------------
_FALLBACK_PALETTE: list[tuple[int, int, int]] = [
    (200, 200, 200),  # grey
    (128, 128, 255),  # salmon
    (128, 255, 128),  # mint
    (255, 128, 128),  # periwinkle
]
FALLBACK_DEFAULT = _FALLBACK_PALETTE[0]


def color_for(cls_name: str | None, track_id: int | None = None) -> tuple[int, int, int]:
    """Return the BGR colour for a class.

    Known class -> fixed class colour. Unknown class -> track-id fallback
    from ``_FALLBACK_PALETTE``. No class and no track -> grey.
    """
    if cls_name:
        key = str(cls_name).strip().lower()
        if key in CLASS_COLORS:
            return CLASS_COLORS[key]
    if track_id is not None and track_id >= 0:
        return _FALLBACK_PALETTE[track_id % len(_FALLBACK_PALETTE)]
    return FALLBACK_DEFAULT


def label_for(cls_name: str | None, track_id: int | None = None) -> str:
    """Return the on-frame label: '#42 car', 'car', '#42', or '?'."""
    parts = []
    if track_id is not None and track_id >= 0:
        parts.append(f"#{track_id}")
    if cls_name:
        parts.append(str(cls_name))
    return " ".join(parts) or "?"
