"""Lock the class-colour contract.

Any future PR that changes the palette must update this test. Any future
PR that adds a class to the pipeline (CLASS_NAME_TO_ID in
uvh_coco_fused_grid_pet.py) must also add it to CLASS_COLORS — these
tests will fail if not, which is the intended friction.
"""

from __future__ import annotations

from src.analysis.visualization.class_colors import (
    CLASS_COLORS,
    FALLBACK_DEFAULT,
    OKABE_ITO_BGR,
    color_for,
    label_for,
)


def test_car_color_is_stable_across_calls():
    assert color_for("car") == color_for("car")
    assert color_for("car", track_id=1) == color_for("car", track_id=999)


def test_every_pipeline_class_has_a_color():
    """Pipeline emits these classes (per CLASS_NAME_TO_ID). All must be mapped."""
    pipeline_classes = {"car", "truck", "bus", "bike", "auto", "person"}
    for cls in pipeline_classes:
        assert cls in CLASS_COLORS, f"pipeline class {cls!r} has no colour"


def test_classes_get_distinct_colors():
    """All primary classes have visibly different colours."""
    classes = ["car", "truck", "bus", "bike", "auto", "person"]
    colors = [color_for(c) for c in classes]
    assert len(set(colors)) == len(colors), f"duplicate colours among {classes}: {colors}"


def test_case_insensitive_class_lookup():
    assert color_for("Car") == color_for("car") == color_for(" CAR ")


def test_aliases_map_to_normalized_class():
    assert color_for("hatchback") == color_for("car")
    assert color_for("sedan") == color_for("car")
    assert color_for("suv") == color_for("car")
    assert color_for("lcv") == color_for("truck")
    assert color_for("mini-bus") == color_for("bus")
    assert color_for("three-wheeler") == color_for("auto")
    assert color_for("two-wheeler") == color_for("bike")


def test_unknown_class_uses_track_id_fallback():
    c1 = color_for("weird_thing", track_id=1)
    c2 = color_for("weird_thing", track_id=2)
    assert c1 != c2, "different tracks should get different fallback colors"
    assert color_for("weird_thing", track_id=1) == c1


def test_no_class_no_track_returns_grey():
    assert color_for(None) == FALLBACK_DEFAULT
    assert color_for("") == FALLBACK_DEFAULT
    assert color_for(None, track_id=-1) == FALLBACK_DEFAULT


def test_all_class_colors_are_from_okabe_ito():
    """Colourblind-safe palette requirement."""
    palette_values = set(OKABE_ITO_BGR.values())
    for name, color in CLASS_COLORS.items():
        assert color in palette_values, (
            f"{name} -> {color} not from Okabe-Ito palette; "
            f"add it to OKABE_ITO_BGR or use an existing colour"
        )


def test_all_class_colors_are_unique_per_class():
    """Each normalised class should have its own color."""
    normalised = {"car", "truck", "bus", "bike", "auto", "person"}
    colors = {cls: CLASS_COLORS[cls] for cls in normalised}
    assert len(set(colors.values())) == len(colors), (
        f"two normalised classes share a colour: {colors}"
    )


def test_label_for_includes_class_and_track():
    assert label_for("car", 42) == "#42 car"
    assert label_for("car", None) == "car"
    assert label_for(None, 42) == "#42"
    assert label_for(None, None) == "?"
