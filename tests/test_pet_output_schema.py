from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEMO_CSV = ROOT / "docs/data_samples/petevents_bev_demo.csv"

REQUIRED_COLUMNS = [
    "event_id",
    "pet",
    "pet_time_based",
    "frame",
    "track_a",
    "track_b",
    "orig_track_a",
    "seg_a",
    "orig_track_b",
    "seg_b",
    "conflict_type",
    "grid_cell",
    "track_a_entry_frame",
    "track_a_exit_frame",
    "track_a_exit_time_sec",
    "track_b_entry_frame",
    "track_b_entry_time_sec",
    "track_b_exit_frame",
    "world_traj_i",
    "world_traj_j",
    "traj_a_json",
    "traj_b_json",
    "video_source",
    "time_of_day_label",
    "gate_a_entry",
    "gate_b_entry",
]


def test_demo_csv_has_required_columns():
    assert DEMO_CSV.exists(), f"Missing {DEMO_CSV}"
    df = pd.read_csv(DEMO_CSV)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    assert not missing, f"Missing columns: {missing}"


def test_demo_csv_has_no_missing_ids_or_tracks():
    df = pd.read_csv(DEMO_CSV)
    assert not df["event_id"].isna().any(), "event_id contains NaN"
    assert not df["frame"].isna().any(), "frame contains NaN"
    assert not df["track_a"].isna().any(), "track_a contains NaN"
    assert not df["track_b"].isna().any(), "track_b contains NaN"


def test_demo_csv_conflict_type_allowed():
    df = pd.read_csv(DEMO_CSV)
    allowed = {"image_intersection", "UNKNOWN"}
    invalid = set(df["conflict_type"].dropna().unique()) - allowed
    assert not invalid, f"Invalid conflict_type values: {invalid}"


def test_demo_csv_world_traj_format():
    df = pd.read_csv(DEMO_CSV)
    for col in ["world_traj_i", "world_traj_j"]:
        non_null = df[col].dropna()
        assert all(isinstance(v, str) and v.startswith("track_") for v in non_null), (
            f"Invalid {col} values"
        )


def test_uvh_backend_populates_all_time_sec_columns():
    """Every _time_sec column the CLI CSV declares must be populated by uvh.

    Regression guard for the bug where track_b_exit_time_sec was requested
    by traffic_analyzer._events_to_dataframe but never written by
    uvh_coco_fused_grid_pet.run_uvh_coco_fused_grid_pet, so pandas filled
    it with NaN on every event.

    Static AST check: proves the row dict and the columns list agree.
    """
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / (
        "src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py"
    )
    tree = ast.parse(src.read_text())

    # Every key in the pet-event dict literal
    dict_keys: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
        if "track_b_exit_time_sec" in keys or "track_a_exit_time_sec" in keys:
            dict_keys.update(k for k in keys if isinstance(k, str))

    # Every column declared in pd.DataFrame(columns=[...])
    columns: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "DataFrame"):
            continue
        for kw in node.keywords:
            if kw.arg == "columns" and isinstance(kw.value, ast.List):
                for elt in kw.value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        columns.add(elt.value)

    # The three _time_sec fields the CLI actually wants
    required = {"track_a_exit_time_sec", "track_b_entry_time_sec", "track_b_exit_time_sec"}
    assert required.issubset(dict_keys), (
        f"uvh row dict missing _time_sec fields: {required - dict_keys}"
    )
    assert required.issubset(columns), (
        f"uvh columns list missing _time_sec fields: {required - columns}"
    )


def test_events_to_dataframe_propagates_b_exit_time_sec():
    """_events_to_dataframe must propagate track_b_exit_time_sec end-to-end.

    Regression guard for the PR #78 bug: the uvh backend populates the
    field, but _events_to_dataframe must also carry it through to the
    output DataFrame. If either side drops it, the CLI CSV column
    becomes NaN again (which was the original bug).
    """
    from src.pipeline.traffic_analyzer import _events_to_dataframe

    fps = 30.00853
    event = {
        "pet": 0.5,
        "frame": 100,
        "track_a": 1,
        "track_b": 2,
        "conflict_type": "side_swipe",
        "grid_cell": "CELL_A",
        "track_a_entry_frame": 80,
        "track_a_exit_frame": 95,
        "track_b_entry_frame": 100,
        "track_b_exit_frame": 150,
        "track_a_exit_time_sec": 95 / fps,
        "track_b_entry_time_sec": 100 / fps,
        "track_b_exit_time_sec": 150 / fps,
        "world_traj_i": "track_1",
        "world_traj_j": "track_2",
        "traj_a_json": "[]",
        "traj_b_json": "[]",
        "video_source": "test.mp4",
        "time_of_day_label": "unknown",
        "gate_a_entry": "unknown",
        "gate_b_entry": "unknown",
    }
    df = _events_to_dataframe([event])
    assert "track_b_exit_time_sec" in df.columns, (
        "_events_to_dataframe dropped track_b_exit_time_sec from the output"
    )
    assert not df["track_b_exit_time_sec"].isna().any(), (
        "track_b_exit_time_sec is NaN despite being present in the input event"
    )
    assert abs(df["track_b_exit_time_sec"].iloc[0] - (150 / fps)) < 1e-6, (
        f"track_b_exit_time_sec = {df['track_b_exit_time_sec'].iloc[0]}, expected {150 / fps}"
    )


def test_uvh_backend_computes_b_exit_time_sec_with_fps_divisor():
    """AST: track_b_exit_time_sec must be computed as float(b_exit / fps).

    The existing existence-only AST test catches field deletion, but not
    a wrong divisor (e.g., float(b_exit / 60.0) or float(b_exit / 25.0)
    instead of / fps). This locks the exact arithmetic expression, so
    any future refactor that changes the formula will fail here.
    """
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / (
        "src/analysis/grid_trajectory/uvh_coco_fused_grid_pet.py"
    )
    tree = ast.parse(src.read_text())

    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for k, v in zip(node.keys, node.values, strict=True):
            if isinstance(k, ast.Constant) and k.value == "track_b_exit_time_sec":
                found = True
                # Expected shape: float(b_exit / fps)
                assert isinstance(v, ast.Call), (
                    f"track_b_exit_time_sec: expected float(...), got {ast.unparse(v)}"
                )
                assert isinstance(v.func, ast.Name) and v.func.id == "float", (
                    f"expected float(...) call, got {ast.unparse(v.func)}"
                )
                assert len(v.args) == 1, (
                    f"expected exactly 1 positional arg to float(), got {len(v.args)}"
                )
                inner = v.args[0]
                assert isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Div), (
                    f"expected division expression, got {ast.unparse(inner)}"
                )
                assert isinstance(inner.left, ast.Name) and inner.left.id == "b_exit", (
                    f"expected b_exit as numerator, got {ast.unparse(inner.left)}"
                )
                assert isinstance(inner.right, ast.Name) and inner.right.id == "fps", (
                    f"expected fps as divisor, got {ast.unparse(inner.right)}"
                )
                break
        if found:
            break
    assert found, (
        "track_b_exit_time_sec not found in any dict literal in "
        "uvh_coco_fused_grid_pet.py — the PR #78 fix is missing"
    )
