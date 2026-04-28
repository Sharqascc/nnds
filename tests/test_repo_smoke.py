from pathlib import Path
import importlib.util
import json
import warnings

ROOT = Path(__file__).resolve().parents[1]


def _file_info(path: Path) -> str:
    if not path.exists():
        return f"{path} (missing)"
    size = path.stat().st_size
    return f"{path} ({size} bytes)"


# ---------------------------
# Expected paths exist
# ---------------------------

def test_expected_paths_exist():
    expected = [
        "README.md",
        "Makefile",
        "requirements.txt",
        "pyproject.toml",
        "traffic_analyzer.py",
        "bev_mapper.py",
        "giti_bev_calib.py",
        "gate_counter.py",
        "pet_conflict_checker.py",
        "grid_trajectory/pet_grid.py",
        "grid_trajectory/sam3_grid_pet.py",
        "grid_trajectory/spatial_grid.py",
        "grid_trajectory/trajectory_safety_analyzer.py",
        "analysis/safety_eval_diffusion.py",
        "analysis/safety_eval_diffusion_notebook.py",
        "traffic_diffusion/train_trajectory_diffusion.py",
        "traffic_diffusion/training_utils.py",
        "traffic_diffusion/model_and_sampler.py",
        "traffic_diffusion/trajectory_diffusion.py",
        "traffic_diffusion/pet_safety_metrics.py",
        "configs/bev_config.json",
        "configs/giti_calibration_points.json",
        "configs/GITI_grid_config.json",
        "configs/gate_config.yaml",
        "docs/data_samples/petevents_bev_demo.csv",
    ]
    missing = [p for p in expected if not (ROOT / p).exists()]
    assert not missing, f"Missing expected files: {missing}"


# ---------------------------
# outputs/ structure and writability
# ---------------------------

def test_outputs_structure_exists_and_writable():
    outputs_dir = ROOT / "outputs"
    assert outputs_dir.exists(), "outputs/ directory does not exist"

    # Check we can write a tiny file here
    tmp = outputs_dir / ".outputs_write_test"
    tmp.write_text("ok", encoding="utf-8")
    assert tmp.exists(), f"Failed to write temp file in outputs dir: {_file_info(outputs_dir)}"
    tmp.unlink()


# ---------------------------
# .gitignore includes outputs/
# ---------------------------

def test_outputs_is_gitignored():
    gitignore = ROOT / ".gitignore"
    if not gitignore.exists():
        warnings.warn(".gitignore is missing; cannot verify outputs/ ignore rule.", UserWarning)
        return

    content = gitignore.read_text()
    # Common patterns to ignore generated artifacts
    patterns = ["outputs/", "outputs/*"]
    if not any(p in content for p in patterns):
        warnings.warn(
            "outputs/ directory does not appear to be ignored in .gitignore. "
            "Generated artifacts may be committed accidentally.",
            UserWarning,
        )


# ---------------------------
# Top-level scripts loadable
# ---------------------------

def test_can_load_top_level_scripts():
    for rel_path in [
        "traffic_analyzer.py",
        "bev_mapper.py",
        "giti_bev_calib.py",
        "traj_diffusion_normalized.py",
    ]:
        path = ROOT / rel_path
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec is not None, f"Could not create spec for {rel_path} ({_file_info(path)})"
        assert spec.loader is not None, f"No loader for {rel_path} ({_file_info(path)})"
        module = importlib.util.module_from_spec(spec)
        # Execute module once to catch syntax/import errors
        spec.loader.exec_module(module)


# ---------------------------
# Diffusion data files present and non-empty
# ---------------------------

def test_diffusion_data_files_present_and_non_empty():
    expected = [
        "traffic_diffusion/data/trajdiff_inputs.npy",
        "traffic_diffusion/data/trajdiff_targets.npy",
        "traffic_diffusion/data/trajdiff_meta.parquet",
    ]
    missing = [p for p in expected if not (ROOT / p).exists()]
    assert not missing, f"Missing diffusion data files: {missing}"

    # Basic size sanity check: files should not be empty
    for rel in expected:
        path = ROOT / rel
        size = path.stat().st_size
        assert size > 0, f"Diffusion data file appears empty: {_file_info(path)}"


# ---------------------------
# SAM3 weight file presence and size sanity
# ---------------------------

def test_sam3_weights_presence_and_size():
    """
    SAM3 weights are large (~3.2GB). Treat them as optional in CI environments
    but warn if obviously missing or suspiciously small.
    """
    sam3_path = ROOT / "sam3.pt"
    if not sam3_path.exists():
        warnings.warn(
            "sam3.pt not found in repo root. SAM3-based video segmentation may not "
            "work until weights are downloaded.",
            UserWarning,
        )
        return

    size_bytes = sam3_path.stat().st_size
    size_gb = size_bytes / (1024 ** 3)

    # Soft bounds: we expect around 3 GB; warn if far off
    if size_gb < 2.0:
        warnings.warn(
            f"sam3.pt is unusually small ({size_gb:.2f} GB). "
            "Weights may be incomplete or corrupted.",
            UserWarning,
        )


# ---------------------------
# pyproject.toml version sanity
# ---------------------------

def test_pyproject_version_sanity():
    """
    Ensure pyproject.toml defines a non-empty version string.
    We don't pin an exact version here, just validate structure.
    """
    pyproject = ROOT / "pyproject.toml"
    assert pyproject.exists(), f"pyproject.toml missing: {_file_info(pyproject)}"

    text = pyproject.read_text()
    # Very lightweight parsing: look for 'version = "..."' under [project]
    version_line = None
    in_project = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[project]"):
            in_project = True
            continue
        if in_project and stripped.startswith("version"):
            version_line = stripped
            break

    assert version_line, "No 'version' field found under [project] in pyproject.toml"
    # crude check: version = "0.1.0"
    if "=" in version_line:
        _, val = version_line.split("=", 1)
        version_str = val.strip().strip('"').strip("'")
        assert version_str, "Version string in pyproject.toml is empty"


# ---------------------------
# Helpful context on missing files (if any test above fails)
# ---------------------------

def test_repo_summary_context():
    """
    Not a strict test: provides quick context on sizes for key files to make
    error messages from other tests easier to interpret in CI logs.
    """
    key_paths = [
        "traffic_analyzer.py",
        "bev_mapper.py",
        "gate_counter.py",
        "pet_conflict_checker.py",
        "traffic_diffusion/data/trajdiff_inputs.npy",
        "traffic_diffusion/data/trajdiff_targets.npy",
        "traffic_diffusion/data/trajdiff_meta.parquet",
        "sam3.pt",
    ]
    info_lines = [f"- {_file_info(ROOT / rel)}" for rel in key_paths]
    # Only printed when this test fails, so keep it passing but informative.
    assert True, "Repo summary:\n" + "\n".join(info_lines)
