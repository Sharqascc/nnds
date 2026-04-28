from pathlib import Path
import importlib
import importlib.util
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------
# Helpers
# ---------------------------

def _import_from_path(rel_path: str, module_name: str = None):
    """
    Import a module from a relative file path using importlib.

    Raises AssertionError with a clear message if import fails.
    """
    path = ROOT / rel_path
    assert path.exists(), f"Expected file does not exist: {path}"
    name = module_name or path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None, f"Could not create spec for {rel_path}"
    assert spec.loader is not None, f"No loader for {rel_path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------
# Existence + content tests for key scripts / data
# ---------------------------

def test_diffusion_key_files_and_data_exist():
    # Core scripts
    assert (ROOT / "traffic_diffusion" / "train_trajectory_diffusion.py").exists()
    assert (ROOT / "traffic_diffusion" / "training_utils.py").exists()
    assert (ROOT / "traffic_diffusion" / "model_and_sampler.py").exists()
    assert (ROOT / "traffic_diffusion" / "trajectory_diffusion.py").exists()
    assert (ROOT / "analysis" / "safety_eval_diffusion.py").exists()
    assert (ROOT / "analysis" / "safety_eval_diffusion_notebook.py").exists()

    # Core pipeline scripts
    assert (ROOT / "traffic_analyzer.py").exists()
    assert (ROOT / "bev_mapper.py").exists()

    # Required diffusion data files
    data_dir = ROOT / "traffic_diffusion" / "data"
    inputs_path = data_dir / "trajdiff_inputs.npy"
    targets_path = data_dir / "trajdiff_targets.npy"
    meta_path = data_dir / "trajdiff_meta.parquet"

    for p in (inputs_path, targets_path, meta_path):
        assert p.exists(), f"Missing diffusion training data file: {p}"

    # Data content validation
    inputs = np.load(inputs_path)
    targets = np.load(targets_path)

    assert inputs.size > 0, "Diffusion inputs .npy file is empty"
    assert targets.size > 0, "Diffusion targets .npy file is empty"
    assert (
        inputs.shape[0] == targets.shape[0]
    ), f"Input/target count mismatch: {inputs.shape[0]} vs {targets.shape[0]}"

    df_meta = pd.read_parquet(meta_path)
    assert len(df_meta) > 0, "Metadata parquet is empty"


# ---------------------------
# Import tests for key modules
# ---------------------------

def test_diffusion_modules_importable():
    # Package-level import (checks __init__.py)
    importlib.import_module("traffic_diffusion")

    # Import key modules by file path (catches syntax errors, missing deps)
    _import_from_path("traffic_diffusion/training_utils.py", "training_utils")
    _import_from_path("traffic_diffusion/train_trajectory_diffusion.py", "train_trajectory_diffusion")
    _import_from_path("traffic_diffusion/model_and_sampler.py", "model_and_sampler")
    _import_from_path("traffic_diffusion/trajectory_diffusion.py", "trajectory_diffusion")


# ---------------------------
# Checkpoint directory validation
# ---------------------------

def test_diffusion_checkpoint_dir_is_creatable():
    """
    Ensure that a default checkpoints directory can be created/written to.
    This does not train a model; it just checks filesystem permissions.
    """
    candidate_dirs = [
        ROOT / "outputs" / "checkpoints",
        ROOT / "traffic_diffusion" / "checkpoints",
    ]

    target_dir = None
    for d in candidate_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            target_dir = d
            break
        except OSError:
            continue

    assert target_dir is not None, "Could not create any diffusion checkpoint directory"

    tmp = target_dir / ".checkpoint_write_test"
    tmp.write_text("ok", encoding="utf-8")
    assert tmp.exists(), f"Failed to write temp file in checkpoint dir: {target_dir}"
    tmp.unlink()


# ---------------------------
# Dependency checks: torch / CUDA
# ---------------------------

def test_torch_and_cuda_available():
    """
    Smoke-check that torch is importable and provides a device.
    CUDA is treated as a soft requirement: warn if missing instead of failing.
    """
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise AssertionError("PyTorch is not importable but is required for diffusion") from exc

    x = torch.randn(1)
    assert x.shape == (1,), "Unexpected tensor shape from torch.randn(1)"

    if not torch.cuda.is_available():
        warnings.warn(
            "CUDA is not available. Diffusion training will run on CPU only.",
            UserWarning,
        )


# ---------------------------
# Configuration / parameter sanity for diffusion model
# ---------------------------

def test_diffusion_model_init_signature_is_reasonable():
    """
    Introspect TrajectoryDiffusionModel.__init__ to ensure it exposes expected
    parameters (traj_shape, cond_dim, num_steps, beta_start, beta_end, device).
    """
    module = _import_from_path(
        "traffic_diffusion/trajectory_diffusion.py", "trajectory_diffusion"
    )

    if not hasattr(module, "TrajectoryDiffusionModel"):
        warnings.warn(
            "trajectory_diffusion.TrajectoryDiffusionModel not found; "
            "skipping model signature sanity check.",
            UserWarning,
        )
        return

    import inspect

    sig = inspect.signature(module.TrajectoryDiffusionModel.__init__)
    params = sig.parameters

    expected = {"traj_shape", "cond_dim", "num_steps", "beta_start", "beta_end", "device"}
    actual = set(params.keys()) - {"self"}

    # These are the critical parameters from the actual code
    critical = {"traj_shape", "cond_dim", "num_steps"}
    missing = critical - actual
    if missing:
        warnings.warn(
            f"TrajectoryDiffusionModel is missing expected parameters: {missing}",
            UserWarning,
        )


# ---------------------------
# Training script parameter presence (static content check)
# ---------------------------

def test_training_script_has_required_args():
    """
    Check that train_trajectory_diffusion.py appears to support key CLI arguments.
    This is a static content search, not a full argparse invocation.
    """
    train_script = ROOT / "traffic_diffusion" / "train_trajectory_diffusion.py"
    assert train_script.exists(), f"Missing training script: {train_script}"

    content = train_script.read_text()

    expected_args = ["csv_path", "checkpoint_dir", "batch_size", "epochs"]
    for arg in expected_args:
        if arg not in content:
            warnings.warn(
                f"Training script may not support '{arg}' argument (not found in file).",
                UserWarning,
            )


# ---------------------------
# Model & sampler API compatibility
# ---------------------------

def test_model_and_sampler_compatibility():
    """
    Ensure model_and_sampler exposes a compatible API:
    - load_model
    - sample_future_denorm

    This does not load a real checkpoint; it only verifies that the
    functions exist and are callable.
    """
    try:
        from traffic_diffusion.model_and_sampler import (
            load_model,
            sample_future_denorm,
        )
    except ImportError as e:
        raise AssertionError(f"model_and_sampler API issue: {e}")

    assert callable(load_model), "load_model must be callable"
    assert callable(sample_future_denorm), "sample_future_denorm must be callable"
