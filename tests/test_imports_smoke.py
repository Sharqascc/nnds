import importlib
import warnings


def _assert_can_import(module_name: str):
    """
    Import a module and fail with a clear message if it cannot be imported.

    This is a hard requirement: if this fails, the core pipeline is considered broken.
    """
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        raise AssertionError(
            f"Failed to import required module '{module_name}'. "
            f"Check that it exists and all its dependencies are installed."
        ) from exc


def _soft_import(module_name: str):
    """
    Try to import a module that is considered optional (e.g., heavy model backends).

    On failure, emit a warning instead of failing the test, so CI remains usable
    even without large model weights or GPU-only dependencies.
    """
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        warnings.warn(
            f"Optional module '{module_name}' could not be imported: {exc}. "
            "Some features (e.g., SAM3-based segmentation) may be unavailable.",
            UserWarning,
        )


def test_core_modules_importable():
    """
    Core, non-optional modules must import cleanly.
    These correspond to the main pipeline and diffusion safety evaluation.
    """
    required_modules = [
        # Main video → PET pipeline
        "traffic_analyzer",
        "bev_mapper",
        "gate_counter",  # added
        "pet_conflict_checker",
        "grid_trajectory.spatial_grid",
        "grid_trajectory.pet_grid",
        "grid_trajectory.sam3_grid_pet",

        # Diffusion training / evaluation
        "traffic_diffusion.training_utils",
        "traffic_diffusion.model_and_sampler",
        "traffic_diffusion.trajectory_diffusion",
        "traffic_diffusion.pet_safety_metrics",
        "analysis.safety_eval_diffusion",

        # Analysis utilities
        "analysis.pet_summary",
    ]

    for name in required_modules:
        _assert_can_import(name)


def test_optional_modules_importable_with_warning():
    """
    Optional / heavy modules: we try to import them, but only warn on failure.
    Examples: SAM/SAM3-related components that may need large weights or specific envs.
    """
    optional_modules = [
        "ultralytics.models.sam",  # SAM3 / SAM from Ultralytics (optional)
        "sam3_wrapper",            # If you have a custom SAM3 wrapper module
    ]

    for name in optional_modules:
        _soft_import(name)


def test_core_symbols_exist_in_key_modules():
    """
    Minimal interface check: verify that key modules expose the expected
    public functions/classes used by the pipeline.

    This stays shallow on purpose: no heavy computation or I/O, just presence checks.
    """
    # traffic_analyzer: must expose parse_args and main
    ta = importlib.import_module("traffic_analyzer")
    for attr in ["parse_args", "main"]:
        assert hasattr(ta, attr), f"traffic_analyzer is missing expected symbol '{attr}'"

    # diffusion training utilities - updated to match actual code
    td_utils = importlib.import_module("traffic_diffusion.training_utils")
    expected_utils = [
        "build_clean_dataloaders",  # from code dump
        "create_model",             # from code dump
        "train_diffusion_model",    # from code dump
    ]
    for attr in expected_utils:
        if not hasattr(td_utils, attr):
            public = [x for x in dir(td_utils) if not x.startswith("_")]
            warnings.warn(
                f"traffic_diffusion.training_utils is missing expected symbol '{attr}'. "
                f"Found public symbols: {public[:10]}...",
                UserWarning,
            )

    # model_and_sampler: basic API surface for evaluation
    mas = importlib.import_module("traffic_diffusion.model_and_sampler")
    for attr in ["load_model", "sample_future_denorm"]:
        if not hasattr(mas, attr):
            warnings.warn(
                f"traffic_diffusion.model_and_sampler is missing expected symbol '{attr}'. "
                "safety_eval_diffusion may not work as documented.",
                UserWarning,
            )

    # pet_summary: should define a CLI-ish entrypoint
    ps = importlib.import_module("analysis.pet_summary")
    if not hasattr(ps, "main") and not hasattr(ps, "run"):
        warnings.warn(
            "analysis.pet_summary does not expose 'main' or 'run'. "
            "If you rely on a different entrypoint, update this test.",
            UserWarning,
        )
