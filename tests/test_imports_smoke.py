from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module_exists(module_name: str) -> bool:
    """Check if a module can be found without importing it (avoids heavy deps)."""
    parts = module_name.split(".")
    path = ROOT / "src" / Path(*parts)
    if path.with_suffix(".py").exists():
        return True
    # Check package
    return bool(path.joinpath("__init__.py").exists())


def test_core_modules_present():
    required_modules = [
        "pipeline.traffic_analyzer",
        "bev.bev_mapper",
        "analysis.gate_counter",
        "analysis.pet_conflict_checker",
        "analysis.grid_trajectory.spatial_grid",
        "analysis.grid_trajectory.pet_grid",
        "analysis.grid_trajectory.sam3_grid_pet",
        "diffusion.traffic_diffusion.training_utils",
        "diffusion.traffic_diffusion.model_and_sampler",
        "diffusion.traffic_diffusion.trajectory_diffusion",
        "analysis.safety_eval_diffusion",
        "analysis.pet_summary",
    ]
    missing = [m for m in required_modules if not _module_exists(m)]
    assert not missing, f"Missing core modules: {missing}"
