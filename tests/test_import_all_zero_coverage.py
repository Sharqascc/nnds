import importlib
import pytest

# Modules that initially had 0% coverage.
# The notebook module will be skipped: it is a converted Jupyter notebook
# and may contain code that executes on import and fails outside its environment.
zero_modules = [
    "src.analysis.grid_trajectory.sam3_grid_pet",
    "src.analysis.grid_trajectory.yolo_cpu_grid_pet",
    "src.analysis.pet_diffusion_analysis",
    "src.analysis.research_run",
    "src.analysis.safety_eval_diffusion",
    # "src.analysis.safety_eval_diffusion_notebook",  # intentionally skipped
    "src.diffusion.complete_ddpm",
    "src.diffusion.traffic_diffusion.evaluate_fixed",
    "src.diffusion.traffic_diffusion.model_and_sampler",
    "src.diffusion.traffic_diffusion.sampling_utils",
    "src.diffusion.traffic_diffusion.split_dataset",
    "src.diffusion.traffic_diffusion.train_trajectory_diffusion",
    "src.diffusion.traffic_diffusion.training_utils",
    "src.diffusion.traffic_diffusion.trajectory_diffusion",
    "src.diffusion.traffic_diffusion.transformer_diffusion",
    "src.diffusion.traj_diffusion_normalized",
    "src.pipeline.rt_detr_detector",
    "src.utils.debug_helpers",
    "src.utils.interactive",
    "src.vlm.analyzer",
    "src.vlm.config",
    "src.vlm.gate_validator",
    "src.vlm.test_free_models",
    "src.vlm.utils.image_utils",
    "src.vlm.utils.visualization",
    "src.vlm.vlm_enhanced_pipeline",
]

@pytest.mark.parametrize("module_name", zero_modules)
def test_import_module(module_name):
    """Test that the module can be imported without errors."""
    try:
        importlib.import_module(module_name)
    except Exception as e:
        pytest.fail(f"Failed to import {module_name}: {e}")

# Explicitly skip the notebook module with a reason
@pytest.mark.skip(reason="Converted notebook, not a regular module and may fail on import")
def test_import_notebook_module():
    importlib.import_module("src.analysis.safety_eval_diffusion_notebook")
