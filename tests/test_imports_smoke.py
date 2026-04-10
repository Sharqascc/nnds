import importlib


def _can_import(name: str) -> None:
    importlib.import_module(name)


def test_core_modules_importable():
    _can_import("traffic_analyzer")
    _can_import("traffic_diffusion.training_utils")
    _can_import("traffic_diffusion.model_and_sampler")
    _can_import("analysis.safety_eval_diffusion")
