"""Out-of-scope modules — structural check only.

These modules belong to separate research lines (diffusion models,
vision-language models) or developer tooling. They are excluded from
the paper's scope (see paper/SCOPE.md). This test ensures they remain
importable so nothing breaks silently, without spending property-test
budget on code the paper does not use.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


# Modules marked out of scope — import is best-effort; optional deps may
# be missing in some environments. A successful import is required only
# if the module's dependencies are available.
OUT_OF_SCOPE = [
    "src/diffusion/complete_ddpm.py",
    "src/diffusion/traffic_diffusion/evaluate_fixed.py",
    "src/diffusion/traffic_diffusion/model_and_sampler.py",
    "src/diffusion/traffic_diffusion/sampling_utils.py",
    "src/diffusion/traffic_diffusion/split_dataset.py",
    "src/diffusion/traffic_diffusion/train_trajectory_diffusion.py",
    "src/diffusion/traffic_diffusion/training_utils.py",
    "src/diffusion/traffic_diffusion/trajectory_diffusion.py",
    "src/diffusion/traffic_diffusion/transformer_diffusion.py",
    "src/diffusion/traj_diffusion_normalized.py",
    "src/utils/debug_helpers.py",
    "src/utils/duration_parser.py",
    "src/utils/interactive.py",
    "src/vlm/config.py",
    "src/vlm/gate_validator.py",
    "src/vlm/utils/image_utils.py",
    "src/vlm/vlm_enhanced_pipeline.py",
]


def _to_module(path: str) -> str:
    return path.replace("/", ".").replace(".py", "")


@pytest.mark.parametrize("path", OUT_OF_SCOPE)
def test_out_of_scope_module_path_exists(path):
    """Every entry must correspond to a real file on disk."""
    assert Path(path).exists(), f"{path} referenced but missing"


@pytest.mark.parametrize("path", OUT_OF_SCOPE)
def test_out_of_scope_module_importable_or_skip(path):
    """Module imports if its optional dependencies are available."""
    modname = _to_module(path)
    try:
        importlib.import_module(modname)
    except ImportError as e:
        pytest.skip(f"optional dependency missing: {e}")
