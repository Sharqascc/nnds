
import importlib
import sys
import builtins
import pytest

def _reload_grid():
    import src.analysis.grid_trajectory as gt
    importlib.reload(gt)
    return gt

def test_grid_trajectory_init_success():
    gt = _reload_grid()
    # After normal import, __all__ should contain names
    assert isinstance(gt.__all__, list)
    assert 'SpatialGrid' in gt.__all__ or 'TrajectoryLogger' in gt.__all__

@pytest.mark.parametrize("bad_module", [
    "src.analysis.grid_trajectory.pet_grid",
    "src.analysis.grid_trajectory.spatial_grid",
    "src.analysis.grid_trajectory.trajectory_safety_analyzer",
])
def test_grid_trajectory_import_error(bad_module, monkeypatch):
    import src.analysis.grid_trajectory as gt
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == bad_module:
            raise ImportError(f"mock import error for {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    # Reload should silently pass for the failed import
    importlib.reload(gt)
    # __all__ still exists
    assert isinstance(gt.__all__, list)
    # Restore normal import
    monkeypatch.setattr(builtins, '__import__', original_import)
    importlib.reload(gt)
