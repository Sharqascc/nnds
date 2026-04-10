from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]


def test_diffusion_key_files_exist():
    assert (ROOT / "traffic_diffusion" / "train_trajectory_diffusion.py").exists()
    assert (ROOT / "traffic_diffusion" / "training_utils.py").exists()
    assert (ROOT / "analysis" / "safety_eval_diffusion.py").exists()
    assert (ROOT / "analysis" / "safety_eval_diffusion_notebook.py").exists()


def test_bev_and_main_scripts_exist():
    assert (ROOT / "traffic_analyzer.py").exists()
    assert (ROOT / "bev_mapper.py").exists()


def test_training_utils_importable():
    path = ROOT / "traffic_diffusion" / "training_utils.py"
    spec = importlib.util.spec_from_file_location("training_utils", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
