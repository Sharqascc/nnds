from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]

def test_expected_paths_exist():
    expected = [
        "README.md",
        "Makefile",
        "requirements.txt",
        "traffic_analyzer.py",
        "bev_mapper.py",
        "giti_bev_calib.py",
        "grid_trajectory/pet_grid.py",
        "grid_trajectory/sam3_grid_pet.py",
        "grid_trajectory/spatial_grid.py",
        "analysis/safety_eval_diffusion.py",
        "analysis/safety_eval_diffusion_notebook.py",
        "traffic_diffusion/train_trajectory_diffusion.py",
        "traffic_diffusion/training_utils.py",
        "traffic_diffusion/model_and_sampler.py",
        "configs/bev_config.json",
        "configs/giti_calibration_points.json",
        "configs/GITI_grid_config.json",
    ]
    missing = [p for p in expected if not (ROOT / p).exists()]
    assert not missing, f"Missing expected files: {missing}"

def test_outputs_structure_exists():
    assert (ROOT / "outputs").exists()

def test_can_load_top_level_scripts():
    for rel_path in [
        "traffic_analyzer.py",
        "bev_mapper.py",
        "giti_bev_calib.py",
        "traj_diffusion_normalized.py",
    ]:
        path = ROOT / rel_path
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec is not None, f"Could not create spec for {rel_path}"

def test_diffusion_data_files_present():
    expected = [
        "traffic_diffusion/data/trajdiff_inputs.npy",
        "traffic_diffusion/data/trajdiff_targets.npy",
        "traffic_diffusion/data/trajdiff_meta.parquet",
    ]
    missing = [p for p in expected if not (ROOT / p).exists()]
    assert not missing, f"Missing diffusion data files: {missing}"
