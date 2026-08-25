from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_expected_paths_exist():
    expected = [
        "README.md",
        "Makefile",
        "requirements.txt",
        "pyproject.toml",
        "scripts/run_pipeline.py",
        "src/pipeline/traffic_analyzer.py",
        "src/bev/bev_mapper.py",
        "src/bev/giti_bev_calib.py",
        "src/analysis/gate_counter.py",
        "src/analysis/pet_conflict_checker.py",
        "src/analysis/grid_trajectory/pet_grid.py",
        "src/analysis/grid_trajectory/sam3_grid_pet.py",
        "src/analysis/grid_trajectory/spatial_grid.py",
                "src/analysis/safety_eval_diffusion.py",
        "src/analysis/safety_eval_diffusion_notebook.py",
        "src/diffusion/traffic_diffusion/train_trajectory_diffusion.py",
        "src/diffusion/traffic_diffusion/training_utils.py",
        "src/diffusion/traffic_diffusion/model_and_sampler.py",
        "src/diffusion/traffic_diffusion/trajectory_diffusion.py",
                "configs/bev_config.json",
        "configs/giti_calibration_points.json",
        "configs/GITI_grid_config.json",
        "configs/gate_config.yaml",
        "docs/data_samples/petevents_bev_demo.csv",
    ]
    missing = [p for p in expected if not (ROOT / p).exists()]
    assert not missing, f"Missing expected files: {missing}"


def test_diffusion_data_files_present_and_non_empty():
    data_dir = ROOT / "src/diffusion/traffic_diffusion/data"
    for fname in [
        "trajdiff_inputs.npy",
        "trajdiff_targets.npy",
        "trajdiff_meta.parquet",
    ]:
        p = data_dir / fname
        assert p.exists(), f"Missing diffusion data file: {p}"
        assert p.stat().st_size > 0, f"Empty diffusion data file: {p}"
