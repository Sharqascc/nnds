from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_diffusion_key_files_exist():
    base = ROOT / "src" / "diffusion" / "traffic_diffusion"
    expected = [
        base / "train_trajectory_diffusion.py",
        base / "training_utils.py",
        base / "model_and_sampler.py",
        base / "trajectory_diffusion.py",
        base / "pet_safety_metrics.py",
    ]
    for p in expected:
        assert p.exists(), f"Missing diffusion file: {p.relative_to(ROOT)}"


def test_diffusion_data_files_exist():
    data_dir = ROOT / "src" / "diffusion" / "traffic_diffusion" / "data"
    for fname in [
        "trajdiff_inputs.npy",
        "trajdiff_targets.npy",
        "trajdiff_meta.parquet",
    ]:
        p = data_dir / fname
        assert p.exists(), f"Missing diffusion data file: {p.relative_to(ROOT)}"
        assert p.stat().st_size > 0, f"Empty diffusion data file: {p.relative_to(ROOT)}"
