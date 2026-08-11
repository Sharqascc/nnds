from pathlib import Path


def test_repo_smoke():
    root = Path(".")
    assert (root / "core").exists()
    assert (root / "tests").exists()
