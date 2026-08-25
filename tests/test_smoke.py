from pathlib import Path


def test_repo_smoke():
    root = Path(".")
    assert (root / "src").exists()
    assert (root / "tests").exists()
