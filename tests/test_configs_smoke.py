import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def _iter_json_configs():
    if not CONFIG_DIR.exists():
        return []
    return sorted(CONFIG_DIR.glob("*.json"))


def test_config_dir_exists():
    assert CONFIG_DIR.exists(), f"Config dir not found: {CONFIG_DIR}"


def test_all_json_configs_parse():
    json_files = _iter_json_configs()
    assert json_files, "No .json files found in configs/"

    for cfg in json_files:
        text = cfg.read_text()
        try:
            data = json.loads(text)
        except Exception as exc:
            raise AssertionError(f"Failed to parse JSON config: {cfg}") from exc
        assert isinstance(data, dict) or isinstance(data, list)
