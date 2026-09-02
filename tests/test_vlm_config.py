
import json
from pathlib import Path
from src.vlm.config import VLMConfig, DEFAULT_CONFIG


def test_default_config():
    assert DEFAULT_CONFIG.api_provider == "openai"
    assert DEFAULT_CONFIG.model_name == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert isinstance(DEFAULT_CONFIG.severity_thresholds, dict)


def test_for_free_model_qwen():
    cfg = VLMConfig.for_free_model("qwen")
    assert cfg.api_provider == "huggingface"
    assert cfg.model_name.startswith("Qwen")


def test_for_free_model_unknown_falls_back_to_qwen():
    cfg = VLMConfig.for_free_model("unknown")
    assert cfg.api_provider == "huggingface"
    assert cfg.model_name.startswith("Qwen")


def test_for_free_model_llama():
    cfg = VLMConfig.for_free_model("llama")
    assert cfg.api_provider == "huggingface"
    assert "Llama" in cfg.model_name


def test_for_free_model_deepseek():
    cfg = VLMConfig.for_free_model("deepseek")
    assert cfg.api_provider == "huggingface"
    assert "DeepSeek" in cfg.model_name


def test_for_free_model_ollama():
    cfg = VLMConfig.for_free_model("ollama")
    assert cfg.api_provider == "ollama"
    assert cfg.api_base == "http://localhost:11434"


def test_for_free_model_groq():
    cfg = VLMConfig.for_free_model("groq")
    assert cfg.api_provider == "groq"
    assert "llama" in cfg.model_name


def test_from_file(tmp_path):
    data = {"api_provider": "test_provider", "model_name": "TestModel"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    cfg = VLMConfig.from_file(str(path))
    assert cfg.api_provider == "test_provider"
    assert cfg.model_name == "TestModel"


def test_save(tmp_path):
    cfg = VLMConfig(api_provider="test")
    path = tmp_path / "save.json"
    cfg.save(str(path))
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["api_provider"] == "test"


def test_default_prompt_property():
    prompt = VLMConfig().default_prompt
    assert "traffic safety analyst" in prompt
    assert "{pet:.3f}" in prompt
    assert "{conflict_type}" in prompt
    assert "{tracks}" in prompt
