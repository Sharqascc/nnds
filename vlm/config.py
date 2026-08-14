"""
VLM Configuration
==================

Centralized configuration for VLM models and analysis parameters.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json
from pathlib import Path


@dataclass
class VLMConfig:
    """Configuration for VLM analysis."""
    
    # API Configuration
    api_provider: str = "openai"  # openai, anthropic, google, ollama, huggingface, groq
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    
    # Model Configuration - FREE & OPEN-SOURCE OPTIONS
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"  # Default: best free model
    
    # Recommended free models:
    # - Qwen/Qwen2.5-VL-7B-Instruct (best overall, Apache 2.0)
    # - meta-llama/Llama-3.2-11B-Vision-Instruct (great reasoning, free license)
    # - deepseek-ai/DeepSeek-VL (efficient, MIT license)
    # - google/gemma-3-12b-it (lightweight, open weights)
    
    max_tokens: int = 1024
    temperature: float = 0.0
    
    # Local deployment (Ollama)
    local_model: str = "qwen2.5-vl:7b"  # For Ollama: ollama run qwen2.5-vl:7b
    ollama_base: str = "http://localhost:11434"
    
    # Analysis Configuration
    severity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "SEVERE": 1.0,    # PET < 1.0s
        "HIGH": 1.5,      # PET < 1.5s
        "MODERATE": 2.5,  # PET < 2.5s
        "LOW": float("inf")  # PET >= 2.5s
    })
    
    # PET Event Analysis
    analyze_conflicts: bool = True
    generate_descriptions: bool = True
    recommend_actions: bool = True
    
    # Gate Validation
    validate_gates: bool = True
    gate_config_path: Optional[str] = None
    
    # Output Configuration
    output_dir: str = "outputs/vlm_analysis"
    save_json: bool = True
    save_visualization: bool = True
    
    @classmethod
    def for_free_model(cls, model: str = "qwen") -> "VLMConfig":
        """
        Pre-configured settings for free models.
        
        Args:
            model: 'qwen', 'llama', 'deepseek', 'gemma', or 'ollama'
        """
        configs = {
            "qwen": cls(
                api_provider="huggingface",
                model_name="Qwen/Qwen2.5-VL-7B-Instruct",
                api_key="hf_xxx"  # Get free token from huggingface.co
            ),
            "llama": cls(
                api_provider="huggingface",
                model_name="meta-llama/Llama-3.2-11B-Vision-Instruct",
                api_key="hf_xxx"
            ),
            "deepseek": cls(
                api_provider="huggingface",
                model_name="deepseek-ai/DeepSeek-VL",
                api_key="hf_xxx"
            ),
            "ollama": cls(
                api_provider="ollama",
                api_base="http://localhost:11434",
                local_model="qwen2.5-vl:7b"
            ),
            "groq": cls(
                api_provider="groq",
                model_name="llama-3.2-11b-vision-preview",
                api_key="gsk_xxx"  # Free tier at groq.com
            )
        }
        return configs.get(model, configs["qwen"])
    
    @classmethod
    def from_file(cls, path: str) -> "VLMConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)
    
    @property
    def default_prompt(self) -> str:
        """Default prompt for PET event analysis."""
        return """You are a traffic safety analyst. Analyze this PET (Post-Encroachment Time) event and provide:
1. Severity classification (SEVERE/HIGH/MODERATE/LOW)
2. Natural language description of the conflict
3. Recommended action for traffic management

PET Value: {pet:.3f}s
Conflict Type: {conflict_type}
Tracks Involved: {tracks}

Provide your analysis in JSON format with keys: severity, description, recommended_action."""


# Default configuration instance
DEFAULT_CONFIG = VLMConfig()
