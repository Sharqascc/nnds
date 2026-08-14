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
    api_provider: str = "openai"  # openai, anthropic, google, local
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    
    # Model Configuration
    model_name: str = "gpt-4o-mini"  # or claude-3-haiku, gemini-1.5-flash
    max_tokens: int = 1024
    temperature: float = 0.0
    
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
