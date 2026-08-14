"""
NNDS VLM Module - Vision-Language Model for Traffic Analysis
============================================================

This module provides VLM-based analysis for:
- PET event severity classification
- Natural language descriptions of conflicts
- Gate counting validation
- Traffic scene understanding

Usage:
    from vlm import VLMAnalyzer
    
    analyzer = VLMAnalyzer(api_key="your-key")
    results = analyzer.analyze_pet_events(pet_events_df)
"""

from .analyzer import VLMAnalyzer
from .gate_validator import GateValidator
from .config import VLMConfig

__version__ = "1.0.0"
__all__ = ["VLMAnalyzer", "GateValidator", "VLMConfig"]
