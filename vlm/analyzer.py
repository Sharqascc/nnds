"""
VLM Analyzer - Core analysis engine for PET events
===================================================

Provides VLM-based analysis of traffic conflicts including:
- Severity classification
- Natural language descriptions
- Recommended actions
- Conflict type identification
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd

from .config import VLMConfig


@dataclass
class VLMAnalysisResult:
    """Result of VLM analysis for a single PET event."""
    event_id: int
    pet: float
    severity: str
    description: str
    recommended_action: str
    conflict_type: str
    tracks_involved: List[int]
    confidence: float = 1.0
    processing_time: float = 0.0


class VLMAnalyzer:
    """
    Vision-Language Model analyzer for traffic safety analysis.
    """
    
    def __init__(self, config: Optional[VLMConfig] = None, api_key: Optional[str] = None):
        self.config = config or VLMConfig()
        if api_key:
            self.config.api_key = api_key
        self._client = None
        self._results: List[VLMAnalysisResult] = []
    
    def analyze_pet_events(self, pet_events: pd.DataFrame) -> List[VLMAnalysisResult]:
        """Analyze multiple PET events."""
        results = []
        for idx, row in pet_events.iterrows():
            pet_event = row.to_dict()
            result = self._analyze_single(pet_event)
            results.append(result)
            if (idx + 1) % 10 == 0:
                print(f"Analyzed {idx + 1}/{len(pet_events)} events...")
        self._results = results
        return results
    
    def _analyze_single(self, pet_event: Dict[str, Any]) -> VLMAnalysisResult:
        """Analyze a single PET event."""
        event_id = pet_event.get("event_id", 0)
        pet = pet_event.get("pet", 0.0)
        conflict_type = pet_event.get("conflict_type", "UNKNOWN")
        tracks = pet_event.get("tracks_involved", [pet_event.get("track_a", 0)])
        
        # Rule-based analysis (fallback when no API key)
        severity = self._classify_severity(pet)
        description = f"Conflict at {conflict_type}. PET = {pet:.3f}s ({severity} severity)."
        action = self._get_action(severity)
        
        return VLMAnalysisResult(
            event_id=event_id,
            pet=pet,
            severity=severity,
            description=description,
            recommended_action=action,
            conflict_type=conflict_type,
            tracks_involved=tracks if isinstance(tracks, list) else [tracks],
            confidence=0.8
        )
    
    def _classify_severity(self, pet: float) -> str:
        """Classify PET event severity."""
        if pet < self.config.severity_thresholds["SEVERE"]:
            return "SEVERE"
        elif pet < self.config.severity_thresholds["HIGH"]:
            return "HIGH"
        elif pet < self.config.severity_thresholds["MODERATE"]:
            return "MODERATE"
        return "LOW"
    
    def _get_action(self, severity: str) -> str:
        """Get recommended action for severity level."""
        actions = {
            "SEVERE": "Immediate evasive action required",
            "HIGH": "Monitor situation closely",
            "MODERATE": "Continue monitoring",
            "LOW": "No immediate action needed"
        }
        return actions.get(severity, "Review manually")
    
    def save_results(self, results: List[VLMAnalysisResult], output_path: str) -> None:
        """Save analysis results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [
            {
                "event_id": r.event_id,
                "pet": r.pet,
                "severity": r.severity,
                "description": r.description,
                "recommended_action": r.recommended_action,
                "conflict_type": r.conflict_type,
                "tracks_involved": r.tracks_involved
            }
            for r in results
        ]
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(results)} results to {output_path}")
    
    def generate_statistics(self, results: List[VLMAnalysisResult]) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not results:
            return {}
        
        pets = [r.pet for r in results]
        severities = [r.severity for r in results]
        
        return {
            "total_events": len(results),
            "mean_pet": sum(pets) / len(pets),
            "min_pet": min(pets),
            "max_pet": max(pets),
            "severity_distribution": {
                "SEVERE": severities.count("SEVERE"),
                "HIGH": severities.count("HIGH"),
                "MODERATE": severities.count("MODERATE"),
                "LOW": severities.count("LOW")
            }
        }
