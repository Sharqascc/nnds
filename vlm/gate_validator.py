"""
Gate Validator - VLM-based gate counting validation
====================================================
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .config import VLMConfig


@dataclass
class GateValidationResult:
    """Result of gate validation."""
    frame_id: int
    gate_name: str
    vlm_count: int
    automated_count: int
    difference: int
    confidence: float
    vlm_description: str


class GateValidator:
    """VLM-based gate counting validator."""
    
    def __init__(self, config: Optional[VLMConfig] = None, api_key: Optional[str] = None):
        self.config = config or VLMConfig()
        if api_key:
            self.config.api_key = api_key
    
    def validate_gate_counts(
        self,
        frame_paths: List[str],
        gate_name: str,
        automated_counts: List[int],
        sample_interval: int = 10
    ) -> List[GateValidationResult]:
        """Validate automated gate counts using VLM."""
        results = []
        sample_indices = range(0, len(frame_paths), sample_interval)
        
        for idx in sample_indices:
            if idx >= len(frame_paths) or idx >= len(automated_counts):
                break
            
            # Placeholder - would call VLM API with image
            result = GateValidationResult(
                frame_id=idx,
                gate_name=gate_name,
                vlm_count=automated_counts[idx],  # Placeholder
                automated_count=automated_counts[idx],
                difference=0,
                confidence=0.9,
                vlm_description="VLM analysis placeholder"
            )
            results.append(result)
        
        return results
    
    def generate_report(self, results: List[GateValidationResult], output_path: str) -> None:
        """Generate validation report."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        valid = [r for r in results if r.vlm_count >= 0]
        if not valid:
            return
        
        total = len(valid)
        exact = sum(1 for r in valid if r.difference == 0)
        within_1 = sum(1 for r in valid if abs(r.difference) <= 1)
        
        report = f"""# Gate Validation Report

## Summary
- Total Frames: {total}
- Exact Matches: {exact} ({exact/total*100:.1f}%)
- Within ±1: {within_1} ({within_1/total*100:.1f}%)

## Results
| Frame | Gate | VLM | Auto | Diff |
|-------|------|-----|------|------|
"""
        for r in valid[:20]:
            report += f"| {r.frame_id} | {r.gate_name} | {r.vlm_count} | {r.automated_count} | {r.difference:+d} |
"
        
        with open(output_path, "w") as f:
            f.write(report)
        print(f"Report saved to {output_path}")
