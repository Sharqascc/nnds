"""
Gate Validator - Groq VLM-based gate counting validation
=========================================================

Uses Groq's Llama 3.2 Vision to independently count vehicles crossing gates
and validate the automated tracker's counts.
"""

import json
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from groq import Groq
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
    image_path: Optional[str] = None


class GateValidator:
    """Groq VLM-based gate counting validator."""
    
    def __init__(self, groq_api_key: str, config: Optional[VLMConfig] = None):
        self.config = config or VLMConfig()
        self.client = Groq(api_key=groq_api_key)
        self.model = "llama-3.2-11b-vision-preview"
    
    def validate_frame(
        self,
        image_path: str,
        gate_name: str,
        automated_count: int,
        gate_line: Optional[tuple] = None
    ) -> GateValidationResult:
        """
        Validate gate count for a single frame using Groq VLM.
        
        Args:
            image_path: Path to frame image
            gate_name: Name of the gate
            automated_count: Count from automated tracker
            gate_line: Optional (start, end) coordinates of gate line
        
        Returns:
            GateValidationResult with VLM count and comparison
        """
        # Encode image to base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Build prompt
        gate_info = f"Gate: {gate_name}"
        if gate_line:
            start, end = gate_line
            gate_info += f" (line from {start} to {end})"
        
        prompt = f"""You are a traffic analysis expert. Look at this traffic camera frame.

{gate_info}

Task: Count how many vehicles are crossing or have just crossed the gate line in this image.

Provide your answer in this exact JSON format:
{{
    "vehicle_count": <integer>,
    "confidence": <float 0.0-1.0>,
    "description": "<brief description of what you see>"
}}

Be precise. Count only vehicles that are on or have crossed the gate line."""

        # Call Groq API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.0,
            max_tokens=500
        )
        
        # Parse response
        vlm_text = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        try:
            # Try to find JSON in response
            start_idx = vlm_text.find("{")
            end_idx = vlm_text.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                vlm_json = json.loads(vlm_text[start_idx:end_idx])
            else:
                vlm_json = json.loads(vlm_text)
            
            vlm_count = int(vlm_json.get("vehicle_count", 0))
            confidence = float(vlm_json.get("confidence", 0.5))
            description = vlm_json.get("description", "No description")
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Fallback: try to extract number from text
            import re
            numbers = re.findall(r"\b\d+\b", vlm_text)
            vlm_count = int(numbers[0]) if numbers else 0
            confidence = 0.3
            description = f"Parse error: {vlm_text[:200]}"
        
        return GateValidationResult(
            frame_id=Path(image_path).stem,
            gate_name=gate_name,
            vlm_count=vlm_count,
            automated_count=automated_count,
            difference=vlm_count - automated_count,
            confidence=confidence,
            vlm_description=description,
            image_path=image_path
        )
    
    def validate_gate_counts(
        self,
        frame_paths: List[str],
        gate_name: str,
        automated_counts: List[int],
        gate_line: Optional[tuple] = None,
        sample_interval: int = 10,
        output_path: Optional[str] = None
    ) -> List[GateValidationResult]:
        """
        Validate automated gate counts on multiple frames.
        
        Args:
            frame_paths: List of frame image paths
            gate_name: Name of the gate
            automated_counts: List of automated counts per frame
            gate_line: Optional gate line coordinates
            sample_interval: Validate every N frames
            output_path: Optional path to save report
        
        Returns:
            List of GateValidationResult
        """
        results = []
        total = len(range(0, len(frame_paths), sample_interval))
        
        print(f"Validating {total} frames with Groq VLM...")
        
        for i, idx in enumerate(range(0, len(frame_paths), sample_interval)):
            if idx >= len(frame_paths) or idx >= len(automated_counts):
                break
            
            frame_path = frame_paths[idx]
            auto_count = automated_counts[idx]
            
            if not Path(frame_path).exists():
                print(f"  Skipping {frame_path} (not found)")
                continue
            
            print(f"  [{i+1}/{total}] Frame {idx}: Auto={auto_count}...")
            
            try:
                result = self.validate_frame(
                    image_path=frame_path,
                    gate_name=gate_name,
                    automated_count=auto_count,
                    gate_line=gate_line
                )
                results.append(result)
                print(f"    -> VLM={result.vlm_count}, Diff={result.difference:+d}")
            except Exception as e:
                print(f"    X Error: {e}")
        
        # Generate report
        if output_path and results:
            self.generate_report(results, output_path)
        
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
        
        avg_vlm = sum(r.vlm_count for r in valid) / total
        avg_auto = sum(r.automated_count for r in valid) / total
        
        report = f"""# Gate Validation Report (Groq VLM)

## Summary
- **Total Frames Validated:** {total}
- **Exact Matches:** {exact} ({exact/total*100:.1f}%)
- **Within +-1:** {within_1} ({within_1/total*100:.1f}%)
- **Average VLM Count:** {avg_vlm:.2f}
- **Average Automated Count:** {avg_auto:.2f}
- **Model:** {self.model}

## Results
| Frame | Gate | VLM | Auto | Diff | Confidence |
|-------|------|-----|------|------|------------|
"""
        for r in valid[:50]:  # Show first 50
            diff_str = f"{r.difference:+d}"
            report += f"| {r.frame_id} | {r.gate_name} | {r.vlm_count} | {r.automated_count} | {diff_str} | {r.confidence:.2f} |\n"
        
        # Add discrepancies
        discrepancies = [r for r in valid if abs(r.difference) > 1]
        if discrepancies:
            report += f"""
## Discrepancies (|diff| > 1)
Found {len(discrepancies)} frames with significant differences:

"""
            for r in discrepancies[:10]:
                report += f"- Frame {r.frame_id}: VLM={r.vlm_count}, Auto={r.automated_count}, Diff={r.difference:+d}\n"
                report += f"  Description: {r.vlm_description[:150]}...\n"
        
        with open(output_path, "w") as f:
            f.write(report)
        print(f"\n Report saved to {output_path}")


# Convenience function
def validate_gates_with_groq(
    groq_api_key: str,
    frame_paths: List[str],
    automated_counts: List[int],
    gate_name: str = "MainGate",
    output_path: str = "outputs/vlm_analysis/gate_validation_report.md"
) -> List[GateValidationResult]:
    """
    Quick function to validate gate counts with Groq.
    
    Args:
        groq_api_key: Your Groq API key
        frame_paths: List of frame image paths
        automated_counts: Automated tracker counts
        gate_name: Gate name
        output_path: Report output path
    
    Returns:
        List of validation results
    """
    validator = GateValidator(groq_api_key=groq_api_key)
    return validator.validate_gate_counts(
        frame_paths=frame_paths,
        gate_name=gate_name,
        automated_counts=automated_counts,
        sample_interval=10,
        output_path=output_path
    )
