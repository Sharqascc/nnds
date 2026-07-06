# analysis/vlm_backend_stub.py
from __future__ import annotations
import json
from vlm.vlm_schema import VLMBackend


class EchoStubBackend:
    """Placeholder backend that returns a deterministic dummy JSON.

    Replace this with a real GPT-4V / LLaVA backend once wiring is tested.
    """

    model_name: str = "echo-stub"

    def analyze(self, image_path: str, prompt: str) -> str:
        # Very simple deterministic response for wiring & testing
        dummy = {
            "description": f"Stub analysis for {image_path}",
            "scenario_type": "other",
            "violation_type": "none",
            "severity": "medium",
            "contributing_factors": ["stub_backend"],
            "confidence": 0.1,
            "pet_comparison": "similar",
        }
        return json.dumps(dummy)
