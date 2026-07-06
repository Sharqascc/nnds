# analysis/vlm_schema.py
from __future__ import annotations

import json
from typing import Literal, Protocol
from pydantic import BaseModel, Field, ValidationError


class VLMSafetyAnalysis(BaseModel):
    """Structured output from VLM for traffic conflicts."""
    description: str = Field(description="Narrative of what happened")

    scenario_type: Literal[
        "rear_end",
        "crossing",
        "lane_change",
        "pedestrian_crossing",
        "cyclist_interaction",
        "other",
    ] = Field(description="Type of traffic conflict")

    violation_type: Literal[
        "none",
        "speeding",
        "red_light",
        "yield_failure",
        "lane_discipline",
        "pedestrian_right_of_way",
        "other",
    ] = Field(description="Rule violation if any")

    severity: Literal["low", "medium", "high", "critical"] = Field(
        description="Perceived safety severity",
    )

    contributing_factors: list[str] = Field(
        description="Factors (e.g., 'low visibility', 'high speed')",
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model self-reported confidence",
    )

    pet_comparison: Literal["lower", "similar", "higher"] = Field(
        description="How semantic severity compares to PET value",
    )

    @classmethod
    def from_model_response(cls, text: str) -> "VLMSafetyAnalysis":
        """Parse raw model text into a validated schema with robust fallback."""
        try:
            data = json.loads(text)
            return cls(**data)
        except (json.JSONDecodeError, ValidationError):
            # Fallback when model returns malformed or non-JSON output
            return cls(
                description=text[:200],
                scenario_type="other",
                violation_type="none",
                severity="medium",
                contributing_factors=["unable_to_parse"],
                confidence=0.0,
                pet_comparison="similar",
            )


class VLMBackend(Protocol):
    """Simple interface for a vision-language backend."""

    model_name: str

    def analyze(self, image_path: str, prompt: str) -> str:
        """
        Given an image (or collage) and a prompt, return a JSON string
        matching VLMSafetyAnalysis fields.
        """
        ...
