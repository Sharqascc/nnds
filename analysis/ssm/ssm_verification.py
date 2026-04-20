"""SSM Verification Module for N NDS Pipeline.

Provides verification utilities for surrogate safety metrics
including PET (Post-Encroachment Time) and TTC (Time-to-Collision).
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path


@dataclass
class VerificationResult:
    """Container for SSM verification results."""
    metric_name: str
    passed: bool
    value: float
    threshold: float
    confidence: float
    notes: str = ""


class SSMVerifier:
    """Verifier for surrogate safety metrics.
    
    Validates SSM values against thresholds and checks
    consistency across frames and trajectories.
    """
    
    DEFAULT_PET_THRESHOLD = 5.0  # seconds
    DEFAULT_TTC_THRESHOLD = 5.0  # seconds
    CONFIDENCE_LEVEL = 0.95
    
    def __init__(
        self,
        pet_threshold: float = DEFAULT_PET_THRESHOLD,
        ttc_threshold: float = DEFAULT_TTC_THRESHOLD,
        confidence_level: float = CONFIDENCE_LEVEL,
    ):
        self.pet_threshold = pet_threshold
        self.ttc_threshold = ttc_threshold
        self.confidence_level = confidence_level
    
    def verify_pet_threshold(
        self,
        pet_values: np.ndarray,
        threshold: Optional[float] = None,
    ) -> VerificationResult:
        """Verify PET values against safety threshold.
        
        Args:
            pet_values: Array of PET values in seconds
            threshold: Override default threshold
            
        Returns:
            VerificationResult with pass/fail status
        """
        threshold = threshold or self.pet_threshold
        pet_values = np.asarray(pet_values)
        
        if len(pet_values) == 0:
            return VerificationResult(
                metric_name="PET",
                passed=False,
                value=np.nan,
                threshold=threshold,
                confidence=0.0,
                notes="No PET values provided",
            )
        
        min_pet = float(np.nanmin(pet_values))
        pct_below = float(np.nanmean(pet_values < threshold) * 100)
        
        passed = min_pet >= threshold * 0.5
        confidence = 1.0 - pct_below / 100
        
        return VerificationResult(
            metric_name="PET",
            passed=passed,
            value=min_pet,
            threshold=threshold,
            confidence=confidence,
            notes=f"{pct_below:.1f}% of values below threshold",
        )
    
    def verify_ttc_threshold(
        self,
        ttc_values: np.ndarray,
        threshold: Optional[float] = None,
    ) -> VerificationResult:
        """Verify TTC values against safety threshold.
        
        Args:
            ttc_values: Array of TTC values in seconds
            threshold: Override default threshold
            
        Returns:
            VerificationResult with pass/fail status
        """
        threshold = threshold or self.ttc_threshold
        ttc_values = np.asarray(ttc_values)
        
        if len(ttc_values) == 0:
            return VerificationResult(
                metric_name="TTC",
                passed=False,
                value=np.nan,
                threshold=threshold,
                confidence=0.0,
                notes="No TTC values provided",
            )
        
        min_ttc = float(np.nanmin(ttc_values))
        pct_below = float(np.nanmean(ttc_values < threshold) * 100)
        
        passed = min_ttc >= threshold * 0.5
        confidence = 1.0 - pct_below / 100
        
        return VerificationResult(
            metric_name="TTC",
            passed=passed,
            value=min_ttc,
            threshold=threshold,
            confidence=confidence,
            notes=f"{pct_below:.1f}% of values below threshold",
        )
    
    def verify_consistency(
        self,
        pet_values: np.ndarray,
        ttc_values: np.ndarray,
    ) -> Dict[str, VerificationResult]:
        """Verify consistency between PET and TTC metrics.
        
        Args:
            pet_values: PET values
            ttc_values: TTC values
            
        Returns:
            Dict of verification results
        """
        results = {}
        results["pet"] = self.verify_pet_threshold(pet_values)
        results["ttc"] = self.verify_ttc_threshold(ttc_values)
        
        if len(pet_values) > 0 and len(ttc_values) > 0:
            corr = float(np.corrcoef(pet_values, ttc_values)[0, 1])
            results["correlation"] = VerificationResult(
                metric_name="PET-TTC Correlation",
                passed=np.abs(corr) > 0.3,
                value=corr,
                threshold=0.3,
                confidence=0.95,
                notes=f"Pearson correlation coefficient",
            )
        
        return results
