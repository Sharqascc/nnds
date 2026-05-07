"""NNDS core pipeline modules.

Provides unified access to main traffic analysis components:
- TrafficAnalyzer: Video → PET events pipeline
- BEVMapper: Pixel ↔ World coordinate transformation  
- PETConflictChecker: Conflict detection and classification
- TrafficVolumeCounter: Gate-based vehicle counting

Example:
    from core import TrafficAnalyzer, BEVMapper, PETConflictChecker
    
    analyzer = TrafficAnalyzer()
    df = analyzer.run_video_to_pet("videos/traffic_video.mp4")
    
    checker = PETConflictChecker(pet_threshold=2.0)
    conflicts = checker.detect_from_csv("outputs/petevents_bev.csv")
"""

from traffic_analyzer import TrafficAnalyzer, CompleteTrafficAnalyzer, run_video_to_pet, run_demo
from bev_mapper import BEVMapper, compute_homography_dlt
from pet_conflict_checker import PETConflictChecker, ConflictSeverity, classify_pet_severity, compute_pet
from gate_counter import TrafficVolumeCounter, VirtualGate

__all__ = [
    "TrafficAnalyzer",
    "CompleteTrafficAnalyzer",
    "run_video_to_pet",
    "run_demo",
    "BEVMapper",
    "compute_homography_dlt",
    "PETConflictChecker",
    "ConflictSeverity",
    "classify_pet_severity",
    "compute_pet",
    "TrafficVolumeCounter",
    "VirtualGate",
]

__version__ = "2.0.0"
__author__ = "NNDS Team"
