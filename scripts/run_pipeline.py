#!/usr/bin/env python3
"""
Main entry point for the NNDS Intersection Safety Pipeline.
Usage:
    python scripts/run_pipeline.py --video path/to/video.mp4 --detector uvh-coco-fused
"""
import sys
import runpy
from pathlib import Path

if __name__ == "__main__":
    # Add project root to Python path so imports like 'src.analysis' work
    root_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(root_dir))
    
    # Run the actual pipeline script
    pipeline_path = root_dir / "src" / "pipeline" / "traffic_analyzer.py"
    if not pipeline_path.exists():
        print(f"ERROR: Pipeline not found at {pipeline_path}")
        sys.exit(1)
    
    # Pass command-line arguments through
    sys.argv[0] = str(pipeline_path)  # Makes help messages look correct
    runpy.run_path(str(pipeline_path), run_name="__main__")
