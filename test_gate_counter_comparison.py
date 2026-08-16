"""
Compare Gate Counter vs VLM Counts
===================================
"""

import cv2
import numpy as np
from pathlib import Path
from gate_counter import TrafficVolumeCounter, VirtualGate

# Configuration
VIDEO_PATH = "/content/nnds/sample_data/traffic_video.mp4"
GATE_CONFIG = "/content/nnds/configs/gate_config.yaml"
OUTPUT_DIR = Path("/content/nnds/outputs/gate_comparison")

def simple_detector(frame):
    """Simple detector using background subtraction"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Simple threshold to detect moving objects
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > 500:  # Filter small noise
            x, y, w, h = cv2.boundingRect(cnt)
            centroid = (x + w/2, y + h/2)
            detections.append({
                "track_id": i,
                "centroid": centroid,
                "cls": "vehicle",
                "conf": 0.8,
                "bbox": (x, y, w, h)
            })
    
    return detections

def main():
    print("="*60)
    print("Gate Counter vs VLM Comparison")
    print("="*60)
    
    # Run gate counter
    print("\n1. Running automated gate counter...")
    counter = TrafficVolumeCounter(
        videopath=VIDEO_PATH,
        gate_config=GATE_CONFIG,
        classes_of_interest=["vehicle", "car", "motorcycle", "bus", "truck"]
    )
    
    # Process first 100 frames for quick test
    result = counter.process_video(
        detector=simple_detector,
        max_frames=100,
        show_progress=False
    )
    
    print(f"\nAutomated Gate Counter Results:")
    print(f"  Total entries: {result['total_entries']}")
    print(f"  Total exits: {result['total_exits']}")
    
    for gate_name, stats in result['gates'].items():
        print(f"  {gate_name}: IN={stats['entries']}, OUT={stats['exits']}")
    
    # Compare with VLM
    print("\n2. VLM Counts (from previous test):")
    import json
    vlm_results = Path("/content/nnds/outputs/vlm_gate_validation.json")
    if vlm_results.exists():
        with open(vlm_results) as f:
            vlm_data = json.load(f)
        
        for item in vlm_data:
            print(f"  Frame {item['frame']}: {item['vlm_count']} vehicles (conf={item['confidence']:.2f})")
        
        avg_vlm = sum(item['vlm_count'] for item in vlm_data) / len(vlm_data)
        print(f"\n  Average VLM count: {avg_vlm:.1f} vehicles/frame")
    
    # Save comparison
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    comparison = {
        "automated": result,
        "vlm": vlm_data if vlm_results.exists() else []
    }
    
    with open(OUTPUT_DIR / "comparison.json", "w") as f:
        import json
        json.dump(comparison, f, indent=2)
    
    print(f"\n✅ Comparison saved to {OUTPUT_DIR / 'comparison.json'}")

if __name__ == "__main__":
    main()
