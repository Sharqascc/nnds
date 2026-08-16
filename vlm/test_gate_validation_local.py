"""
Local Qwen2-VL Gate Validation Test
====================================
Uses your saved Qwen2-VL-2B model from Drive to validate gate counting.
"""

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import json

# Configuration
MODEL_PATH = "/content/drive/MyDrive/nnds_models/qwen2-vl-2b"
VIDEO_PATH = "/content/nnds/sample_data/traffic_video.mp4"
GATE_CONFIG = {
    "name": "MainGate",
    "start": (100, 300),
    "end": (600, 300),
    "color": (0, 255, 255)  # Cyan
}

def load_model(model_path: str):
    """Load Qwen2-VL-2B from local path"""
    print(f"Loading model from {model_path}...")
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
    
    print(f"✅ Model loaded on {model.device}")
    return processor, model

def extract_frames_at_gate(
    video_path: str,
    gate_line: Tuple[Tuple[int, int], Tuple[int, int]],
    num_frames: int = 5
) -> List[Tuple[int, np.ndarray]]:
    """Extract frames where vehicles cross the gate line"""
    print(f"\nExtracting frames from {video_path}...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Draw gate line for visualization
            start, end = gate_line
            cv2.line(frame, start, end, (0, 255, 255), 2)
            frames.append((idx, frame))
    
    cap.release()
    print(f"Extracted {len(frames)} frames")
    return frames

def count_vehicles_with_qwen(
    processor,
    model,
    image: np.ndarray,
    gate_name: str
) -> Dict:
    """Use Qwen2-VL to count vehicles at gate"""
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Prepare prompt
    prompt = f"""You are a traffic analysis expert. Look at this traffic camera frame.

Gate: {gate_name}

Task: Count how many vehicles are visible in this image.

Provide your answer in this exact JSON format:
{{
    "vehicle_count": <integer>,
    "confidence": <float 0.0-1.0>,
    "description": "<brief description of what you see>"
}}

Be precise. Count all visible vehicles."""

    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ]
        }
    ]
    
    # Process
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[pil_image],
        return_tensors="pt"
    ).to(model.device)
    
    # Generate
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
        )
    
    # Decode
    result = processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Parse JSON from response
    import re
    try:
        start_idx = result.find("{")
        end_idx = result.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx:
            json_str = result[start_idx:end_idx]
            return json.loads(json_str)
    except:
        pass
    
    # Fallback: extract number from text
    numbers = re.findall(r"\b\d+\b", result)
    return {
        "vehicle_count": int(numbers[0]) if numbers else 0,
        "confidence": 0.3,
        "description": result[:200]
    }

def main():
    print("="*60)
    print("Local Qwen2-VL Gate Validation Test")
    print("="*60)
    
    # Load model
    processor, model = load_model(MODEL_PATH)
    
    # Extract frames
    gate_line = (GATE_CONFIG["start"], GATE_CONFIG["end"])
    frames = extract_frames_at_gate(VIDEO_PATH, gate_line, num_frames=3)
    
    if not frames:
        print("❌ No frames extracted!")
        return
    
    # Validate each frame
    print("\n" + "="*60)
    print("Validating frames with Qwen2-VL")
    print("="*60)
    
    results = []
    for frame_idx, frame in frames:
        print(f"\nFrame {frame_idx}:")
        
        # Count with Qwen
        result = count_vehicles_with_qwen(
            processor, model, frame, GATE_CONFIG["name"]
        )
        
        print(f"  Vehicle count: {result.get('vehicle_count', 'N/A')}")
        print(f"  Confidence: {result.get('confidence', 'N/A'):.2f}")
        print(f"  Description: {result.get('description', 'N/A')[:100]}")
        
        results.append({
            "frame": int(frame_idx),
            "vlm_count": int(result.get("vehicle_count", 0)),
            "confidence": float(result.get("confidence", 0)),
            "description": str(result.get("description", ""))
        })
    
    # Save results
    output_path = Path("/content/nnds/outputs/vlm_gate_validation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_path}")
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)

if __name__ == "__main__":
    main()
