"""
Image utilities for VLM analysis
=================================

Functions for extracting and preparing frames for VLM analysis.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import os


def extract_frames(
    video_path: str,
    output_dir: str,
    frame_interval: int = 30,
    max_frames: Optional[int] = None
) -> List[str]:
    """
    Extract frames from video for VLM analysis.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        frame_interval: Extract every N frames
        max_frames: Maximum number of frames to extract
        
    Returns:
        List of saved frame paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frame_paths = []
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_path = output_dir / f"frame_{saved_count:05d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(str(frame_path))
            saved_count += 1
            
            if max_frames and saved_count >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames to {output_dir}")
    return frame_paths


def prepare_images_for_vlm(
    image_paths: List[str],
    max_size: Tuple[int, int] = (1024, 1024),
    quality: int = 85
) -> List[str]:
    """
    Prepare images for VLM API (resize, compress).
    
    Args:
        image_paths: List of image paths
        max_size: Maximum dimensions (width, height)
        quality: JPEG quality (1-100)
        
    Returns:
        List of prepared image paths
    """
    prepared_paths = []
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Resize if needed
        h, w = img.shape[:2]
        max_w, max_h = max_size
        
        if w > max_w or h > max_h:
            scale = min(max_w / w, max_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Save compressed version
        output_path = Path(img_path).with_name(f"prepared_{Path(img_path).name}")
        cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        prepared_paths.append(str(output_path))
    
    return prepared_paths
