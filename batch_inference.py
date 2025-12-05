
"""
GPU-optimized batched YOLO inference for traffic analysis
"""
import torch
import numpy as np
from typing import List, Tuple, Dict
from ultralytics import YOLO

class BatchedYOLOInference:
    """Batched YOLO inference with GPU optimization"""
    
    def __init__(self, model_path: str, batch_size: int = 8, conf_threshold: float = 0.05, 
                 imgsz: int = 1280, device: str = 'cuda'):
        """
        Initialize batched YOLO inference
        
        Args:
            model_path: Path to YOLO model weights
            batch_size: Number of frames per batch
            conf_threshold: Confidence threshold for detections
            imgsz: Image size for inference
            device: 'cuda' or 'cpu'
        """
        self.model = YOLO(model_path)
        self.batch_size = batch_size
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        print(f"BatchedYOLOInference initialized on {self.device}")
        print(f"Batch size: {batch_size}, Image size: {imgsz}")
    
    def process_frames(self, frames: List[Tuple[int, np.ndarray]]) -> List[Dict]:
        """
        Process frames in batches.
        
        Args:
            frames: List of (frame_idx, frame_array)
        
        Returns:
            List of detection dicts with keys: "frame_idx", "boxes", "confs", "classes"
        """
        all_detections = []
        
        for i in range(0, len(frames), self.batch_size):
            batch = frames[i:i + self.batch_size]
            batch_imgs = [f[1] for f in batch]
            batch_indices = [f[0] for f in batch]
            
            # Batched inference
            results = self.model(
                batch_imgs,
                conf=self.conf_threshold,
                imgsz=self.imgsz,
                device=0 if self.device == 'cuda' else 'cpu',
                verbose=False
            )
            
            # Extract detections
            for idx, dets in zip(batch_indices, results):
                if dets.boxes is not None:
                    boxes = dets.boxes.xyxy.cpu().numpy()
                    confs = dets.boxes.conf.cpu().numpy()
                    classes = dets.boxes.cls.cpu().numpy()
                else:
                    boxes = np.zeros((0, 4), dtype=np.float32)
                    confs = np.zeros((0,), dtype=np.float32)
                    classes = np.zeros((0,), dtype=np.float32)
                
                all_detections.append({
                    "frame_idx": idx,
                    "boxes": boxes,
                    "confs": confs,
                    "classes": classes
                })
        
        return all_detections
    
    def get_model_info(self) -> Dict:
        """Return basic model configuration and device info."""
        return {
            "model_name": getattr(self.model, "name", str(self.model)),
            "batch_size": self.batch_size,
            "conf_threshold": self.conf_threshold,
            "imgsz": self.imgsz,
            "device": self.device
        }
