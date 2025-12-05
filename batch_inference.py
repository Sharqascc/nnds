
"""
GPU-optimized batched YOLO inference for traffic analysis with simple enhancement.
"""
import torch
import numpy as np
from typing import List, Tuple, Dict
from ultralytics import YOLO
import cv2

def enhance_frame(frame: np.ndarray) -> np.ndarray:
    # convert to YCrCb and apply CLAHE on luminance
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    y_enh = clahe.apply(y)
    ycrcb_enh = cv2.merge((y_enh, cr, cb))
    enh = cv2.cvtColor(ycrcb_enh, cv2.COLOR_YCrCb2BGR)
    # mild sharpening
    kernel = np.array([[0,-1,0],
                       [-1,5,-1],
                       [0,-1,0]], dtype=np.float32)
    enh = cv2.filter2D(enh, -1, kernel)
    return enh

class BatchedYOLOInference:
    """Batched YOLO inference with GPU optimization"""
    
    def __init__(self, model_path: str, batch_size: int = 8, conf_threshold: float = 0.05, 
                 imgsz: int = 1280, device: str = 'cuda'):
        self.model = YOLO(model_path)
        self.batch_size = batch_size
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"BatchedYOLOInference initialized on {self.device}")
        print(f"Batch size: {batch_size}, Image size: {imgsz}")
    
    def process_frames(self, frames: List[Tuple[int, np.ndarray]]) -> List[Dict]:
        all_detections = []
        for i in range(0, len(frames), self.batch_size):
            batch = frames[i:i + self.batch_size]
            # apply enhancement per frame
            batch_imgs = [enhance_frame(f[1]) for f in batch]
            batch_indices = [f[0] for f in batch]
            
            results = self.model(
                batch_imgs,
                conf=self.conf_threshold,
                imgsz=self.imgsz,
                device=0 if self.device == 'cuda' else 'cpu',
                verbose=False
            )
            
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
        return {
            "model_name": getattr(self.model, "name", str(self.model)),
            "batch_size": self.batch_size,
            "conf_threshold": self.conf_threshold,
            "imgsz": self.imgsz,
            "device": self.device
        }
