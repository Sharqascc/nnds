from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from ultralytics import RTDETR


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: int


class RTDetrDetector:
    def __init__(self, weights_path: str = "rtdetr-l.pt"):
        self.model = RTDETR(weights_path)

    def detect(self, frame_bgr: np.ndarray, conf: float = 0.25) -> List[Detection]:
        results = self.model(frame_bgr, conf=conf)[0]
        dets: List[Detection] = []
        if results.boxes is None:
            return dets
        boxes_xyxy = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), s, c in zip(boxes_xyxy, scores, classes):
            dets.append(Detection(float(x1), float(y1), float(x2), float(y2), float(s), int(c)))
        return dets
