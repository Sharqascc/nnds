import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision import models


class ReIDEncoder:
    """Lightweight appearance encoder using MobileNetV3-Small pretrained on ImageNet."""

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        self.model = models.mobilenet_v3_small(weights=weights)
        self.model.classifier = torch.nn.Identity()  # remove classification head
        self.model.eval().to(self.device)
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((128, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def encode_crop(self, frame_bgr, x1, y1, x2, y2):
        """Return normalized 1D numpy embedding for an image crop."""
        try:
            x1i, y1i = max(0, int(x1)), max(0, int(y1))
            x2i, y2i = (
                min(frame_bgr.shape[1], int(x2)),
                min(frame_bgr.shape[0], int(y2)),
            )
            if x2i <= x1i or y2i <= y1i:
                return None
            crop = frame_bgr[y1i:y2i, x1i:x2i]
            if crop.size == 0:
                return None
            # Convert BGR to RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.model(tensor)
            emb = emb.cpu().numpy().reshape(-1)
            norm = np.linalg.norm(emb)
            if norm < 1e-6:
                return None
            return emb / norm
        except Exception:
            return None
