from __future__ import annotations

from typing import Literal
from pathlib import Path

import torch

from vlm.vlm_schema import VLMBackend  # Protocol with analyze(image_path, prompt)

from transformers import AutoProcessor, AutoModelForImageTextToText


class HFVisionLanguageBackend(VLMBackend):
    """
    Hugging Face-based vision-language backend implementing VLMBackend.
    Uses Qwen2-VL-2B-Instruct by default, with a local cache directory.
    """

    model_name: str

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: Literal["cpu", "cuda"] = "cuda",
        max_new_tokens: int = 512,
        trust_remote_code: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.trust_remote_code = trust_remote_code

        models_dir = Path("/content/nnds/models")
        cache_dir = models_dir / "models--Qwen--Qwen2-VL-2B-Instruct"

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=trust_remote_code,
        ).to(device)

    def analyze(self, image_path: str, prompt: str) -> str:
        """
        Given an image path (collage) and prompt, return model text.
        The prompt should already contain JSON schema instructions.
        """
        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )

        out_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return out_text
