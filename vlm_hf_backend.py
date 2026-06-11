from __future__ import annotations

from typing import Literal

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

from vlm_schema import VLMBackend  # Protocol with analyze(image_path, prompt)


class HFVisionLanguageBackend:
    """
    Hugging Face-based vision-language backend implementing VLMBackend.
    Tested with LLaVA/Qwen-VL style models that accept (image, prompt).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen-VL-Chat",  # or a specific LLaVA HF ID
        device: Literal["cpu", "cuda"] = "cuda",
        max_new_tokens: int = 512,
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
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
