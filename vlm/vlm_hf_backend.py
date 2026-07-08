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
        # Derive the cache subfolder from the actual model_name passed in,
        # instead of hardcoding the 2B model's folder name -- previously this
        # was hardcoded, so passing a different model (e.g. the 7B variant
        # vlm_events.py actually uses) would load/cache into the wrong folder.
        cache_dir_name = "models--" + model_name.replace("/", "--")
        cache_dir = models_dir / cache_dir_name

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

        # generate() returns the input prompt tokens concatenated with the
        # newly generated tokens for this decoder-based model. Previously the
        # full sequence was decoded and returned as-is, meaning downstream
        # JSON parsing (VLMSafetyAnalysis.from_model_response) always failed
        # on real output since "<prompt><json>" is not valid JSON on its own.
        # Slice off the input length so only the new tokens are decoded.
        input_length = inputs["input_ids"].shape[1]
        new_tokens = output_ids[:, input_length:]
        out_text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return out_text
