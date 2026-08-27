"""
Lightweight VLM analyzer using BLIP VQA (Salesforce/blip-vqa-base).

This is a smaller model (~385M params) suitable for CPU or limited memory.
"""

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering


class VLLMAnalyzer:
    def __init__(self, model_name="Salesforce/blip-vqa-base", device="cpu"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Loading VLM model: {model_name} on {self.device}")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForQuestionAnswering.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print("✅ VLM loaded.")

    def analyze_image(self, image_path: str, prompt: str | None = None) -> str:
        """Basic image question answering."""
        image = Image.open(image_path).convert("RGB")
        if prompt is None:
            prompt = "Describe this traffic scene in detail."
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=50)
        return self.processor.decode(generated_ids[0], skip_special_tokens=True)
