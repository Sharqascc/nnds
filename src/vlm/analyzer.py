"""
VLM analyzer supporting small BLIP VQA and larger BLIP2-2.7B.

By default, uses small BLIP VQA for low memory. If model_name contains "blip2",
uses Blip2ForConditionalGeneration with proper loading (no .to after device_map).
"""
import torch
from PIL import Image
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
    BlipForQuestionAnswering,
    BlipProcessor,
)


class VLLMAnalyzer:
    def __init__(self, model_name="Salesforce/blip-vqa-base", device="cpu"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Loading VLM model: {model_name} on {self.device}")
        if "blip2" in model_name.lower():
            # BLIP2 large model
            self.processor = Blip2Processor.from_pretrained(model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )
            # Do not call .to(self.device) after device_map
        else:
            # small BLIP VQA
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForQuestionAnswering.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
        print("✅ VLM loaded.")

    def analyze_image(self, image_path: str, prompt: str | None = None) -> str:
        image = Image.open(image_path).convert("RGB")
        if prompt is None:
            prompt = "Describe this traffic scene in detail."
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        # Move only allowed keys
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.device)
        if "input_ids" in inputs:
            inputs["input_ids"] = inputs["input_ids"].to(self.device)
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"].to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=50)
        return self.processor.decode(generated_ids[0], skip_special_tokens=True)
