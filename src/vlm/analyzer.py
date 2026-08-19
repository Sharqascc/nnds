import os
import json
import torch
from PIL import Image
import pandas as pd
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from typing import Union, Dict, Any, Optional
import re

class VLLMAnalyzer:
    """Enhanced VLM analyzer with trajectory-aware reasoning."""
    
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Loading VLM model: {model_name} on {self.device}")
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device=="cuda" else torch.float32,
            device_map="auto" if device=="cuda" else None
        ).to(self.device)
        print("✅ VLM loaded.")

    def analyze_image(self, image_path: str, prompt: str = None) -> str:
        """Basic image-to-text."""
        image = Image.open(image_path).convert('RGB')
        if prompt is None:
            prompt = "Describe this traffic scene in detail."
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items() if k in ["pixel_values", "input_ids"]}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                num_beams=3,
                temperature=0.7,
                do_sample=True
            )
        return self.processor.decode(outputs[0], skip_special_tokens=True)

    def analyze_with_trajectory(
        self,
        image_path: str,
        trajectory_df: pd.DataFrame,
        prompt_template: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform VLM reasoning that incorporates trajectory data.
        Returns a structured JSON with analysis.
        """
        # Load image
        image = Image.open(image_path).convert('RGB')

        # Prepare trajectory summary
        if trajectory_df is not None and not trajectory_df.empty:
            summary = {
                "num_vehicles": len(trajectory_df['track_id'].unique()) if 'track_id' in trajectory_df else len(trajectory_df),
                "avg_speed": trajectory_df['speed'].mean() if 'speed' in trajectory_df else None,
                "max_speed": trajectory_df['speed'].max() if 'speed' in trajectory_df else None,
                "pet_events": len(trajectory_df[trajectory_df.get('pet', 0) < 2.0]) if 'pet' in trajectory_df else 0,
                "ttc_events": len(trajectory_df[trajectory_df.get('ttc', 10) < 3.0]) if 'ttc' in trajectory_df else 0,
            }
            summary_str = ", ".join(f"{k}: {v}" for k, v in summary.items() if v is not None)
        else:
            summary_str = "No trajectory data provided."

        # Build prompt
        if prompt_template is None:
            prompt = f"""You are a traffic safety analyst. Given the traffic scene in the image and the following trajectory summary: {summary_str}, provide a detailed analysis covering:
1. Overall risk level (low/medium/high)
2. Main safety concerns visible in the scene (e.g., conflicts, near-misses)
3. Potential causes of these concerns
4. Suggested countermeasures to improve safety

Output your answer as a JSON object with keys: risk_level, concerns (list of strings), causes (list of strings), suggestions (list of strings)."""
        else:
            prompt = prompt_template.format(summary=summary_str)

        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items() if k in ["pixel_values", "input_ids"]}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                num_beams=3,
                temperature=0.7,
                do_sample=True
            )
        raw_text = self.processor.decode(outputs[0], skip_special_tokens=True)

        # Attempt to parse JSON from the raw output
        try:
            json_match = re.search(r'\{.*?\}', raw_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {"raw_analysis": raw_text, "risk_level": "unknown"}
        except:
            result = {"raw_analysis": raw_text, "risk_level": "unknown"}

        result["image"] = image_path
        result["trajectory_summary"] = summary_str
        return result
