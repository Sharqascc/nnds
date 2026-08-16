"""
Test Qwen2-VL-2B for gate validation
"""

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import requests
from vlm.config import VLMConfig

def load_model():
    """Load Qwen2-VL-2B model"""
    print(f"Loading {VLMConfig.MODEL_NAME}...")
    
    processor = AutoProcessor.from_pretrained(
        VLMConfig.MODEL_NAME,
        trust_remote_code=True
    )
    
    model = AutoModelForImageTextToText.from_pretrained(
        VLMConfig.MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"✅ Model loaded on {model.device}")
    return processor, model

def count_gates(processor, model, image_path: str) -> dict:
    """Count gates in an image"""
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    print(f"Image size: {image.size}")
    
    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": VLMConfig.GATE_COUNT_PROMPT},
            ]
        }
    ]
    
    # Process
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt"
    ).to(model.device)
    
    # Generate
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=VLMConfig.MAX_NEW_TOKENS,
            temperature=VLMConfig.TEMPERATURE,
            do_sample=VLMConfig.DO_SAMPLE,
        )
    
    # Decode
    result = processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    return {
        "image": image_path,
        "response": result,
    }

if __name__ == "__main__":
    # Load model
    processor, model = load_model()
    
    # Test with a sample image
    test_image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
    
    print(f"\nTesting with: {test_image}")
    result = count_gates(processor, model, test_image)
    
    print("\n" + "="*50)
    print("RESULT:")
    print("="*50)
    print(result["response"])
    print("="*50)
