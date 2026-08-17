"""
VLM Configuration - Lightweight Computer Vision Models
"""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class VLMConfig:
    """Configuration for VLM-based gate validation"""
    
    # Model selection
    MODEL_NAME: Literal[
        "Qwen/Qwen2-VL-2B-Instruct",      # Recommended: 2B, fast, excellent CV
        "microsoft/Phi-3.5-vision-instruct",  # 4B, good alternative
        "HuggingFaceTB/SmolVLM-256M-Instruct", # 256M, ultra-lightweight
    ] = "Qwen/Qwen2-VL-2B-Instruct"
    
    # Device settings
    DEVICE: str = "cuda"
    DTYPE: str = "float16"  # or "bfloat16" for A100
    
    # Inference settings
    MAX_NEW_TOKENS: int = 256
    TEMPERATURE: float = 0.1  # Low for deterministic counting
    DO_SAMPLE: bool = False
    
    # Image preprocessing
    IMAGE_SIZE: tuple = (512, 512)  # Qwen2-VL handles various sizes
    MAX_PIXELS: int = 1280 * 28 * 28  # Qwen2-VL max
    
    # Batch processing
    BATCH_SIZE: int = 4
    NUM_WORKERS: int = 2
    
    # Gate validation specific
    GATE_COUNT_PROMPT: str = """Analyze this image carefully. Count the number of gates visible.

Rules:
1. Count only complete gate structures
2. A gate is a controlled entry/exit point with barriers or checkpoints
3. Look for gate houses, barriers, or checkpoint structures
4. Count each distinct gate separately

Provide your answer in this format:
Gate count: [number]

If you see no gates, respond:
Gate count: 0"""

    DISCREPANCY_PROMPT: str = """Compare the detected gate count ({detected_count}) with the expected count ({expected_count}).

Analyze the image and explain:
1. What might have been missed or double-counted
2. Any ambiguous structures that could be gates
3. Your final assessment of the correct count

Be specific about locations and visual evidence."""

    # Performance
    USE_FLASH_ATTN: bool = False  # Enable if available
    USE_CACHE: bool = True  # Use KV cache for faster inference


# Free API models (no GPU needed)
FREE_VLM_MODELS = {
    "groq-llama3.2-vision": {
        "provider": "groq",
        "model": "llama-3.2-90b-vision-preview",
        "max_tokens": 1024,
    },
    "hf-inference": {
        "provider": "huggingface",
        "model": "Qwen/Qwen2-VL-7B-Instruct",
        "max_tokens": 512,
    },
}
