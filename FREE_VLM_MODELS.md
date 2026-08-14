# FREE VLM Models - Quick Start Guide

## 🆓 Best Free & Open-Source VLM Models

### Option 1: HuggingFace Inference API (RECOMMENDED)

**Free tier available** - No credit card needed for testing

```bash
# Install
pip install huggingface_hub

# Get free token: https://huggingface.co/settings/tokens
```

**Best models:**
- `Qwen/Qwen2.5-VL-7B-Instruct` - Best overall (Apache 2.0)
- `meta-llama/Llama-3.2-11B-Vision-Instruct` - Great reasoning
- `deepseek-ai/DeepSeek-VL` - Most efficient

**Usage:**
```python
from vlm import VLMAnalyzer, VLMConfig

config = VLMConfig(
    api_provider="huggingface",
    model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    api_key="hf_xxx"  # Your free token
)
analyzer = VLMAnalyzer(config=config)
```

### Option 2: Ollama (100% FREE - Local)

**Completely free** - Runs on your machine

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull model
ollama run qwen2.5-vl:7b

# Install Python client
pip install ollama
```

**Usage:**
```python
from vlm import VLMAnalyzer, VLMConfig

config = VLMConfig(
    api_provider="ollama",
    api_base="http://localhost:11434",
    local_model="qwen2.5-vl:7b"
)
analyzer = VLMAnalyzer(config=config)
```

### Option 3: Groq (FREE tier)

**Free tier** - Fast inference, limited requests

```bash
# Get free API key: https://console.groq.com
pip install groq
```

**Usage:**
```python
from vlm import VLMAnalyzer, VLMConfig

config = VLMConfig(
    api_provider="groq",
    model_name="llama-3.2-11b-vision-preview",
    api_key="gsk_xxx"
)
analyzer = VLMAnalyzer(config=config)
```

## Model Comparison

| Model | Size | License | Speed | Quality | Best For |
|-------|------|---------|-------|---------|----------|
| Qwen 2.5 VL | 7B | Apache 2.0 | Fast | ⭐⭐⭐⭐⭐ | Overall use |
| Llama 3.2 Vision | 11B | Free | Medium | ⭐⭐⭐⭐⭐ | Reasoning |
| DeepSeek-VL | 7B | MIT | Fast | ⭐⭐⭐⭐ | Efficiency |
| Gemma 3 | 12B | Open | Medium | ⭐⭐⭐⭐ | Lightweight |

## Quick Test

```python
from vlm import VLMConfig, VLMAnalyzer

# Use pre-configured free model
config = VLMConfig.for_free_model("qwen")
analyzer = VLMAnalyzer(config=config)

# Analyze PET events
import pandas as pd
pet_df = pd.read_csv("outputs/petevents_recovered.csv")
results = analyzer.analyze_pet_events(pet_df.head(10))
```

## Get Started Now

1. **HuggingFace**: Get token at https://huggingface.co/settings/tokens
2. **Ollama**: Install at https://ollama.com
3. **Groq**: Get key at https://console.groq.com

**Recommended**: Start with **Qwen 2.5 VL** on HuggingFace - it's free, fast, and excellent quality!
