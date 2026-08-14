# FREE VLM Models - Working Setup Guide

## ⚠️ Important: Model Size Issues

**Qwen2.5-VL-7B is 16GB** - too large for free HuggingFace API!

Use these **working alternatives** instead:

---

## ✅ Option 1: Ollama (BEST - 100% FREE)

**No API limits, completely free, runs locally**

### Setup (5 minutes):

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull Qwen2.5-VL model (7B, ~5GB)
ollama pull qwen2.5-vl:7b

# 3. Install Python client
pip install ollama
```

### Usage:

```python
from vlm import VLMAnalyzer, VLMConfig

config = VLMConfig(
    api_provider="ollama",
    api_base="http://localhost:11434",
    local_model="qwen2.5-vl:7b"
)
analyzer = VLMAnalyzer(config=config)

# Test it
import pandas as pd
pet_df = pd.read_csv("outputs/petevents_recovered.csv")
results = analyzer.analyze_pet_events(pet_df.head(5))
print(f"Analyzed {len(results)} events")
```

---

## ✅ Option 2: HuggingFace (FREE - Smaller Models)

**Free tier available, but limited to smaller models**

### Setup:

```bash
# Get token: https://huggingface.co/settings/tokens
pip install huggingface_hub
```

### Working Models:

```python
from vlm import VLMAnalyzer, VLMConfig

# Option A: Qwen2-VL-2B (smallest, fastest)
config = VLMConfig(
    api_provider="huggingface",
    model_name="Qwen/Qwen2-VL-2B-Instruct",
    api_key="hf_xxx"
)

# Option B: Qwen2-VL-7B (better quality, may need Pro)
config = VLMConfig(
    api_provider="huggingface",
    model_name="Qwen/Qwen2-VL-7B-Instruct",
    api_key="hf_xxx"
)

analyzer = VLMAnalyzer(config=config)
```

---

## ✅ Option 3: Groq (FREE Tier - Fast)

**Free tier with rate limits, very fast**

### Setup:

```bash
# Get free API key: https://console.groq.com
pip install groq
```

### Usage:

```python
from vlm import VLMAnalyzer, VLMConfig

config = VLMConfig(
    api_provider="groq",
    model_name="llama-3.2-11b-vision-preview",
    api_key="gsk_xxx"  # Your free key from console.groq.com
)
analyzer = VLMAnalyzer(config=config)
```

---

## 🏆 Recommendation

**Use Ollama** - it's:
- ✅ 100% FREE
- ✅ No API limits
- ✅ No internet needed after download
- ✅ Qwen2.5-VL-7B works perfectly
- ✅ ~5GB download, runs on 8GB RAM

**Quick Start with Ollama:**

```bash
# Install
curl -fsSL https://ollama.com/install.sh | sh

# Pull model
ollama pull qwen2.5-vl:7b

# Test
ollama run qwen2.5-vl:7b "What is 2+2?"
```

Then use the Python code above!

---

## Model Comparison

| Model | Size | Free? | Speed | Quality |
|-------|------|-------|-------|---------|
| Ollama Qwen2.5-VL-7B | 7B | ✅ Yes | Fast | ⭐⭐⭐⭐⭐ |
| HF Qwen2-VL-2B | 2B | ✅ Yes | Very Fast | ⭐⭐⭐ |
| HF Qwen2-VL-7B | 7B | ⚠️ Limited | Fast | ⭐⭐⭐⭐ |
| Groq Llama-3.2-11B | 11B | ✅ Free tier | Very Fast | ⭐⭐⭐⭐⭐ |
