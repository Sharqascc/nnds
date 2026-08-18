#!/usr/bin/env python3
"""
Test VLM Module with FREE Models
================================

Run this to test different free VLM providers.
"""

import sys
from pathlib import Path
import pandas as pd

# Add current directory
sys.path.insert(0, str(Path(__file__).parent))

from vlm import VLMAnalyzer, VLMConfig

def test_ollama():
    """Test with Ollama (100% FREE, recommended)"""
    print("\n" + "="*60)
    print("Testing Ollama (Qwen2.5-VL-7B)")
    print("="*60)
    
    try:
        config = VLMConfig(
            api_provider="ollama",
            api_base="http://localhost:11434",
            local_model="qwen2.5-vl:7b"
        )
        analyzer = VLMAnalyzer(config=config)
        
        # Load PET events
        pet_df = pd.read_csv("outputs/petevents_recovered.csv")
        print(f"Loaded {len(pet_df)} PET events")
        
        # Analyze first 5
        print("Analyzing 5 events...")
        results = analyzer.analyze_pet_events(pet_df.head(5))
        
        print(f"\n✓ Successfully analyzed {len(results)} events!")
        for r in results:
            print(f"  Event {r.event_id}: PET={r.pet:.3f}s, Severity={r.severity}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Make sure Ollama is running: ollama run qwen2.5-vl:7b")
        return False


def test_huggingface_small():
    """Test with HuggingFace Qwen2-VL-2B (FREE)"""
    print("\n" + "="*60)
    print("Testing HuggingFace (Qwen2-VL-2B)")
    print("="*60)
    
    try:
        config = VLMConfig(
            api_provider="huggingface",
            model_name="Qwen/Qwen2-VL-2B-Instruct",
            api_key="hf_xxx"  # Replace with your token
        )
        analyzer = VLMAnalyzer(config=config)
        
        # Load PET events
        pet_df = pd.read_csv("outputs/petevents_recovered.csv")
        print(f"Loaded {len(pet_df)} PET events")
        
        # Analyze first 5
        print("Analyzing 5 events...")
        results = analyzer.analyze_pet_events(pet_df.head(5))
        
        print(f"\n✓ Successfully analyzed {len(results)} events!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Make sure you have a valid HuggingFace token")
        return False


def test_groq():
    """Test with Groq (FREE tier)"""
    print("\n" + "="*60)
    print("Testing Groq (Llama-3.2-11B)")
    print("="*60)
    
    try:
        config = VLMConfig(
            api_provider="groq",
            model_name="llama-3.2-11b-vision-preview",
            api_key="gsk_xxx"  # Replace with your key
        )
        analyzer = VLMAnalyzer(config=config)
        
        # Load PET events
        pet_df = pd.read_csv("outputs/petevents_recovered.csv")
        print(f"Loaded {len(pet_df)} PET events")
        
        # Analyze first 5
        print("Analyzing 5 events...")
        results = analyzer.analyze_pet_events(pet_df.head(5))
        
        print(f"\n✓ Successfully analyzed {len(results)} events!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Make sure you have a valid Groq API key from console.groq.com")
        return False


def main():
    print("\n" + "="*70)
    print("VLM Module - FREE Model Test")
    print("="*70)
    
    print("\nChoose a provider to test:")
    print("1. Ollama (100% FREE, recommended)")
    print("2. HuggingFace Qwen2-VL-2B (FREE)")
    print("3. Groq (FREE tier)")
    print("4. Test all")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        test_ollama()
    elif choice == "2":
        test_huggingface_small()
    elif choice == "3":
        test_groq()
    elif choice == "4":
        print("\n" + "="*70)
        print("Testing all providers...")
        print("="*70)
        
        results = []
        results.append(("Ollama", test_ollama()))
        results.append(("HuggingFace", test_huggingface_small()))
        results.append(("Groq", test_groq()))
        
        print("\n" + "="*70)
        print("Test Results:")
        print("="*70)
        for name, success in results:
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"{status} - {name}")
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
