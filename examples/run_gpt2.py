#!/usr/bin/env python3
"""
Minimal example: Run FGRF on GPT-2.
Equivalent to: python3 fgrf_multi_model.py --model gpt2 --layers 12
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fgrf_multi_model import FGRFEngine
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    print("=" * 60)
    print("FGRF Example: GPT-2")
    print("=" * 60)
    
    # 1. Load model
    print("\n[1] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Configure
    config = {
        'max_layers_to_process': 12,
        'alpha': 0.001,
        'k_UV': 100.0,
        'k_IR': 1.0,
        'num_steps': 100,
        'output_dir': './fgrf_results'
    }
    
    # 3. Run
    engine = FGRFEngine(model, tokenizer, config)
    results = engine.run(
        input_text="The neural network learns patterns from data."
    )
    
    # 4. Show result
    if results:
        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)
        print(f"  Baseline perplexity: {results['baseline_perplexity']:.4f}")
        print(f"  Final perplexity:    {results['final_perplexity']:.4f}")
        print(f"  Improvement:         {(results['final_perplexity'] - results['baseline_perplexity']) / results['baseline_perplexity'] * 100:+.1f}%")

if __name__ == "__main__":
    import torch
    main()
