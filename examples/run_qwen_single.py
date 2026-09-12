#!/usr/bin/env python3
"""
Minimal example: Run FGRF on Qwen2.5-0.5B with a single input.
Equivalent to: python3 fgrf_multi_model.py --model Qwen/Qwen2.5-0.5B --layers 20
"""

import sys
import os

# Add parent directory to path so we can import the main script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fgrf_multi_model import FGRFEngine
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    print("=" * 60)
    print("FGRF Example: Qwen2.5-0.5B (Single Input)")
    print("=" * 60)
    
    # 1. Load model
    print("\n[1] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Configure
    config = {
        'max_layers_to_process': 20,
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
