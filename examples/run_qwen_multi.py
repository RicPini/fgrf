#!/usr/bin/env python3
"""
Minimal example: Run FGRF on Qwen2.5-0.5B with multiple inputs.
Equivalent to: python3 fgrf_multi_input.py --model Qwen/Qwen2.5-0.5B --inputs 5 --baseline none
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fgrf_multi_input import FGRFEngine, SAMPLE_TEXTS
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    print("=" * 60)
    print("FGRF Example: Qwen2.5-0.5B (Multiple Inputs)")
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
        'max_layers_to_process': 25,
        'alpha': 0.001,
        'k_UV': 100.0,
        'k_IR': 1.0,
        'num_steps': 100,
        'output_dir': './fgrf_multi_results'
    }
    
    # 3. Run on first 5 inputs
    texts = SAMPLE_TEXTS[:5]
    print(f"\n[2] Testing on {len(texts)} inputs...")
    
    engine = FGRFEngine(model, tokenizer, config)
    
    improvements = []
    for i, text in enumerate(texts):
        result, baseline, final = engine.run(text)
        if result and baseline is not None and final is not None:
            imp = (final - baseline) / baseline * 100
            improvements.append(imp)
            print(f"    Input {i+1}: {baseline:.4f} → {final:.4f} ({imp:+.1f}%)")
    
    # 4. Summary
    if improvements:
        import numpy as np
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Mean improvement: {np.mean(improvements):+.1f}%")
        print(f"  Std deviation:    {np.std(improvements):+.1f}%")
        print(f"  Samples:          {len(improvements)}")

if __name__ == "__main__":
    import torch
    import numpy as np
    main()
