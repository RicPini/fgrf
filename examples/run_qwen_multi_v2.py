#!/usr/bin/env python3
"""
Minimal example: Run FGRF v2 on Qwen2.5-0.5B with multiple inputs.
Equivalent to: python3 fgrf_multi_input_v2.py --model Qwen/Qwen2.5-0.5B --inputs 5 --baseline none
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fgrf_multi_input_v2 import FGRFEngine, FGRFConfig, TRAIN_TEXTS, EVAL_TEXTS
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    print("=" * 60)
    print("FGRF v2 Example: Qwen2.5-0.5B (Multiple Inputs)")
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
    config = FGRFConfig(
        max_layers_to_process=25,
        alpha=0.001,
        k_UV=100.0,
        k_IR=1.0,
        num_steps=100,
        output_dir='./fgrf_multi_results',
        ablation_mode='full',
        batch_size=2,  # Smaller batch for multi-input
        use_fast_hausdorff=True
    )
    
    # 3. Run on first 5 inputs
    train_texts = TRAIN_TEXTS[:5]
    eval_texts = EVAL_TEXTS[:5]
    
    print(f"\n[2] Testing on {len(train_texts)} training inputs...")
    print(f"    Evaluating on {len(eval_texts)} test inputs...")
    
    engine = FGRFEngine(model, tokenizer, config)
    results = engine.run(train_texts, eval_texts)
    
    # 4. Summary
    if results and 'improvement_percent' in results:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Improvement:         {results['improvement_percent']:+.1f}%")
        print(f"  Baseline perplexity: {results['baseline']['mean_perplexity']:.4f}")
        print(f"  Final perplexity:    {results['final']['mean_perplexity']:.4f}")
        print(f"  Layers processed:    {results['processed_layers']}")
        print(f"  Thermo rejections:   {results['thermo_rejections']}")
        print(f"  Thermo accepts:      {results['thermo_accepts']}")
        print(f"  Elapsed time:        {results['elapsed_time']:.1f}s")
        
        # Show duality analysis if available
        if 'duality_stats' in results and results['duality_stats']['D_values']:
            D_values = results['duality_stats']['D_values']
            print(f"\n  Duality Analysis:")
            print(f"    Mean D: {sum(D_values)/len(D_values):.3f}")
            print(f"    D range: [{min(D_values):.3f}, {max(D_values):.3f}]")

if __name__ == "__main__":
    import torch
    main()
