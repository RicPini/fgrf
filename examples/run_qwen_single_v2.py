#!/usr/bin/env python3
"""
Minimal example: Run FGRF v2 on Qwen2.5-0.5B with a single input.
Equivalent to: python3 fgrf_multi_model_v2.py --model Qwen/Qwen2.5-0.5B --layers 20
"""

import sys
import os

# Add parent directory to path so we can import the main script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fgrf_multi_model_v2 import FGRFEngine, FGRFConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    print("=" * 60)
    print("FGRF v2 Example: Qwen2.5-0.5B (Single Input)")
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
    
    # 2. Configure with fast approximations for efficiency
    config = FGRFConfig(
        max_layers_to_process=20,
        alpha=0.001,
        k_UV=100.0,
        k_IR=1.0,
        num_steps=100,
        output_dir='./fgrf_results',
        ablation_mode='full',
        use_fast_hausdorff=True,  # Faster computation
        use_robust_ode=True,
        dtype='float32'
    )
    
    # 3. Prepare texts
    train_texts = [
        "The neural network learns patterns from data through iterative optimization.",
        "Machine learning models require large datasets for effective training.",
        "Deep learning has revolutionized computer vision and natural language processing.",
        "Gradient descent is a fundamental optimization algorithm in neural networks.",
        "Transformers use attention mechanisms to process sequences efficiently."
    ]
    
    eval_texts = [
        "Quantum mechanics describes the behavior of particles at the smallest scales.",
        "The Industrial Revolution transformed manufacturing and society in the 19th century.",
        "Photosynthesis is the process by which plants convert light into energy.",
        "The human genome consists of approximately three billion base pairs.",
        "Ancient civilizations developed writing systems for record keeping."
    ]
    
    # 4. Run
    engine = FGRFEngine(model, tokenizer, config)
    results = engine.run(train_texts, eval_texts)
    
    # 5. Show result
    if results and 'improvement_percent' in results:
        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)
        print(f"  Baseline perplexity: {results['baseline']['mean_perplexity']:.4f}")
        print(f"  Final perplexity:    {results['final']['mean_perplexity']:.4f}")
        print(f"  Improvement:         {results['improvement_percent']:+.1f}%")
        print(f"  Layers processed:    {results['processed_layers']}")
        print(f"  Total updates:       {results['total_updates']}")
        print(f"  Elapsed time:        {results['elapsed_time']:.1f}s")

if __name__ == "__main__":
    import torch
    main()
