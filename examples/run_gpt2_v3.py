#!/usr/bin/env python3
"""
Minimal example: Run FGRF v3 on GPT-2-medium.
Equivalent to:
    python3 fgrf_multi_model_v3.py --model openai-community/gpt2-medium --layers 20
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fgrf_multi_model_v3 import (
    FGRFEngine,
    FGRFConfig,
    TRAIN_TEXTS,
    EVAL_TEXTS,
    GENERATION_PROMPTS,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    print("=" * 60)
    print("FGRF v3 Example: GPT-2-medium")
    print("=" * 60)

    model_name = "openai-community/gpt2-medium"

    print("\n[1] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = FGRFConfig(
        max_layers_to_process=20,
        output_dir="./fgrf_results",
        ablation_mode="full",
        dtype="float32",
        batch_size=4,
        seed=42,
    )

    engine = FGRFEngine(model, tokenizer, config)
    results = engine.run(TRAIN_TEXTS, EVAL_TEXTS, GENERATION_PROMPTS)

    if results and "improvement_percent" in results and results["improvement_percent"] is not None:
        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)
        print(f"  Baseline perplexity: {results['baseline']['mean_perplexity']:.4f}")
        print(f"  Final perplexity:    {results['final']['mean_perplexity']:.4f}")
        print(f"  Improvement:         {results['improvement_percent']:+.1f}%")
        print(f"  Layers processed:    {results['processed_layers']}")
        print(f"  Thermo rejections:   {results['thermo_rejections']}")
        print(f"  Thermo accepts:      {results['thermo_accepts']}")


if __name__ == "__main__":
    main()
