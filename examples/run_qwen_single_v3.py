#!/usr/bin/env python3
"""
Minimal example: Run FGRF v3 on Qwen2.5-0.5B with the default text set.
Equivalent to:
    python3 fgrf_multi_model_v3.py --model Qwen/Qwen2.5-0.5B --layers 20
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
    print("FGRF v3 Example: Qwen2.5-0.5B (single input set)")
    print("=" * 60)

    model_name = "Qwen/Qwen2.5-0.5B"

    print("\n[1] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
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
