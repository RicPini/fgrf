#!/usr/bin/env python3
"""
FGRF Multi-Model Testing Script
Run the Fractal Graph Rewriting Framework on any Hugging Face model.

Usage:
    python3 fgrf_multi_model.py --model gpt2
    python3 fgrf_multi_model.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python3 fgrf_multi_model.py --model Qwen/Qwen2.5-0.5B --layers 30
    
Author: Riccardo Pini
"""

import torch
import numpy as np
import argparse
import json
import os
import gc
import warnings
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
warnings.filterwarnings('ignore')

# ============================================================================
# CORE FGRF MATH FUNCTIONS (Same as before)
# ============================================================================

def solve_fractal_flow(k_UV, k_IR, C_func, num_steps=100):
    """Solve: k * dQ/dk = -C(k) * Q * (1 - Q)"""
    k_values = np.logspace(np.log10(k_IR), np.log10(k_UV), num_steps)
    Q = np.zeros(num_steps)
    Q[0] = 0.99
    
    for i in range(num_steps - 1):
        k = k_values[i]
        dk = k_values[i+1] - k_values[i]
        C_k = C_func(k)
        dQ_dk = -C_k * Q[i] * (1 - Q[i]) / k
        Q[i+1] = Q[i] + dQ_dk * dk
        Q[i+1] = max(1e-8, min(1.0 - 1e-8, Q[i+1]))
    
    I = np.zeros(len(k_values))
    for i in range(1, len(k_values) - 1):
        dlogQ = np.log(max(Q[i+1], 1e-10) / max(Q[i-1], 1e-10))
        dlogk = np.log(k_values[i+1] / k_values[i-1])
        I[i] = -dlogQ / dlogk if dlogk > 0 else 0.0
    I[0] = I[1] if len(I) > 1 else 0.0
    I[-1] = I[-2] if len(I) > 1 else 0.0
    I = np.clip(I, 0.0, 100.0)
    I = np.nan_to_num(I, nan=0.0, posinf=10.0, neginf=0.0)
    
    return k_values, Q, I

def compute_topological_complexity(weight_matrix, k):
    """C(k) = d log N(k^{-1}) / d log k"""
    if torch.isnan(weight_matrix).any() or torch.isinf(weight_matrix).any():
        return 1.0
    
    try:
        weight_matrix_clean = torch.nan_to_num(weight_matrix.detach().clone(), 
                                               nan=0.0, posinf=1.0, neginf=-1.0)
        U, S, Vt = torch.linalg.svd(weight_matrix_clean, full_matrices=False)
        S = S.detach().numpy()
    except:
        try:
            S = np.linalg.svd(weight_matrix.detach().numpy(), compute_uv=False)
            S = np.nan_to_num(S, nan=0.0, posinf=1.0, neginf=0.0)
        except:
            return 1.0
    
    S = np.nan_to_num(S, nan=0.0, posinf=1.0, neginf=0.0)
    Lambda = 1.0 / max(k, 1e-10)
    N = np.sum(S > Lambda)
    N = max(N, 1.0)
    
    d = max(len(S), 1)
    C = d * N / (d + 1)
    return float(min(max(C, 0.1), 1000.0))

def compute_hausdorff_dimension(weight_matrix):
    """D ≈ 2 * (sum of singular values) / (max singular value)"""
    if torch.isnan(weight_matrix).any() or torch.isinf(weight_matrix).any():
        return 3.0
    
    try:
        weight_matrix_clean = torch.nan_to_num(weight_matrix.detach().clone(),
                                               nan=0.0, posinf=1.0, neginf=-1.0)
        U, S, Vt = torch.linalg.svd(weight_matrix_clean, full_matrices=False)
        S = S.detach().numpy()
    except:
        try:
            S = np.linalg.svd(weight_matrix.detach().numpy(), compute_uv=False)
            S = np.nan_to_num(S, nan=0.0, posinf=1.0, neginf=0.0)
        except:
            return 3.0
    
    total = np.sum(S)
    max_s = S[0] if len(S) > 0 and S[0] > 0 else 1.0
    
    if total > 0 and max_s > 0:
        D = 2 * total / max_s
    else:
        D = 3.0
    
    return float(min(max(D, 0.1), 100.0))

def compute_plastic_hysteresis(gradients, alpha=0.01):
    """H_n = α * Σ ||∂V ε||²"""
    H = 0.0
    for grad in gradients:
        if grad is not None:
            grad_clean = torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)
            H += float(torch.norm(grad_clean).item() ** 2)
    return alpha * H

def compute_semantic_entropy(activations):
    """S = -∫ p(a) log p(a) da"""
    if len(activations) == 0:
        return 0.0
    
    flat = activations.flatten().detach().numpy()
    flat = np.nan_to_num(flat, nan=0.0, posinf=1.0, neginf=-1.0)
    
    hist, _ = np.histogram(flat, bins=20)
    hist = hist / (np.sum(hist) + 1e-10)
    hist = hist[hist > 0]
    
    if len(hist) == 0:
        return 0.0
    
    return float(-np.sum(hist * np.log(hist + 1e-10)))

# ============================================================================
# FGRF LAYER PROCESSOR
# ============================================================================

def fgrf_process_layer(model, tokenizer, layer_idx, param, param_name, 
                       input_ids, k_values, Q, I, alpha=0.001,
                       k_B=1.38e-23, T=300.0, a=0.01):
    """Apply FGRF math to a single layer."""
    
    prev_weight = param.data.clone()
    
    D_initial = compute_hausdorff_dimension(param.data)
    D_dual_initial = 6 - D_initial
    
    def C_func(k):
        return compute_topological_complexity(param.data, k)
    
    k_idx = min(int(layer_idx / 10) * len(k_values) // 10, len(k_values) - 1)
    k = k_values[k_idx]
    Q_k = Q[k_idx]
    I_k = I[k_idx]
    C_k = C_func(k)
    
    model.zero_grad()
    model.train()
    
    try:
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        
        if loss is None:
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        loss.backward()
    except Exception as e:
        return False, {'error': str(e)}, 0.0
    
    if param.grad is None:
        return False, {'error': 'No gradient'}, 0.0
    
    grad = param.grad
    grad_norm = float(torch.norm(grad).item())
    
    Q_factor = float(Q_k * (1 - Q_k))
    Q_factor = min(max(Q_factor, 0.0), 0.25)
    C_factor = float(min(max(C_k, 0.1), 100.0))
    I_factor = float(min(max(I_k, 0.1), 10.0))
    
    scale_factor = alpha * Q_factor * C_factor * I_factor
    scale_factor = min(max(scale_factor, 1e-8), 0.1)
    
    grad_clean = torch.nan_to_num(grad, nan=0.0, posinf=0.1, neginf=-0.1)
    delta = -scale_factor * grad_clean
    param.data = param.data + delta
    param.data = torch.clamp(param.data, -10.0, 10.0)
    
    H_step = compute_plastic_hysteresis([grad])
    
    with torch.no_grad():
        try:
            outputs = model(input_ids=input_ids)
            activations = outputs.logits
            S_current = compute_semantic_entropy(activations)
        except:
            S_current = 0.0
    
    dS = S_current - 1.0
    dH = H_step
    
    if dS > 0:
        constraint_threshold = k_B * T * dS * 1e10
        if dH < constraint_threshold:
            param.data = prev_weight
            return False, {'error': 'Thermodynamic constraint violated'}, H_step
    
    diff = param.data - prev_weight
    numerator = torch.norm(diff).item()
    denominator = torch.norm(prev_weight).item() + 1e-10
    rel_change = numerator / denominator
    
    threshold = np.sqrt(k_B * T / a) * H_step * 1e10
    
    D_final = compute_hausdorff_dimension(param.data)
    D_dual_final = 6 - D_final
    
    param.grad = None
    gc.collect()
    
    history = {
        'layer_idx': int(layer_idx),
        'param_name': str(param_name),
        'k': float(k),
        'Q_k': float(Q_k),
        'I_k': float(I_k),
        'C_k': float(C_k),
        'scale_factor': float(scale_factor),
        'grad_norm': float(grad_norm),
        'H_step': float(H_step),
        'rel_change': float(rel_change),
        'threshold': float(threshold),
        'stopped': bool(rel_change < threshold and rel_change > 1e-12),
        'D_initial': float(D_initial),
        'D_dual_initial': float(D_dual_initial),
        'D_final': float(D_final),
        'D_dual_final': float(D_dual_final),
        'S_semantic': float(S_current)
    }
    
    return True, history, H_step

# ============================================================================
# FGRF ENGINE
# ============================================================================

class FGRFEngine:
    def __init__(self, model, tokenizer, config=None):
        self.model = model
        self.tokenizer = tokenizer
        
        self.config = {
            'k_UV': 100.0,
            'k_IR': 1.0,
            'num_steps': 100,
            'alpha': 0.001,
            'k_B': 1.38e-23,
            'T': 300.0,
            'a': 0.01,
            'max_layers_to_process': 50,
            'output_dir': './fgrf_results'
        }
        if config:
            self.config.update(config)
        
        self.results = []
        self.H_total = 0.0
        self.layer_history = {}
        
        os.makedirs(self.config['output_dir'], exist_ok=True)
    
    def get_weight_params(self, max_layers=None):
        """Get weight parameters, limited to max_layers."""
        params = []
        skip_patterns = ['embed_tokens', 'wte', 'embedding', 'lm_head', 'output', 'head']
        
        max_layers = max_layers or self.config.get('max_layers_to_process', 50)
        count = 0
        
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.requires_grad:
                should_skip = any(p in name.lower() for p in skip_patterns)
                if not should_skip and count < max_layers:
                    params.append((name, param))
                    count += 1
        
        return params
    
    def get_model_layers(self):
        """Find transformer layers dynamically."""
        try:
            return self.model.model.layers
        except AttributeError:
            try:
                return self.model.layers
            except AttributeError:
                try:
                    return self.model.transformer.h
                except AttributeError:
                    try:
                        return self.model.encoder.layer
                    except:
                        return None
    
    def run(self, input_text=None):
        """Run FGRF rewriting."""
        print("\n" + "=" * 80)
        print("FGRF MULTI-MODEL ENGINE")
        print("=" * 80)
        
        if input_text is None:
            input_text = "The neural network learns patterns from data."
        
        print(f"\n[0] Input text: {input_text[:60]}...")
        print(f"    Model: {self.model.__class__.__name__}")
        
        layers = self.get_model_layers()
        if layers is None:
            print("    ERROR: Could not find model layers!")
            return None
        
        num_layers = len(layers)
        print(f"    Total transformer layers: {num_layers}")
        
        max_layers = self.config.get('max_layers_to_process', 50)
        params = self.get_weight_params(max_layers=max_layers)
        print(f"    Processing {len(params)} layers (limit: {max_layers})")
        
        inputs = self.tokenizer(input_text, return_tensors="pt", 
                                truncation=True, max_length=32)
        input_ids = inputs["input_ids"]
        
        print(f"\n[1] Baseline evaluation...")
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            baseline_loss = outputs.loss
            if baseline_loss is not None:
                baseline_perp = torch.exp(baseline_loss).item()
                print(f"    Baseline perplexity: {baseline_perp:.4f}")
            else:
                baseline_perp = None
                print("    Baseline perplexity: N/A")
        
        print(f"\n[2] Running FGRF rewriting on {len(params)} layers...")
        print("    " + "-" * 70)
        
        first_param = params[0][1] if params else None
        if first_param is not None:
            def C_func_global(k):
                return compute_topological_complexity(first_param.data, k)
            
            k_values, Q, I = solve_fractal_flow(
                self.config['k_UV'],
                self.config['k_IR'],
                C_func_global,
                self.config['num_steps']
            )
            
            print(f"    Fractal Flow: Q_UV={Q[0]:.4f}, Q_IR={Q[-1]:.4f}")
        
        total_processed = 0
        total_skipped = 0
        H_total = 0.0
        
        for idx, (name, param) in enumerate(params):
            if idx % 10 == 0 or idx == len(params) - 1:
                print(f"    Layer {idx+1}/{len(params)}: {name[:40]}...")
            
            success, history, H_step = fgrf_process_layer(
                self.model, self.tokenizer, idx, param, name,
                input_ids, k_values, Q, I,
                alpha=self.config['alpha'],
                k_B=self.config['k_B'],
                T=self.config['T'],
                a=self.config['a']
            )
            
            if success:
                total_processed += 1
                H_total += H_step
                self.layer_history[name] = history
                self.results.append(history)
            else:
                total_skipped += 1
            
            if idx % 20 == 0:
                gc.collect()
        
        self.H_total = H_total
        
        print(f"\n[3] Final evaluation...")
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            final_loss = outputs.loss
            if final_loss is not None:
                final_perp = torch.exp(final_loss).item()
                print(f"    Final perplexity: {final_perp:.4f}")
            else:
                final_perp = None
                print("    Final perplexity: N/A")
        
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        
        print(f"\n[4] Rewriting Statistics")
        print(f"    Layers processed: {total_processed}")
        print(f"    Layers skipped: {total_skipped}")
        print(f"    Total plastic hysteresis: {H_total:.6f}")
        
        if baseline_perp is not None and final_perp is not None:
            print(f"\n[5] Perplexity")
            print(f"    Baseline: {baseline_perp:.4f}")
            print(f"    Final:    {final_perp:.4f}")
            print(f"    Change:   {final_perp - baseline_perp:+.4f}")
            print(f"    Improvement: {'✓' if final_perp < baseline_perp else '✗'}")
        
        # Save results
        results_data = {
            'model_name': self.model.__class__.__name__,
            'timestamp': datetime.now().isoformat(),
            'processed_layers': total_processed,
            'skipped_layers': total_skipped,
            'H_total': float(H_total),
            'baseline_perplexity': float(baseline_perp) if baseline_perp else None,
            'final_perplexity': float(final_perp) if final_perp else None,
            'config': self.config
        }
        
        output_path = os.path.join(self.config['output_dir'], 
                                   f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"\n[6] Results saved to: {output_path}")
        
        return results_data

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FGRF Multi-Model Testing")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="Model name (e.g., gpt2, TinyLlama/TinyLlama-1.1B-Chat-v1.0)")
    parser.add_argument("--layers", type=int, default=20,
                        help="Number of layers to process (default: 20)")
    parser.add_argument("--text", type=str, default="The neural network learns patterns from data.",
                        help="Input text for evaluation")
    parser.add_argument("--output", type=str, default="./fgrf_results",
                        help="Output directory")
    parser.add_argument("--4bit", action="store_true",
                        help="Use 4-bit quantization (for large models)")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("FGRF DEPLOYMENT")
    print("=" * 80)
    print(f"\nModel: {args.model}")
    print(f"Layers to process: {args.layers}")
    
    print("\n[Loading model...]")
    
    try:
        if getattr(args, '_4bit', False):
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                quantization_config=quantization_config,
                device_map="cpu",
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTrying fallback to GPT-2...")
        model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    config = {
        'max_layers_to_process': args.layers,
        'output_dir': args.output,
        'alpha': 0.001,
        'k_UV': 100.0,
        'k_IR': 1.0,
        'num_steps': 100
    }
    
    engine = FGRFEngine(model, tokenizer, config)
    results = engine.run(input_text=args.text)
    
    print("\n" + "=" * 80)
    print("✓ FGRF DEPLOYMENT COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
