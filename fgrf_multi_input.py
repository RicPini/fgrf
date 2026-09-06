#!/usr/bin/env python3
"""
FGRF Multi-Input Testing Script
Runs FGRF on multiple inputs and compares to baselines.
Designed for CPU-only, low-memory hardware.

Usage:
    python3 fgrf_multi_input.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python3 fgrf_multi_input.py --model gpt2 --inputs 10
    python3 fgrf_multi_input.py --model Qwen/Qwen2.5-0.5B --baseline random
    
Author: Riccardo Pini
"""

import torch
import numpy as np
import argparse
import json
import os
import gc
import time
import warnings
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
warnings.filterwarnings('ignore')

# ============================================================================
# SAMPLE INPUT TEXTS (Diverse topics for testing generalization)
# ============================================================================

SAMPLE_TEXTS = [
    # General knowledge
    "The theory of relativity fundamentally changed our understanding of space and time.",
    "Photosynthesis converts light energy into chemical energy in plants and algae.",
    "The human brain contains approximately 86 billion neurons and 100 trillion synapses.",
    
    # Technical/Programming
    "A transformer neural network uses attention mechanisms to process sequential data.",
    "The PyTorch framework provides automatic differentiation for deep learning models.",
    "Recursive algorithms solve problems by reducing them to smaller instances of the same problem.",
    
    # Creative/Literary
    "In the quiet hours of dawn, the city slowly awakened to the rhythm of distant birds.",
    "The old library contained thousands of forgotten stories waiting to be rediscovered.",
    "She traced the constellations with her finger, connecting ancient myths to the night sky.",
    
    # Scientific
    "Entropy is a measure of disorder that increases in isolated systems over time.",
    "Quantum entanglement connects particles across vast distances instantaneously.",
    "DNA replication ensures genetic information is passed accurately to daughter cells.",
    
    # Everyday/Descriptive
    "A warm cup of coffee on a cold morning is one of life's simplest pleasures.",
    "The garden bloomed with roses, lavender, and wildflowers after the spring rain.",
    "Walking through the market, the smells of fresh bread and spices filled the air.",
    
    # Technical/Mathematical
    "The Fourier transform decomposes signals into their constituent frequencies.",
    "Matrix multiplication is fundamental to linear algebra and neural network operations.",
    "Bayesian inference updates probability estimates as new evidence becomes available.",
    
    # Historical
    "The Renaissance period marked a rebirth of art, science, and human inquiry in Europe.",
    "Ancient civilizations developed complex systems of writing, trade, and governance.",
    "The scientific revolution challenged traditional beliefs about the natural world."
]

# ============================================================================
# FGRF CORE MATH FUNCTIONS
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
# FGRF LAYER PROCESSOR (Optimized for CPU)
# ============================================================================

def fgrf_process_layer(model, tokenizer, layer_idx, param, param_name, 
                       input_ids, k_values, Q, I, alpha=0.001,
                       k_B=1.38e-23, T=300.0, a=0.01, verbose=False):
    """Apply FGRF math to a single layer."""
    
    prev_weight = param.data.clone()
    
    def C_func(k):
        return compute_topological_complexity(param.data, k)
    
    # Scale selection based on layer index
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
    
    # Scale factor
    Q_factor = float(Q_k * (1 - Q_k))
    Q_factor = min(max(Q_factor, 0.0), 0.25)
    C_factor = float(min(max(C_k, 0.1), 100.0))
    I_factor = float(min(max(I_k, 0.1), 10.0))
    
    scale_factor = alpha * Q_factor * C_factor * I_factor
    scale_factor = min(max(scale_factor, 1e-8), 0.1)
    
    grad_clean = torch.nan_to_num(grad, nan=0.0, posinf=0.1, neginf=-0.1)
    param.data = param.data - scale_factor * grad_clean
    param.data = torch.clamp(param.data, -10.0, 10.0)
    
    H_step = compute_plastic_hysteresis([grad])
    
    # Thermodynamic constraint check
    with torch.no_grad():
        try:
            outputs = model(input_ids=input_ids)
            S_current = compute_semantic_entropy(outputs.logits)
        except:
            S_current = 0.0
    
    dS = S_current - 1.0
    if dS > 0:
        constraint = k_B * T * dS * 1e10
        if H_step < constraint:
            param.data = prev_weight
            return False, {'error': 'Thermodynamic constraint violated'}, H_step
    
    param.grad = None
    gc.collect()
    
    return True, {}, H_step

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
            'max_layers_to_process': 30,
            'output_dir': './fgrf_multi_results'
        }
        if config:
            self.config.update(config)
        
        os.makedirs(self.config['output_dir'], exist_ok=True)
    
    def get_weight_params(self):
        """Get weight parameters, limited to max_layers."""
        params = []
        skip_patterns = ['embed_tokens', 'wte', 'embedding', 'lm_head', 'output', 'head']
        
        max_layers = self.config.get('max_layers_to_process', 30)
        count = 0
        
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.requires_grad:
                should_skip = any(p in name.lower() for p in skip_patterns)
                if not should_skip and count < max_layers:
                    params.append((name, param))
                    count += 1
        
        return params
    
    def run(self, input_text):
        """Run FGRF rewriting on a single input."""
        
        inputs = self.tokenizer(input_text, return_tensors="pt", 
                                truncation=True, max_length=32)
        input_ids = inputs["input_ids"]
        
        # Baseline
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            baseline_loss = outputs.loss
            if baseline_loss is not None:
                baseline_perp = torch.exp(baseline_loss).item()
            else:
                baseline_perp = None
        
        # Get parameters
        params = self.get_weight_params()
        
        # Solve Fractal Flow
        if params:
            first_param = params[0][1]
            def C_func_global(k):
                return compute_topological_complexity(first_param.data, k)
            
            k_values, Q, I = solve_fractal_flow(
                self.config['k_UV'],
                self.config['k_IR'],
                C_func_global,
                self.config['num_steps']
            )
        else:
            return None, baseline_perp, None
        
        # Process layers
        H_total = 0.0
        processed = 0
        
        for idx, (name, param) in enumerate(params):
            success, _, H_step = fgrf_process_layer(
                self.model, self.tokenizer, idx, param, name,
                input_ids, k_values, Q, I,
                alpha=self.config['alpha'],
                k_B=self.config['k_B'],
                T=self.config['T'],
                a=self.config['a']
            )
            
            if success:
                processed += 1
                H_total += H_step
            
            if idx % 10 == 0:
                gc.collect()
        
        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            final_loss = outputs.loss
            if final_loss is not None:
                final_perp = torch.exp(final_loss).item()
            else:
                final_perp = None
        
        result = {
            'processed': processed,
            'H_total': H_total,
            'baseline_perp': baseline_perp,
            'final_perp': final_perp
        }
        
        return result, baseline_perp, final_perp

# ============================================================================
# BASELINE COMPARISONS
# ============================================================================

def random_perturbation(model, tokenizer, input_text, strength=0.01):
    """Baseline: Random weight perturbation."""
    original_state = {name: param.data.clone() for name, param in model.named_parameters()}
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            noise = torch.randn_like(param.data) * strength
            param.data = param.data + noise
            param.data = torch.clamp(param.data, -10.0, 10.0)
    
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=32)
    input_ids = inputs["input_ids"]
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        if loss is not None:
            perp = torch.exp(loss).item()
        else:
            perp = None
    
    # Restore original
    for name, param in model.named_parameters():
        if name in original_state:
            param.data = original_state[name]
    
    return perp

def simple_gradient_descent(model, tokenizer, input_text, lr=0.001, steps=1):
    """Baseline: Simple gradient descent (no scale factors)."""
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=32)
    input_ids = inputs["input_ids"]
    
    params = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            params.append((name, param))
    
    for _ in range(steps):
        model.zero_grad()
        model.train()
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
        
        for name, param in params:
            if param.grad is not None:
                grad_clean = torch.nan_to_num(param.grad, nan=0.0, posinf=0.1, neginf=-0.1)
                param.data = param.data - lr * grad_clean
                param.data = torch.clamp(param.data, -10.0, 10.0)
                param.grad = None
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        if loss is not None:
            perp = torch.exp(loss).item()
        else:
            perp = None
    
    return perp

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="FGRF Multi-Input Testing")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Model name (e.g., gpt2, Qwen/Qwen2.5-0.5B)")
    parser.add_argument("--inputs", type=int, default=20,
                        help="Number of input texts to test (default: 20)")
    parser.add_argument("--layers", type=int, default=20,
                        help="Number of layers to process (default: 20)")
    parser.add_argument("--baseline", type=str, choices=['random', 'gd', 'both', 'none'],
                        default='both', help="Baseline comparisons to run")
    parser.add_argument("--output", type=str, default="./fgrf_multi_results",
                        help="Output directory")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("FGRF MULTI-INPUT TESTING")
    print("=" * 80)
    print(f"\nModel: {args.model}")
    print(f"Inputs: {args.inputs}")
    print(f"Layers: {args.layers}")
    
    # Load model
    print("\n[Loading model...]")
    try:
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
        print("Falling back to GPT-2...")
        model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Select inputs
    texts = SAMPLE_TEXTS[:args.inputs]
    print(f"\n[Testing on {len(texts)} inputs]")
    
    # Setup engine
    config = {
        'max_layers_to_process': args.layers,
        'output_dir': args.output,
        'alpha': 0.001,
        'k_UV': 100.0,
        'k_IR': 1.0,
        'num_steps': 100
    }
    engine = FGRFEngine(model, tokenizer, config)
    
    # Run tests
    results = []
    fgrf_perps = []
    random_perps = []
    gd_perps = []
    
    for i, text in enumerate(texts):
        print(f"\n  Input {i+1}/{len(texts)}: {text[:40]}...")
        
        # Save original model state
        original_state = {name: param.data.clone() for name, param in model.named_parameters()}
        
        # Run FGRF
        result, baseline, final = engine.run(text)
        if result:
            fgrf_perps.append(final)
            print(f"    FGRF: {baseline:.4f} → {final:.4f} ({(final-baseline)/baseline*100:+.1f}%)")
        
        # Restore model
        for name, param in model.named_parameters():
            if name in original_state:
                param.data = original_state[name]
        gc.collect()
        
        # Random perturbation baseline
        if args.baseline in ['random', 'both']:
            rand_perp = random_perturbation(model, tokenizer, text, strength=0.01)
            if rand_perp is not None:
                random_perps.append(rand_perp)
                print(f"    Random: {baseline:.4f} → {rand_perp:.4f} ({(rand_perp-baseline)/baseline*100:+.1f}%)")
            
            # Restore model
            for name, param in model.named_parameters():
                if name in original_state:
                    param.data = original_state[name]
            gc.collect()
        
        # Simple gradient descent baseline
        if args.baseline in ['gd', 'both']:
            gd_perp = simple_gradient_descent(model, tokenizer, text, lr=0.001, steps=1)
            if gd_perp is not None:
                gd_perps.append(gd_perp)
                print(f"    GD:     {baseline:.4f} → {gd_perp:.4f} ({(gd_perp-baseline)/baseline*100:+.1f}%)")
            
            # Restore model
            for name, param in model.named_parameters():
                if name in original_state:
                    param.data = original_state[name]
            gc.collect()
        
        results.append({
            'text': text[:60],
            'baseline': baseline,
            'fgrf': final,
            'random': rand_perp if args.baseline in ['random', 'both'] else None,
            'gd': gd_perp if args.baseline in ['gd', 'both'] else None
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    fgrf_improvements = [(r['fgrf'] - r['baseline']) / r['baseline'] * 100 
                         for r in results if r['fgrf'] is not None]
    random_improvements = [(r['random'] - r['baseline']) / r['baseline'] * 100 
                           for r in results if r['random'] is not None]
    gd_improvements = [(r['gd'] - r['baseline']) / r['baseline'] * 100 
                       for r in results if r['gd'] is not None]
    
    print(f"\nFGRF:   {np.mean(fgrf_improvements):+.1f}% ± {np.std(fgrf_improvements):.1f}% (n={len(fgrf_improvements)})")
    if random_improvements:
        print(f"Random: {np.mean(random_improvements):+.1f}% ± {np.std(random_improvements):.1f}% (n={len(random_improvements)})")
    if gd_improvements:
        print(f"GD:     {np.mean(gd_improvements):+.1f}% ± {np.std(gd_improvements):.1f}% (n={len(gd_improvements)})")
    
    # Save results
    output_path = os.path.join(args.output, f"multi_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(args.output, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'model': args.model,
            'timestamp': datetime.now().isoformat(),
            'inputs': len(texts),
            'layers': args.layers,
            'results': results,
            'summary': {
                'fgrf_mean': float(np.mean(fgrf_improvements)) if fgrf_improvements else None,
                'fgrf_std': float(np.std(fgrf_improvements)) if fgrf_improvements else None,
                'random_mean': float(np.mean(random_improvements)) if random_improvements else None,
                'random_std': float(np.std(random_improvements)) if random_improvements else None,
                'gd_mean': float(np.mean(gd_improvements)) if gd_improvements else None,
                'gd_std': float(np.std(gd_improvements)) if gd_improvements else None
            }
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    print("\n" + "=" * 80)
    print("✓ COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
