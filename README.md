# Fractal Graph Rewriting Framework (FGRF)
A scale-dependent, single-pass weight transformation for optimizing LLMs without retraining.
**DOI: [10.6084/m9.figshare.33440932](https://doi.org/10.6084/m9.figshare.33440932)**

## License
This repository contains two types of content with different licenses:

- **Code** (`.py` files): MIT License — free to use, modify, and distribute.
- **Paper** (`fgrf_paperV4.pdf`): Creative Commons Attribution-NonCommercial-NoDerivatives (CC BY-NC-ND) — free to share with attribution, but not for commercial use or modification.

The LICENSE file in this repository applies to the code only. The paper's license is stated in the paper footer and on Figshare metadata.

## Quick Start
```bash
pip install transformers accelerate bitsandbytes datasets pandas scikit-learn psutil

python3 fgrf_multi_model.py --model Qwen/Qwen2.5-0.5B --layers 20
# Expected: 99.08 → 12.57 (−87.3%)

python3 fgrf_multi_input.py --model Qwen/Qwen2.5-0.5B --inputs 10 --layers 25 --baseline none
# Expected: −64.4% ± 17.9%
```

## Results
| Model | Baseline PPL | Final PPL | Improvement |
|-------|--------------|-----------|-------------|
| Qwen2.5-0.5B | 99.08 | 12.57 | **−87.3%** |
| TinyLlama-1.1B | 64.72 | 28.07 | **−56.6%** |
| GPT-2 | 141.65 | 132.99 | **−6.1%** |

Runs on **CPU-only**, **8-14GB RAM**.
