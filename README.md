# Fractal Graph Rewriting Framework (FGRF)

A scale-dependent, single-pass weight transformation for analyzing and probing the structural integrity of LLMs.

**Version:** 2.0.0  
**DOI (Original Paper):** [10.6084/m9.figshare.33440932](https://doi.org/10.6084/m9.figshare.33440932)  
**Release Notes:** [See GitHub Releases](https://github.com/RicPini/fgrf/releases)

## License

This repository contains two types of content with different licenses:

- **Code** (`.py` files): MIT License — free to use, modify, and distribute.
- **Paper** (`fgrf_paperV4.pdf` and `fgrf_paper_v2.pdf`): Creative Commons Attribution-NonCommercial-NoDerivatives (CC BY-NC-ND) — free to share with attribution, but not for commercial use or modification.

The LICENSE file in this repository applies to the code only. The paper's license is stated in the paper footer and on Figshare metadata.

## What's New in Version 2

Version 2 represents a major evolution from the original proof-of-concept. It reframes FGRF from a direct optimizer into a **structural integrity probe** for neural networks.

**Key Improvements:**
- **Numerical Stability:** A hybrid ODE integrator (explicit with adaptive damping + implicit midpoint fallback) ensures the fractal flow remains stable.
- **High-Fidelity Math:** Finite-difference topological complexity and a participation-ratio Hausdorff dimension estimator align closely with the theoretical equations.
- **Rigorous Evaluation:** Held-out evaluation is now standard; the framework includes a full ablation suite (`full`, `plain_gd`, `no_thermo`, `no_fractal`, `random`).
- **Diagnostic Utility:** Near-zero updates on highly optimized models (e.g., Qwen2.5-0.5B) indicate that the model is close to its theoretical ideal, making FGRF a new tool for model analysis and surgical tuning.

For full details, see the [new paper](paper/fgrf_paper_v2.pdf).

## Quick Start

Install the required dependencies:

```bash
pip install -r requirements.txt

