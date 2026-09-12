# Fractal Graph Rewriting Framework (FGRF)

A scale-dependent, single-pass weight transformation for analyzing and probing the structural integrity of LLMs.

**Version:** 3.0.0
**DOI (Original Paper):** [10.6084/m9.figshare.33440932](https://doi.org/10.6084/m9.figshare.33440932)
**Release Notes:** [See GitHub Releases](https://github.com/RicPini/fgrf/releases)

---

## What's New in Version 3

Version 3 transitions FGRF from a *proposed* diagnostic to a *validated* one. Version 2 reframed the framework from an optimizer into a structural probe for weight matrices. Version 3 tests that reframing empirically.

**Key improvements:**

- **De-saturated estimators.** The topological complexity `C`, the interdependence exponent `I`, and the Hausdorff dimension `D` no longer sit at their clip floors. The scale `k` is now derived from each matrix's own singular-value spectrum, and the finite-difference windows used to estimate `C` and `I` adapt to real transitions in the spectral counting function.
- **Trained-vs-untrained experiment.** Across five random seeds and three architectures (Qwen2.5-0.5B, SmolLM2-360M, GPT-2-medium), the probe shows a reproducible signature: key and value projections of trained models have systematically lower `C` and higher `D` than the same matrices in an untrained model of the same architecture.
- **Ablation invariance.** The probe table under `full` is byte-identical to the one under `no_thermo`, and `no_fractal` and `plain_gd` produce identical flat tables. The probe reads the matrix before any update; the update has no effect on the measurement.
- **Thermodynamic gate as a second readout.** The gate is silent on the heavily optimized Qwen2.5-0.5B, fires once on SmolLM2-360M, and fires nine times on GPT-2-medium — on matrices that the geometric probe independently flags as structurally ordered.

For full details, see the [Version 3 paper](paper/fgrf_v3_paper.pdf).

---

## Quick Start

Install the required dependencies:

    pip install -r requirements.txt

Run the canonical v3 entry point on a small model:

    python3 fgrf_multi_model_v3.py --model Qwen/Qwen2.5-0.5B --ablation full --layers 20 --seed 42 --dtype float32

To reproduce the ablations reported in the paper, run:

    python3 fgrf_multi_model_v3.py --model Qwen/Qwen2.5-0.5B --ablation no_thermo  --layers 20 --seed 42 --dtype float32
    python3 fgrf_multi_model_v3.py --model Qwen/Qwen2.5-0.5B --ablation no_fractal --layers 20 --seed 42 --dtype float32
    python3 fgrf_multi_model_v3.py --model Qwen/Qwen2.5-0.5B --ablation plain_gd   --layers 20 --seed 42 --dtype float32
    python3 fgrf_multi_model_v3.py --model Qwen/Qwen2.5-0.5B --ablation random     --layers 20 --seed 42 --dtype float32

Results are written to `./fgrf_results/` by default.

---

## Canonical Entry Point

The canonical entry point for Version 3 is:

    fgrf_multi_model_v3.py

The Version 2 scripts `fgrf_multi_model_v2.py` and `fgrf_multi_input_v2.py` are retained in the repository for backward compatibility, but they use the Version 2 estimators and their outputs are not the ones reported in the v3 paper. The Version 1 scripts `fgrf_multi_model.py` and `fgrf_multi_input.py` are retained for the same reason.

---

## Results Directory

The `results/` folder contains the raw JSON outputs for each version of the framework.

    results/
    ├── V1/
    ├── V2/
    └── V3/
        ├── fgrf_v3_gpt2-medium/
        ├── fgrf_v3_Qwen2.5-0.5B/
        └── fgrf_v3_SmolLM2-360M/

Each V3 model folder contains five JSON files, one for each ablation condition (`full`, `no_thermo`, `no_fractal`, `plain_gd`, `random`). The filenames are listed in the Appendix of the Version 3 paper.

**Note on JSON files.** The `multi_results_*.json` files are written by Python's `json.dump`, which emits bare `NaN` tokens for non-finite floats (typically in the `thermo_required_H` field when the Landauer bound is undefined for a given layer). These are valid for Python's `json.load` and for the pipeline that produces them, but they are not valid strict JSON and will be rejected by JavaScript's `JSON.parse`. If you need to load these files with a strict JSON parser, preprocess them to replace `NaN` with `null`, for example:

    sed -i 's/NaN/null/g' multi_results_*.json

This does not affect reproducibility from the Python toolchain.

---

## Examples

The `examples/` folder contains minimal scripts for each version of the framework, including the three architectures reported in the v3 paper. See [`examples/README.md`](examples/README.md) for details.

---

## Papers

| Version | Paper |
|---------|-------|
| v1 | [`paper/fgrf_v1_paper.pdf`](paper/fgrf_v1_paper.pdf) |
| v2 | [`paper/fgrf_v2_paper.pdf`](paper/fgrf_v2_paper.pdf) |
| v3 | [`paper/fgrf_v3_paper.pdf`](paper/fgrf_v3_paper.pdf) |

---

## License

This repository contains two types of content with different licenses:

- **Code** (`.py` files): MIT License — free to use, modify, and distribute.
- **Papers** (`fgrf_v1_paper.pdf`, `fgrf_v2_paper.pdf`, `fgrf_v3_paper.pdf`): Creative Commons Attribution-NonCommercial-NoDerivatives (CC BY-NC-ND) — free to share with attribution, but not for commercial use or modification.

The `LICENSE` file in this repository applies to the code only. The paper's license is stated in the paper footer and on Figshare metadata.

---

## Citation

If you use FGRF in your research, please cite the original paper:

    @misc{fgrf2026,
      title        = {Fractal Graph Rewriting Framework (FGRF)},
      author       = {Pini, Riccardo},
      year         = {2026},
      doi          = {10.6084/m9.figshare.33440932},
      howpublished = {Figshare}
    }

Adjust the BibTeX entry to match the actual publication metadata.

---

## Contributing

Contributions are welcome. Please open an issue or submit a pull request on GitHub.

## Contact

For questions or feedback, please open an issue in this repository.
