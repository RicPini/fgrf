# Examples

Minimal examples for running FGRF on small language models.

## Versioning

- `*_v3.py` — current release (Version 3, diagnostic probe framing). Use these.
- `*_v2.py` — Version 2 scripts, retained for reproducing the v2 paper.
- `*.py` (no suffix) — Version 1 scripts, retained for reproducing the v1 paper.

## Available examples (v3)

| Script | Model | Notes |
|---|---|---|
| `run_gpt2_v3.py` | GPT-2-medium | Fused QKV architecture |
| `run_qwen_single_v3.py` | Qwen2.5-0.5B | Default text set |
| `run_qwen_multi_v3.py` | Qwen2.5-0.5B | Larger text batch |
| `run_smollm2_single_v3.py` | SmolLM2-360M | Single-input run |

These are the three architectures reported in the v3 paper.

## Requirements

```bash
pip install torch transformers numpy
