# GeoBind: Hyperbolic-Euclidean Composite Token Learning for Semantic Binding in Text-to-Image Generation

Official implementation for **GeoBind** (ACM MM 2025).

GeoBind is a training-free method that integrates hyperbolic and Euclidean geometry for composite token learning, improving semantic binding in text-to-image generation across SDXL, SD 3, and FLUX.1.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

## Project Structure

```
├── pipe_geobind.py              # GeoBind pipeline (extends SDXL)
├── run_geobind.py                # T2I-CompBench attribute binding generation
├── run_object_binding_generate.py # GOB-Bench object binding generation (200 prompts)
├── run_object_binding_eval.py    # GOB-Bench evaluation (GPT-4o / Qwen2-VL)
├── eval_image_reward.py          # ImageReward evaluation
├── eval_clip_score.py            # CLIP Score evaluation
├── eval_gromov_hyperbolicity.py  # Gromov δ-hyperbolicity analysis (Table 1)
├── prompt_utils.py               # SpaCy-based prompt parsing
├── utils/
│   ├── hyperbolic_utils.py       # Poincaré ball operations & TokenMerger
│   ├── ptp_utils.py              # Attention store & aggregation
│   ├── gaussian_smoothing.py     # Spatial smoothing
│   └── vis_utils.py              # Attention visualization
├── data/
│   └── t2i_compbench/            # T2I-CompBench prompts (color/shape/texture)
└── supplementary/
    ├── supplementary.tex         # Supplementary material
    └── gob_bench_prompts.txt     # GOB-Bench 200 prompts (4 difficulty levels)
```

## Experiments

### 1. T2I-CompBench Attribute Binding (Table 2)

Generate images on color/shape/texture subsets:
```bash
python run_geobind.py --output_dir eval_results --subsets color shape texture
```

Evaluate with BLIP-VQA using the official [T2I-CompBench](https://github.com/Karine-Huang/T2I-CompBench) evaluation code:

```bash
# Set up a separate environment for BLIP-VQA evaluation
pip install -r requirements-BLIP-VQA

# Clone the official T2I-CompBench repo and run BLIP-VQA
git clone https://github.com/Karine-Huang/T2I-CompBench.git
cd T2I-CompBench/BLIPvqa_eval
python BLIP_vqa.py --out_dir ../../eval_results/GeoBind/color
python BLIP_vqa.py --out_dir ../../eval_results/GeoBind/texture
python BLIP_vqa.py --out_dir ../../eval_results/GeoBind/shape
```

> **Note:** BLIP-VQA requires a different dependency environment (e.g., `transformers==4.30.2`). Use `requirements-BLIP-VQA` instead of `requirements.txt` for this step.

Evaluate with ImageReward:
```bash
python eval_image_reward.py --image_root eval_results
```

Evaluate with CLIP Score:
```bash
python eval_clip_score.py --image_root eval_results
```

### 2. GOB-Bench Object Binding (Table 4)

Generate images for all 200 prompts:
```bash
python run_object_binding_generate.py --methods GeoBind --output_dir eval_results_gob_bench
```

Generate for specific difficulty levels:
```bash
python run_object_binding_generate.py --levels Easy Medium --methods GeoBind ToMe SDXL
```

Evaluate with GPT-4o:
```bash
OPENAI_API_KEY=sk-xxx python run_object_binding_eval.py \
    --output_dir eval_results_gob_bench --evaluator gpt4o --methods GeoBind
```

Evaluate with Qwen2-VL:
```bash
python run_object_binding_eval.py \
    --output_dir eval_results_gob_bench --evaluator qwen2vl --methods GeoBind
```

### 3. Gromov δ-Hyperbolicity Analysis (Table 1)

Verify the intrinsic hyperbolic structure of CLIP token embeddings:
```bash
python eval_gromov_hyperbolicity.py --n_prompts 500
```

### 4. Multi-Seed Evaluation

All experiments use 5 random seeds for statistical significance:
```bash
for seed in 42 123 456 789 1024; do
    python run_geobind.py --seed $seed --output_dir eval_results_seed${seed}
done
```

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `c` | 1.0 | Poincaré ball curvature |
| `γ` | 0.1 | Hyperbolic-Euclidean fusion weight |
| `α` | 1.1 | Noun aggregation weight |
| `β` | 1.2 | Attribute aggregation weight |
| `λ₁` | 1.0 | Semantic binding loss weight |
| `λ₂` | 10⁻⁶ | Hyperbolic contrastive loss weight |
| `τ` | 0.07 | Contrastive temperature |
| Steps | 50 | Denoising steps |
| CFG | 7.5 | Classifier-free guidance scale (SDXL) |
| T_tok | 5 | Token optimization steps |
| T_attn | 5 | Attention refinement steps |

## Method Overview

1. **Dual-Space Embedding**: Token embeddings are processed in both Euclidean and hyperbolic (Poincaré ball) spaces, with positional encoding via Möbius addition.
2. **Attention-Based Token Fusion**: Parameter-free multi-head cross-attention merges noun and attribute tokens in both spaces.
3. **Geometry-Aware Fusion**: Euclidean and hyperbolic composite tokens are fused via logarithmic map projection (Eq. 15).
4. **Iterative Refinement**: Semantic binding loss (L_sem) and hyperbolic contrastive loss (L_hyp) refine composite tokens during early denoising steps (Eq. 18).
5. **Entropy-Guided Attention**: Shannon entropy minimization on cross-attention maps ensures spatially focused generation (Eq. 19-20).

## Citation

```bibtex
@inproceedings{geobind2025,
  title={GeoBind: Hyperbolic-Euclidean Composite Token Learning for Semantic Binding in Text-to-Image Generation},
  author={Anonymous},
  booktitle={ACM Multimedia},
  year={2025}
}
```
