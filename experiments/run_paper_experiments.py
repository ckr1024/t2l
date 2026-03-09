"""
Master Experiment Runner for Paper
===================================

Uses the official T2I-CompBench dataset (300 prompts per subset) for
all attribute binding evaluations.

Generates ALL results needed for the paper in a single run:

  Table 1 (main):       SDXL vs ToMe (Eucl.) vs ToMe (Hyp.) on BLIP-VQA + ImageReward
  Table 2 (ablation):   Configs A-F + Ours (Euclidean ablation)
  Table 3 (hyp_comp):   Euclidean vs Hyperbolic paired comparison
  Table 4 (curvature):  Curvature c sensitivity analysis
  Table 5 (time):       Inference time overhead
  Table 6 (gpt4o):      GPT-4o object binding scores
  Figure 1 (qual):      Qualitative comparison images
  Figure 2 (attention): Cross-attention map visualization
  Figure 3 (curv_plot): Curvature sensitivity curve

Dataset:
  T2I-CompBench (NeurIPS 2023) - auto-downloaded on first run.
  Source: https://github.com/Karine-Huang/T2I-CompBench

Usage:
    # Full experiment (300 prompts x 3 seeds ≈ 50-80 hours on single GPU)
    python -m experiments.run_paper_experiments

    # Medium run (100 prompts x 2 seeds ≈ 20-30 hours)
    python -m experiments.run_paper_experiments --num_prompts 100 --seeds 42 123

    # Quick test (10 prompts x 1 seed ≈ 2-4 hours)
    python -m experiments.run_paper_experiments --quick

    # Selective experiments
    python -m experiments.run_paper_experiments --experiments main ablation

    # Resume from checkpoint (skips completed experiments)
    python -m experiments.run_paper_experiments --resume
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


EXPERIMENT_ORDER = [
    "main",       # Table 1
    "ablation",   # Table 2
    "hyp_comp",   # Table 3
    "curvature",  # Table 4
    "time",       # Table 5
    "gpt4o",      # Table 6
    "qualitative",  # Figure 1
    "attention",    # Figure 2
]


def load_checkpoint(output_dir: str) -> Dict:
    ckpt_path = os.path.join(output_dir, "checkpoint.json")
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r") as f:
            return json.load(f)
    return {"completed": [], "results": {}, "sub_completed": {}}


def save_checkpoint(output_dir: str, checkpoint: Dict):
    ckpt_path = os.path.join(output_dir, "checkpoint.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(ckpt_path, "w") as f:
        json.dump(checkpoint, f, indent=2, default=str)


def _sub_key(exp_name: str, config_name: str, subset: str) -> str:
    """Generate a unique key for config+subset level checkpointing."""
    return f"{exp_name}::{config_name}::{subset}"


def _is_sub_done(checkpoint: Dict, exp_name: str, config_name: str, subset: str) -> bool:
    key = _sub_key(exp_name, config_name, subset)
    return key in checkpoint.get("sub_completed", {})


def _mark_sub_done(checkpoint: Dict, output_dir: str, exp_name: str,
                   config_name: str, subset: str, result: Dict):
    if "sub_completed" not in checkpoint:
        checkpoint["sub_completed"] = {}
    key = _sub_key(exp_name, config_name, subset)
    checkpoint["sub_completed"][key] = result
    save_checkpoint(output_dir, checkpoint)


def _get_sub_result(checkpoint: Dict, exp_name: str, config_name: str, subset: str) -> Optional[Dict]:
    key = _sub_key(exp_name, config_name, subset)
    return checkpoint.get("sub_completed", {}).get(key)


def log(msg: str):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")


# ============================================================
# Experiment 1: Main Comparison (Table 1)
# ============================================================

def run_experiment_main(model, prompt_parser, seeds, num_prompts, output_dir,
                        data_dir=None, use_ir=True, checkpoint=None):
    """SDXL baseline vs ToMe (Euclidean) vs ToMe (Hyperbolic) on T2I-CompBench."""
    from configs.experiment_config import ConfigA, ConfigOurs, HyperbolicConfigOurs
    from experiments.run_t2i_compbench import generate_images_for_prompts, load_t2i_compbench_prompts
    from experiments.eval_metrics import BLIPVQAEvaluator, ImageRewardEvaluator

    log("=" * 70)
    log("TABLE 1: Main Comparison (SDXL vs ToMe-Eucl. vs ToMe-Hyp.)")
    log(f"  Dataset: T2I-CompBench ({num_prompts} prompts per subset)")
    log("=" * 70)

    configs = {
        "SDXL (baseline)": ConfigA(),
        "ToMe (Eucl.)": ConfigOurs(),
        "ToMe (Hyp.)": HyperbolicConfigOurs(),
    }

    blip_eval = BLIPVQAEvaluator()
    ir_eval = ImageRewardEvaluator() if use_ir else None

    results = {}
    for config_name, config in configs.items():
        config.seeds = seeds
        results[config_name] = {}

        for subset in ["color", "shape", "texture"]:
            cached = _get_sub_result(checkpoint, "main", config_name, subset) if checkpoint else None
            if cached:
                results[config_name][subset] = cached
                log(f"  {config_name} | {subset} | SKIPPED (cached)")
                continue

            prompts = load_t2i_compbench_prompts(subset, num_prompts, data_dir)
            safe_name = config_name.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
            img_dir = os.path.join(output_dir, "main", safe_name, subset, "images")

            log(f"  {config_name} | {subset} | {len(prompts)} prompts x {len(seeds)} seeds")

            start = time.time()
            gen_results = generate_images_for_prompts(
                prompts, config, model, prompt_parser, seeds, img_dir
            )
            gen_time = time.time() - start

            eval_paths, eval_prompts = _build_eval_lists(gen_results)
            blip_res = blip_eval.evaluate_batch(eval_paths, eval_prompts, subset) if eval_paths else {
                "mean_score": 0.0, "std_score": 0.0, "num_samples": 0
            }

            entry = {
                "blip_vqa": blip_res["mean_score"],
                "blip_std": blip_res["std_score"],
                "num_images": len(gen_results),
                "gen_time": gen_time,
            }

            if ir_eval:
                try:
                    ir_prompts = [r["prompt"] for r in gen_results]
                    ir_res = ir_eval.evaluate_batch(eval_paths, ir_prompts)
                    entry["image_reward"] = ir_res["mean_score"]
                    entry["image_reward_std"] = ir_res["std_score"]
                except Exception as e:
                    log(f"  ImageReward failed: {e}")

            results[config_name][subset] = entry
            if checkpoint:
                _mark_sub_done(checkpoint, output_dir, "main", config_name, subset, entry)
            log(f"    BLIP-VQA: {blip_res['mean_score']:.4f} ({len(eval_paths)} images) | Time: {gen_time:.1f}s")

    _print_table(results, "Table 1: Main Comparison")

    from experiments.latex_utils import generate_main_comparison_table
    generate_main_comparison_table(results, os.path.join(output_dir, "latex", "table1_main.tex"))
    log(f"  LaTeX table saved")

    return results


# ============================================================
# Experiment 2: Ablation Study (Table 2)
# ============================================================

def run_experiment_ablation(model, prompt_parser, seeds, num_prompts, output_dir,
                            data_dir=None, checkpoint=None):
    """Standard ablation study (Euclidean only) on T2I-CompBench."""
    from configs.experiment_config import ABLATION_CONFIGS
    from experiments.run_t2i_compbench import generate_images_for_prompts, load_t2i_compbench_prompts
    from experiments.eval_metrics import BLIPVQAEvaluator

    log("=" * 70)
    log("TABLE 2: Ablation Study (Configs A-F + Ours)")
    log(f"  Dataset: T2I-CompBench ({num_prompts} prompts per subset)")
    log("=" * 70)

    config_order = ["A", "B", "C", "D", "E", "F", "Ours"]
    blip_eval = BLIPVQAEvaluator()
    results = {}

    for config_name in config_order:
        config_cls = ABLATION_CONFIGS[config_name]
        config = config_cls()
        config.seeds = seeds
        results[config_name] = {}

        for subset in ["color", "shape", "texture"]:
            cached = _get_sub_result(checkpoint, "ablation", config_name, subset) if checkpoint else None
            if cached:
                results[config_name][subset] = cached
                log(f"  Config {config_name} | {subset} | SKIPPED (cached)")
                continue

            prompts = load_t2i_compbench_prompts(subset, num_prompts, data_dir)
            img_dir = os.path.join(output_dir, "ablation", f"config_{config_name}", subset, "images")

            log(f"  Config {config_name} | {subset}")

            gen_results = generate_images_for_prompts(
                prompts, config, model, prompt_parser, seeds, img_dir
            )

            eval_paths, eval_prompts = _build_eval_lists(gen_results)
            blip_res = blip_eval.evaluate_batch(eval_paths, eval_prompts, subset) if eval_paths else {
                "mean_score": 0.0, "std_score": 0.0, "num_samples": 0
            }
            entry = {
                "blip_vqa": blip_res["mean_score"],
                "blip_std": blip_res["std_score"],
            }
            results[config_name][subset] = entry
            if checkpoint:
                _mark_sub_done(checkpoint, output_dir, "ablation", config_name, subset, entry)
            log(f"    BLIP-VQA: {blip_res['mean_score']:.4f}")

    _print_table(results, "Table 2: Ablation Study")

    from experiments.latex_utils import generate_ablation_table
    generate_ablation_table(results, os.path.join(output_dir, "latex", "table2_ablation.tex"))

    return results


# ============================================================
# Experiment 3: Hyperbolic Comparison (Table 3)
# ============================================================

def run_experiment_hyp_comp(model, prompt_parser, seeds, num_prompts, output_dir,
                            data_dir=None, checkpoint=None):
    """Paired Euclidean vs Hyperbolic comparison on T2I-CompBench."""
    from configs.experiment_config import (
        ConfigB, ConfigC, ConfigF, ConfigOurs,
        HyperbolicConfigB, HyperbolicConfigC, HyperbolicConfigF, HyperbolicConfigOurs,
    )
    from experiments.run_t2i_compbench import generate_images_for_prompts, load_t2i_compbench_prompts
    from experiments.eval_metrics import BLIPVQAEvaluator

    log("=" * 70)
    log("TABLE 3: Euclidean vs Hyperbolic Paired Comparison")
    log(f"  Dataset: T2I-CompBench ({num_prompts} prompts per subset)")
    log("=" * 70)

    config_pairs = [
        ("B", ConfigB, "Hyp-B", HyperbolicConfigB),
        ("C", ConfigC, "Hyp-C", HyperbolicConfigC),
        ("F", ConfigF, "Hyp-F", HyperbolicConfigF),
        ("Ours", ConfigOurs, "Hyp-Ours", HyperbolicConfigOurs),
    ]

    blip_eval = BLIPVQAEvaluator()
    eucl_results = {}
    hyp_results = {}

    for eucl_name, eucl_cls, hyp_name, hyp_cls in config_pairs:
        for name, cls, target_dict in [(eucl_name, eucl_cls, eucl_results), (hyp_name, hyp_cls, hyp_results)]:
            config = cls()
            config.seeds = seeds
            target_dict[name] = {}

            for subset in ["color", "shape", "texture"]:
                cached = _get_sub_result(checkpoint, "hyp_comp", name, subset) if checkpoint else None
                if cached:
                    target_dict[name][subset] = cached
                    log(f"  {name} | {subset} | SKIPPED (cached)")
                    continue

                prompts = load_t2i_compbench_prompts(subset, num_prompts, data_dir)
                safe = name.replace("-", "_")
                img_dir = os.path.join(output_dir, "hyp_comp", safe, subset, "images")

                log(f"  {name} | {subset}")
                gen_results = generate_images_for_prompts(
                    prompts, config, model, prompt_parser, seeds, img_dir
                )

                eval_paths, eval_prompts = _build_eval_lists(gen_results)
                blip_res = blip_eval.evaluate_batch(eval_paths, eval_prompts, subset) if eval_paths else {
                    "mean_score": 0.0, "std_score": 0.0, "num_samples": 0
                }
                entry = {
                    "blip_vqa": blip_res["mean_score"],
                    "blip_std": blip_res["std_score"],
                }
                target_dict[name][subset] = entry
                if checkpoint:
                    _mark_sub_done(checkpoint, output_dir, "hyp_comp", name, subset, entry)
                log(f"    BLIP-VQA: {blip_res['mean_score']:.4f}")

    from experiments.latex_utils import generate_hyperbolic_comparison_table
    generate_hyperbolic_comparison_table(
        eucl_results, hyp_results,
        os.path.join(output_dir, "latex", "table3_hyp_comparison.tex"),
    )

    return {"euclidean": eucl_results, "hyperbolic": hyp_results}


# ============================================================
# Experiment 4: Curvature Sensitivity (Table 4 + Figure 3)
# ============================================================

def run_experiment_curvature(model, prompt_parser, seeds, num_prompts, output_dir,
                             curvatures=None, data_dir=None, checkpoint=None):
    """Curvature sensitivity on T2I-CompBench."""
    from configs.experiment_config import HyperbolicConfigOurs
    from experiments.run_t2i_compbench import generate_images_for_prompts, load_t2i_compbench_prompts
    from experiments.eval_metrics import BLIPVQAEvaluator

    curvatures = curvatures or [0.1, 0.5, 1.0, 2.0, 5.0]

    log("=" * 70)
    log(f"TABLE 4: Curvature Sensitivity (c = {curvatures})")
    log(f"  Dataset: T2I-CompBench ({num_prompts} prompts per subset)")
    log("=" * 70)

    blip_eval = BLIPVQAEvaluator()
    results = {}

    for c in curvatures:
        config = HyperbolicConfigOurs()
        config.hyperbolic_curvature = c
        config.seeds = seeds
        c_name = f"c={c}"
        results[c_name] = {}

        for subset in ["color", "shape", "texture"]:
            cached = _get_sub_result(checkpoint, "curvature", c_name, subset) if checkpoint else None
            if cached:
                results[c_name][subset] = cached
                log(f"  c={c} | {subset} | SKIPPED (cached)")
                continue

            prompts = load_t2i_compbench_prompts(subset, num_prompts, data_dir)
            img_dir = os.path.join(output_dir, "curvature", f"c_{c}", subset, "images")

            log(f"  c={c} | {subset}")
            gen_results = generate_images_for_prompts(
                prompts, config, model, prompt_parser, seeds, img_dir
            )

            eval_paths, eval_prompts = _build_eval_lists(gen_results)
            blip_res = blip_eval.evaluate_batch(eval_paths, eval_prompts, subset) if eval_paths else {
                "mean_score": 0.0, "std_score": 0.0, "num_samples": 0
            }
            entry = {
                "blip_vqa": blip_res["mean_score"],
                "blip_std": blip_res["std_score"],
            }
            results[c_name][subset] = entry
            if checkpoint:
                _mark_sub_done(checkpoint, output_dir, "curvature", c_name, subset, entry)
            log(f"    BLIP-VQA: {blip_res['mean_score']:.4f}")

    _print_table(results, "Table 4: Curvature Sensitivity")

    from experiments.latex_utils import generate_curvature_table, generate_curvature_plot_script
    generate_curvature_table(results, os.path.join(output_dir, "latex", "table4_curvature.tex"))
    generate_curvature_plot_script(results, os.path.join(output_dir, "figures", "curvature_plot.pdf"))

    plot_script = os.path.join(output_dir, "figures", "curvature_plot.py")
    try:
        exec(open(plot_script).read())
    except Exception as e:
        log(f"  Plot generation failed (run manually): {e}")

    return results


# ============================================================
# Experiment 5: Time Complexity (Table 5)
# ============================================================

def run_experiment_time(model, prompt_parser, seeds, output_dir, data_dir=None):
    """Measure inference time overhead."""
    from configs.experiment_config import ConfigA, ConfigOurs, HyperbolicConfigOurs
    from experiments.run_t2i_compbench import generate_images_for_prompts, load_t2i_compbench_prompts

    log("=" * 70)
    log("TABLE 5: Time Complexity Comparison")
    log("=" * 70)

    test_prompts = load_t2i_compbench_prompts("color", 5, data_dir)

    configs = {
        "SDXL (baseline)": ConfigA(),
        "ToMe (Eucl.)": ConfigOurs(),
        "ToMe (Hyp.)": HyperbolicConfigOurs(),
    }

    n_warmup = 1
    n_runs = 3

    results = {}
    for config_name, config in configs.items():
        config.seeds = seeds[:1]
        safe = config_name.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
        img_dir = os.path.join(output_dir, "time", safe)

        log(f"  Warmup: {config_name}")
        generate_images_for_prompts(
            test_prompts[:n_warmup], config, model, prompt_parser,
            seeds[:1], os.path.join(img_dir, "warmup"),
        )

        times = []
        for run_i in range(n_runs):
            start = time.time()
            gen = generate_images_for_prompts(
                test_prompts, config, model, prompt_parser,
                seeds[:1], os.path.join(img_dir, f"run{run_i}"),
            )
            elapsed = time.time() - start
            per_img = elapsed / max(len(gen), 1)
            times.append(per_img)
            log(f"    Run {run_i+1}/{n_runs}: {per_img:.2f}s/image")

        results[config_name] = {
            "per_image": np.mean(times),
            "per_image_std": np.std(times),
            "total_images": len(test_prompts) * n_runs,
        }

    log("\n  Time Complexity Summary:")
    baseline = results.get("SDXL (baseline)", {}).get("per_image", 1)
    for name, data in results.items():
        overhead = (data["per_image"] / baseline - 1) * 100 if baseline > 0 else 0
        log(f"    {name}: {data['per_image']:.2f}s/img (overhead: +{overhead:.1f}%)")

    from experiments.latex_utils import generate_time_table
    generate_time_table(results, os.path.join(output_dir, "latex", "table5_time.tex"))

    return results


# ============================================================
# Experiment 6: GPT-4o Benchmark (Table 6)
# ============================================================

def run_experiment_gpt4o(model, prompt_parser, seeds, num_prompts, output_dir):
    """GPT-4o object binding: generate images (scoring done separately via eval_gpt4o_offline.py)."""
    from configs.experiment_config import ConfigA, ConfigOurs, HyperbolicConfigOurs
    from experiments.eval_gpt4o import get_gpt4o_prompts
    from experiments.run_gpt4o_benchmark import generate_gpt4o_benchmark_images
    from experiments.eval_metrics import save_evaluation_results

    log("=" * 70)
    log("TABLE 6: GPT-4o Object Binding - Image Generation")
    log("=" * 70)
    log("  Images will be generated here. Score them later with:")
    log(f"  python -m experiments.eval_gpt4o_offline --base_dir {output_dir}/gpt4o")

    prompts = get_gpt4o_prompts()[:num_prompts]

    configs = {
        "SDXL_baseline": ConfigA(),
        "ToMe_Eucl": ConfigOurs(),
        "ToMe_Hyp": HyperbolicConfigOurs(),
    }

    results = {}
    for config_name, config in configs.items():
        config.seeds = seeds
        config.use_pose_loss = True
        img_dir = os.path.join(output_dir, "gpt4o", config_name, "images")

        log(f"  Generating: {config_name} ({len(prompts)} prompts x {len(seeds)} seeds)")
        gen_results = generate_gpt4o_benchmark_images(
            prompts, config, model, prompt_parser, seeds, img_dir
        )

        manifest = {
            "results": gen_results,
            "num_images": len(gen_results),
            "prompts_used": len(prompts),
            "seeds": seeds,
        }
        save_evaluation_results(
            manifest,
            os.path.join(output_dir, "gpt4o", config_name, "generation_results.json"),
        )
        log(f"    Generated {len(gen_results)} images")

        results[config_name] = {"num_images": len(gen_results)}

    log(f"\n  All GPT-4o images saved to {output_dir}/gpt4o/")
    log(f"  To score, run on a machine with API access:")
    log(f"    export OPENAI_API_KEY='sk-...'")
    log(f"    python -m experiments.eval_gpt4o_offline --base_dir {output_dir}/gpt4o")

    return results


# ============================================================
# Figure 1: Qualitative Comparison
# ============================================================

def run_experiment_qualitative(model, prompt_parser, output_dir):
    """Generate side-by-side qualitative comparison images."""
    from configs.experiment_config import ConfigA, ConfigOurs, HyperbolicConfigOurs
    from utils.ptp_utils import AttentionStore
    from run_demo import run_on_prompt, filter_text
    from PIL import Image

    log("=" * 70)
    log("FIGURE 1: Qualitative Comparison")
    log("=" * 70)

    qual_prompts = [
        "a cat wearing sunglasses and a dog wearing hat",
        "a red apple and a green pear",
        "a boy with hat and a girl with sunglasses",
        "a white cat and a black dog",
        "a wooden table and a metal chair",
        "a round clock and a square frame",
    ]

    configs = {
        "SDXL": ConfigA(),
        "ToMe_Eucl": ConfigOurs(),
        "ToMe_Hyp": HyperbolicConfigOurs(),
    }

    try:
        import en_core_web_trf
        nlp = en_core_web_trf.load()
    except ImportError:
        import spacy
        nlp = spacy.load("en_core_web_sm")

    fig_dir = os.path.join(output_dir, "figures", "qualitative")
    os.makedirs(fig_dir, exist_ok=True)

    seed = 42

    for prompt_idx, prompt in enumerate(qual_prompts):
        log(f"  Prompt {prompt_idx+1}/{len(qual_prompts)}: {prompt[:50]}")
        row_images = []

        doc = nlp(prompt)
        prompt_parser.set_doc(doc)
        token_indices = prompt_parser._get_indices(prompt)
        prompt_anchor = prompt_parser._split_prompt(doc)
        token_indices, prompt_anchor = filter_text(token_indices, prompt_anchor)

        fallback_standard = not token_indices or not prompt_anchor

        for config_name, config in configs.items():
            config.prompt = prompt
            nouns = [chunk.root.text for chunk in doc.noun_chunks]
            merged = " and ".join([f"a {n}" for n in nouns[:2]]) or prompt
            config.prompt_merged = merged
            config.prompt_length = len(model.tokenizer(prompt)["input_ids"]) - 2

            orig_std = config.run_standard_sd
            if fallback_standard:
                config.run_standard_sd = True

            g = torch.Generator("cuda").manual_seed(seed)
            controller = AttentionStore()

            try:
                use_indices = token_indices if token_indices else [[0], [0]]
                use_anchor = prompt_anchor if prompt_anchor else [prompt]
                image = run_on_prompt(
                    prompt=prompt, model=model, controller=controller,
                    token_indices=use_indices, prompt_anchor=use_anchor,
                    seed=g, config=config,
                )
                image.save(os.path.join(fig_dir, f"p{prompt_idx}_{config_name}.png"))
                row_images.append(image)
            except Exception as e:
                log(f"    Error ({config_name}): {e}")
                row_images.append(Image.new("RGB", (1024, 1024), "gray"))
            finally:
                if fallback_standard:
                    config.run_standard_sd = orig_std

        if row_images:
            w, h = row_images[0].size
            grid = Image.new("RGB", (w * len(row_images), h))
            for i, img in enumerate(row_images):
                grid.paste(img.resize((w, h)), (i * w, 0))
            grid.save(os.path.join(fig_dir, f"comparison_p{prompt_idx}.png"))

    log(f"  Qualitative images saved to {fig_dir}")
    return {"status": "completed", "output_dir": fig_dir}


# ============================================================
# Figure 2: Attention Visualization
# ============================================================

def run_experiment_attention(model, prompt_parser, output_dir):
    """Cross-attention map visualization."""
    log("=" * 70)
    log("FIGURE 2: Cross-Attention Visualization")
    log("=" * 70)

    from experiments.visualize_attention import run_attention_visualization

    attn_dir = os.path.join(output_dir, "figures", "attention")

    for prompt in [
        "a cat wearing sunglasses and a dog wearing hat",
        "a red apple and a green pear",
    ]:
        safe_prompt = prompt[:30].replace(" ", "_")
        run_attention_visualization(
            prompt=prompt,
            config_names=["B", "C", "Ours"],
            model=model,
            prompt_parser=prompt_parser,
            seed=42,
            output_dir=os.path.join(attn_dir, safe_prompt),
        )

    log(f"  Attention visualizations saved to {attn_dir}")
    return {"status": "completed"}


# ============================================================
# Utilities
# ============================================================

def _build_eval_lists(gen_results: List[Dict]):
    """Extract (image_paths, prompts) from generation results for BLIP-VQA."""
    image_paths = [r["image_path"] for r in gen_results]
    prompts = [r["prompt"] for r in gen_results]
    return image_paths, prompts


def _print_table(results: Dict, title: str):
    subsets = ["color", "shape", "texture"]
    log(f"\n{'='*70}")
    log(f"  {title}")
    log(f"{'='*70}")

    header = f"{'Method':<25}"
    for s in subsets:
        header += f" {s.capitalize():>10}"
    header += f" {'Avg':>10}"
    log(header)
    log("-" * 70)

    for name, subset_data in results.items():
        line = f"{name:<25}"
        scores = []
        for s in subsets:
            if s in subset_data:
                v = subset_data[s].get("blip_vqa", 0)
                scores.append(v)
                line += f" {v:>10.4f}"
            else:
                line += f" {'N/A':>10}"
        avg = np.mean(scores) if scores else 0
        line += f" {avg:>10.4f}"
        log(line)
    log("=" * 70)


def estimate_time(num_prompts, num_seeds, experiments, per_image_sec=15):
    """Rough time estimate. Default 15s/image assumes H100/H200-class GPU."""
    n_images_per_exp = {
        "main": num_prompts * num_seeds * 3 * 3,
        "ablation": num_prompts * num_seeds * 7 * 3,
        "hyp_comp": num_prompts * num_seeds * 8 * 3,
        "curvature": num_prompts * num_seeds * 5 * 3,
        "time": 5 * 3 * 3,
        "gpt4o": 50 * num_seeds * 3,
        "qualitative": 6 * 3,
        "attention": 2 * 3,
    }
    total = sum(n_images_per_exp.get(e, 0) for e in experiments)
    hours = total * per_image_sec / 3600
    breakdown = {e: n_images_per_exp.get(e, 0) for e in experiments}
    return total, hours, breakdown


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run all paper experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run (estimated 12-24h on single GPU):
  python -m experiments.run_paper_experiments

  # Quick test run (~1-2h):
  python -m experiments.run_paper_experiments --quick

  # Only specific experiments:
  python -m experiments.run_paper_experiments --experiments main hyp_comp curvature

  # Resume from checkpoint:
  python -m experiments.run_paper_experiments --resume

  # Custom settings:
  python -m experiments.run_paper_experiments --num_prompts 30 --seeds 42 123 456
        """,
    )

    parser.add_argument("--experiments", type=str, nargs="+",
                       default=["all"],
                       choices=EXPERIMENT_ORDER + ["all"],
                       help="Which experiments to run")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test: 10 prompts, 1 seed")
    parser.add_argument("--num_prompts", type=int, default=300,
                       help="Prompts per subset (default 300 = full T2I-CompBench)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                       help="Random seeds for statistical significance")
    parser.add_argument("--curvatures", type=float, nargs="+",
                       default=[0.1, 0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--model_path", type=str,
                       default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--output_dir", type=str,
                       default="./paper_results")
    parser.add_argument("--data_dir", type=str, default=None,
                       help="T2I-CompBench data directory (auto-downloaded if not specified)")
    parser.add_argument("--no_image_reward", action="store_true")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint, skip completed sub-tasks")
    parser.add_argument("--estimate_only", action="store_true",
                       help="Print time estimate and exit without running")
    parser.add_argument("--per_image_sec", type=float, default=15,
                       help="Seconds per image for time estimate (default 15 for H200)")
    args = parser.parse_args()

    if args.quick:
        args.num_prompts = 10
        args.seeds = [42]
        args.curvatures = [0.5, 1.0, 2.0]

    experiments = EXPERIMENT_ORDER if "all" in args.experiments else args.experiments

    total_images, est_hours, breakdown = estimate_time(
        args.num_prompts, len(args.seeds), experiments, args.per_image_sec
    )
    log(f"Experiment Plan:")
    log(f"  Experiments: {experiments}")
    log(f"  Prompts/subset: {args.num_prompts}")
    log(f"  Seeds: {args.seeds}")
    log(f"  Estimated images: {total_images}")
    log(f"  Estimated time: {est_hours:.1f} hours (at {args.per_image_sec}s/image)")
    log(f"  Output: {args.output_dir}")
    log(f"")
    log(f"  Per-experiment breakdown:")
    for exp_name in experiments:
        n = breakdown.get(exp_name, 0)
        h = n * args.per_image_sec / 3600
        log(f"    {exp_name:15s}: {n:>7d} images  ~{h:>5.1f}h")
    log("")

    if args.estimate_only:
        log("Exiting (--estimate_only mode)")
        return

    checkpoint = load_checkpoint(args.output_dir) if args.resume else {"completed": [], "results": {}, "sub_completed": {}}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using device: {device}")

    from configs.experiment_config import BaseExperimentConfig
    base_config = BaseExperimentConfig()
    base_config.model_path = args.model_path

    from run_demo import load_model
    log("Loading model...")
    model, prompt_parser = load_model(base_config, device)
    log("Model loaded.\n")

    total_start = time.time()

    experiment_funcs = {
        "main": lambda: run_experiment_main(
            model, prompt_parser, args.seeds, args.num_prompts,
            args.output_dir, data_dir=args.data_dir,
            use_ir=not args.no_image_reward, checkpoint=checkpoint,
        ),
        "ablation": lambda: run_experiment_ablation(
            model, prompt_parser, args.seeds, args.num_prompts,
            args.output_dir, data_dir=args.data_dir, checkpoint=checkpoint,
        ),
        "hyp_comp": lambda: run_experiment_hyp_comp(
            model, prompt_parser, args.seeds, args.num_prompts,
            args.output_dir, data_dir=args.data_dir, checkpoint=checkpoint,
        ),
        "curvature": lambda: run_experiment_curvature(
            model, prompt_parser, args.seeds, args.num_prompts,
            args.output_dir, args.curvatures,
            data_dir=args.data_dir, checkpoint=checkpoint,
        ),
        "time": lambda: run_experiment_time(
            model, prompt_parser, args.seeds, args.output_dir,
            data_dir=args.data_dir,
        ),
        "gpt4o": lambda: run_experiment_gpt4o(
            model, prompt_parser, args.seeds, min(args.num_prompts, 50), args.output_dir,
        ),
        "qualitative": lambda: run_experiment_qualitative(
            model, prompt_parser, args.output_dir,
        ),
        "attention": lambda: run_experiment_attention(
            model, prompt_parser, args.output_dir,
        ),
    }

    for exp_name in experiments:
        if args.resume and exp_name in checkpoint["completed"]:
            log(f"SKIPPING {exp_name} (already completed)")
            continue

        log(f"\n{'#'*70}")
        log(f"# Starting experiment: {exp_name}")
        log(f"{'#'*70}\n")

        exp_start = time.time()
        try:
            result = experiment_funcs[exp_name]()
            checkpoint["results"][exp_name] = result
            checkpoint["completed"].append(exp_name)
            save_checkpoint(args.output_dir, checkpoint)

            exp_time = time.time() - exp_start
            log(f"\n  {exp_name} completed in {exp_time/60:.1f} minutes")

        except Exception as e:
            log(f"\n  ERROR in {exp_name}: {e}")
            import traceback
            traceback.print_exc()
            checkpoint["results"][exp_name] = {"error": str(e)}
            save_checkpoint(args.output_dir, checkpoint)

    total_time = time.time() - total_start

    summary_path = os.path.join(args.output_dir, "all_results.json")
    checkpoint["total_time_seconds"] = total_time
    with open(summary_path, "w") as f:
        json.dump(checkpoint["results"], f, indent=2, default=str)

    log(f"\n{'='*70}")
    log(f"ALL EXPERIMENTS COMPLETED in {total_time/3600:.1f} hours")
    log(f"{'='*70}")
    log(f"Results: {args.output_dir}/all_results.json")
    log(f"Dataset: T2I-CompBench ({args.num_prompts} prompts per subset)")
    log(f"LaTeX tables: {args.output_dir}/latex/")
    log(f"Figures: {args.output_dir}/figures/")
    log(f"")
    log(f"Output structure:")
    log(f"  {args.output_dir}/")
    log(f"    latex/")
    log(f"      table1_main.tex          <- Table 1: Main comparison")
    log(f"      table2_ablation.tex       <- Table 2: Ablation study")
    log(f"      table3_hyp_comparison.tex <- Table 3: Eucl. vs Hyp.")
    log(f"      table4_curvature.tex      <- Table 4: Curvature sensitivity")
    log(f"      table5_time.tex           <- Table 5: Time complexity")
    log(f"      table6_gpt4o.tex          <- Table 6: GPT-4o scores")
    log(f"    figures/")
    log(f"      qualitative/              <- Figure 1: Side-by-side images")
    log(f"      attention/                <- Figure 2: Attention maps")
    log(f"      curvature_plot.pdf        <- Figure 3: Curvature curve")


if __name__ == "__main__":
    main()
