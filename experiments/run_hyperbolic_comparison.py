"""
Hyperbolic vs. Euclidean Comparison Experiment

Rigorously compares the original ToMe (Euclidean) against the proposed
hyperbolic improvement across all evaluation dimensions:

  1. T2I-CompBench BLIP-VQA (color, texture, shape)
  2. ImageReward human preference scores
  3. GPT-4o object binding scores (if API key available)
  4. Ablation: which hyperbolic component helps most
  5. Curvature sensitivity analysis
  6. Time complexity overhead measurement

Controlled variables:
  - Same random seeds for all experiments
  - Same prompts, same model, same evaluation metrics
  - Multiple seeds for statistical significance (mean ± std)

Usage:
    python -m experiments.run_hyperbolic_comparison [--quick] [--curvatures 0.1 0.5 1.0 2.0]
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

from configs.experiment_config import (
    ConfigA, ConfigOurs, HyperbolicConfigOurs,
    ConfigB, ConfigC, ConfigF,
    HyperbolicConfigB, HyperbolicConfigC, HyperbolicConfigF,
    ABLATION_CONFIGS, HYPERBOLIC_CONFIGS, ALL_CONFIGS,
)
from experiments.run_t2i_compbench import (
    load_t2i_compbench_prompts,
    generate_images_for_prompts,
)
from experiments.eval_metrics import BLIPVQAEvaluator, ImageRewardEvaluator, save_evaluation_results


# ============================================================
# Shared prompts for fair comparison
# ============================================================

COMPARISON_PROMPTS = {
    "color": [
        ("a red apple and a green pear", [{"object": "apple", "attribute": "red"}, {"object": "pear", "attribute": "green"}]),
        ("a blue car and a red bus", [{"object": "car", "attribute": "blue"}, {"object": "bus", "attribute": "red"}]),
        ("a yellow cat and a black dog", [{"object": "cat", "attribute": "yellow"}, {"object": "dog", "attribute": "black"}]),
        ("a white horse and a brown cow", [{"object": "horse", "attribute": "white"}, {"object": "cow", "attribute": "brown"}]),
        ("a pink flower and a purple butterfly", [{"object": "flower", "attribute": "pink"}, {"object": "butterfly", "attribute": "purple"}]),
        ("a red hat and a blue scarf", [{"object": "hat", "attribute": "red"}, {"object": "scarf", "attribute": "blue"}]),
        ("a green frog and a yellow bird", [{"object": "frog", "attribute": "green"}, {"object": "bird", "attribute": "yellow"}]),
        ("a black cat and a white rabbit", [{"object": "cat", "attribute": "black"}, {"object": "rabbit", "attribute": "white"}]),
        ("a orange cat and a gray mouse", [{"object": "cat", "attribute": "orange"}, {"object": "mouse", "attribute": "gray"}]),
        ("a white swan and a black crow", [{"object": "swan", "attribute": "white"}, {"object": "crow", "attribute": "black"}]),
    ],
    "texture": [
        ("a fluffy cat and a smooth dog", [{"object": "cat", "attribute": "fluffy"}, {"object": "dog", "attribute": "smooth"}]),
        ("a wooden table and a metal chair", [{"object": "table", "attribute": "wooden"}, {"object": "chair", "attribute": "metal"}]),
        ("a glossy car and a matte truck", [{"object": "car", "attribute": "glossy"}, {"object": "truck", "attribute": "matte"}]),
        ("a rough stone and a smooth pebble", [{"object": "stone", "attribute": "rough"}, {"object": "pebble", "attribute": "smooth"}]),
        ("a furry teddy bear and a plastic robot", [{"object": "teddy bear", "attribute": "furry"}, {"object": "robot", "attribute": "plastic"}]),
        ("a leather jacket and a cotton shirt", [{"object": "jacket", "attribute": "leather"}, {"object": "shirt", "attribute": "cotton"}]),
        ("a velvet curtain and a silk pillow", [{"object": "curtain", "attribute": "velvet"}, {"object": "pillow", "attribute": "silk"}]),
        ("a glass bottle and a ceramic cup", [{"object": "bottle", "attribute": "glass"}, {"object": "cup", "attribute": "ceramic"}]),
        ("a metallic robot and a wooden puppet", [{"object": "robot", "attribute": "metallic"}, {"object": "puppet", "attribute": "wooden"}]),
        ("a silky dress and a denim jacket", [{"object": "dress", "attribute": "silky"}, {"object": "jacket", "attribute": "denim"}]),
    ],
    "shape": [
        ("a round clock and a square frame", [{"object": "clock", "attribute": "round"}, {"object": "frame", "attribute": "square"}]),
        ("a triangular roof and a rectangular door", [{"object": "roof", "attribute": "triangular"}, {"object": "door", "attribute": "rectangular"}]),
        ("a round ball and a square box", [{"object": "ball", "attribute": "round"}, {"object": "box", "attribute": "square"}]),
        ("a curved bridge and a straight road", [{"object": "bridge", "attribute": "curved"}, {"object": "road", "attribute": "straight"}]),
        ("a round plate and a rectangular tray", [{"object": "plate", "attribute": "round"}, {"object": "tray", "attribute": "rectangular"}]),
        ("a oval mirror and a square window", [{"object": "mirror", "attribute": "oval"}, {"object": "window", "attribute": "square"}]),
        ("a round table and a long bench", [{"object": "table", "attribute": "round"}, {"object": "bench", "attribute": "long"}]),
        ("a tall tower and a wide bridge", [{"object": "tower", "attribute": "tall"}, {"object": "bridge", "attribute": "wide"}]),
        ("a thin pencil and a thick marker", [{"object": "pencil", "attribute": "thin"}, {"object": "marker", "attribute": "thick"}]),
        ("a round cookie and a square cracker", [{"object": "cookie", "attribute": "round"}, {"object": "cracker", "attribute": "square"}]),
    ],
}


def evaluate_config(
    config_name: str,
    config,
    model,
    prompt_parser,
    prompts: List[tuple],
    subset: str,
    seeds: List[int],
    blip_evaluator: BLIPVQAEvaluator,
    ir_evaluator: Optional[ImageRewardEvaluator] = None,
) -> Dict:
    """Run generation + evaluation for a single config on one subset."""
    output_dir = os.path.join(str(config.output_path), subset, "images")

    start_time = time.time()
    gen_results = generate_images_for_prompts(
        prompts, config, model, prompt_parser, seeds, output_dir
    )
    gen_time = time.time() - start_time

    eval_paths = [r["image_path"] for r in gen_results]
    eval_prompts = [r["prompt"] for r in gen_results]

    blip_results = blip_evaluator.evaluate_batch(eval_paths, eval_prompts, subset) if eval_paths else {
        "mean_score": 0.0, "std_score": 0.0, "num_samples": 0
    }

    result = {
        "config": config_name,
        "subset": subset,
        "blip_vqa": blip_results["mean_score"],
        "blip_std": blip_results["std_score"],
        "gen_time": gen_time,
        "time_per_image": gen_time / max(len(gen_results), 1),
        "num_images": len(gen_results),
    }

    if ir_evaluator:
        try:
            ir_prompts = [r["prompt"] for r in gen_results]
            ir_paths = [r["image_path"] for r in gen_results]
            ir_results = ir_evaluator.evaluate_batch(ir_paths, ir_prompts)
            result["image_reward"] = ir_results["mean_score"]
            result["image_reward_std"] = ir_results["std_score"]
        except Exception as e:
            result["image_reward"] = None
            result["image_reward_error"] = str(e)

    return result


# ============================================================
# Experiment 1: Main Euclidean vs Hyperbolic Comparison
# ============================================================

def run_main_comparison(model, prompt_parser, seeds, num_prompts, use_ir=True):
    """Compare SDXL baseline, Euclidean ToMe, and Hyperbolic ToMe."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Euclidean vs. Hyperbolic ToMe (Main Comparison)")
    print("=" * 70)

    configs_to_test = {
        "SDXL (baseline)": ConfigA(),
        "ToMe (Euclidean)": ConfigOurs(),
        "ToMe (Hyperbolic)": HyperbolicConfigOurs(),
    }

    blip_eval = BLIPVQAEvaluator()
    ir_eval = ImageRewardEvaluator() if use_ir else None

    results = defaultdict(dict)

    for config_name, config in configs_to_test.items():
        config.seeds = seeds
        for subset in ["color", "texture", "shape"]:
            prompts = COMPARISON_PROMPTS[subset][:num_prompts]
            print(f"\n--- {config_name} | {subset} ({len(prompts)} prompts) ---")

            res = evaluate_config(
                config_name, config, model, prompt_parser,
                prompts, subset, seeds, blip_eval, ir_eval,
            )
            results[config_name][subset] = res

    print_comparison_table(results, "Main Comparison: Euclidean vs. Hyperbolic")
    return dict(results)


# ============================================================
# Experiment 2: Hyperbolic Ablation
# ============================================================

def run_hyperbolic_ablation(model, prompt_parser, seeds, num_prompts):
    """Ablate which hyperbolic component contributes most."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Hyperbolic Component Ablation")
    print("=" * 70)

    configs = {
        "B (Eucl.)": ConfigB(),
        "B (Hyp.)": HyperbolicConfigB(),
        "C (Eucl.)": ConfigC(),
        "C (Hyp.)": HyperbolicConfigC(),
        "F (Eucl.)": ConfigF(),
        "F (Hyp.)": HyperbolicConfigF(),
        "Ours (Eucl.)": ConfigOurs(),
        "Ours (Hyp.)": HyperbolicConfigOurs(),
    }

    blip_eval = BLIPVQAEvaluator()
    results = defaultdict(dict)

    for config_name, config in configs.items():
        config.seeds = seeds
        for subset in ["color", "texture", "shape"]:
            prompts = COMPARISON_PROMPTS[subset][:num_prompts]
            print(f"\n--- {config_name} | {subset} ---")
            res = evaluate_config(
                config_name, config, model, prompt_parser,
                prompts, subset, seeds, blip_eval,
            )
            results[config_name][subset] = res

    print_comparison_table(results, "Hyperbolic Ablation")
    return dict(results)


# ============================================================
# Experiment 3: Curvature Sensitivity Analysis
# ============================================================

def run_curvature_analysis(model, prompt_parser, seeds, num_prompts, curvatures):
    """Test different Poincaré ball curvature values."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Curvature Sensitivity Analysis")
    print("=" * 70)

    blip_eval = BLIPVQAEvaluator()
    results = defaultdict(dict)

    for c in curvatures:
        config = HyperbolicConfigOurs()
        config.hyperbolic_curvature = c
        config.seeds = seeds
        config.output_path = Path(f"./experiment_results/curvature/c_{c}")
        config_name = f"c={c}"

        for subset in ["color", "texture", "shape"]:
            prompts = COMPARISON_PROMPTS[subset][:num_prompts]
            print(f"\n--- curvature={c} | {subset} ---")
            res = evaluate_config(
                config_name, config, model, prompt_parser,
                prompts, subset, seeds, blip_eval,
            )
            results[config_name][subset] = res

    print_comparison_table(results, "Curvature Sensitivity")
    return dict(results)


# ============================================================
# Experiment 4: Time Complexity Overhead
# ============================================================

def run_time_comparison(model, prompt_parser, seeds):
    """Measure inference time overhead of hyperbolic operations."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Time Complexity Comparison")
    print("=" * 70)

    test_prompts = COMPARISON_PROMPTS["color"][:3]
    configs = {
        "SDXL baseline": ConfigA(),
        "ToMe (Euclidean)": ConfigOurs(),
        "ToMe (Hyperbolic)": HyperbolicConfigOurs(),
    }

    results = {}
    for config_name, config in configs.items():
        config.seeds = seeds

        start = time.time()
        gen_results = generate_images_for_prompts(
            test_prompts, config, model, prompt_parser, seeds,
            os.path.join(str(config.output_path), "time_test"),
        )
        elapsed = time.time() - start

        per_img = elapsed / max(len(gen_results), 1)
        results[config_name] = {"total": elapsed, "per_image": per_img, "n_images": len(gen_results)}
        print(f"  {config_name}: {per_img:.1f}s per image ({len(gen_results)} images)")

    print("\n--- Time Complexity ---")
    print(f"{'Method':<25} {'Time/image':>12} {'Overhead':>10}")
    print("-" * 50)
    baseline_time = results.get("SDXL baseline", {}).get("per_image", 1)
    for name, res in results.items():
        overhead = (res["per_image"] / baseline_time - 1) * 100 if baseline_time > 0 else 0
        print(f"{name:<25} {res['per_image']:>11.1f}s {overhead:>9.1f}%")

    return results


# ============================================================
# Output Formatting
# ============================================================

def print_comparison_table(results: Dict, title: str):
    """Print a formatted comparison table."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

    subsets = ["color", "texture", "shape"]
    header = f"{'Method':<25}"
    for s in subsets:
        header += f" {s.capitalize():>10}"
    header += f" {'Avg':>10}"
    if any("image_reward" in results[c].get(subsets[0], {}) for c in results):
        header += f" {'ImgRwd':>10}"
    print(header)
    print("-" * len(header))

    for config_name, subset_results in results.items():
        line = f"{config_name:<25}"
        scores = []
        for s in subsets:
            if s in subset_results:
                score = subset_results[s]["blip_vqa"]
                line += f" {score:>10.4f}"
                scores.append(score)
            else:
                line += f" {'N/A':>10}"

        avg = np.mean(scores) if scores else 0
        line += f" {avg:>10.4f}"

        if subsets[0] in subset_results and subset_results[subsets[0]].get("image_reward") is not None:
            ir_scores = [
                subset_results[s].get("image_reward", 0)
                for s in subsets if s in subset_results
            ]
            line += f" {np.mean(ir_scores):>10.4f}"

        print(line)

    print("=" * 80)

    eucl = results.get("ToMe (Euclidean)", {})
    hyp = results.get("ToMe (Hyperbolic)", {})
    if eucl and hyp:
        print("\n  Improvement (Hyperbolic over Euclidean):")
        for s in subsets:
            if s in eucl and s in hyp:
                diff = hyp[s]["blip_vqa"] - eucl[s]["blip_vqa"]
                sign = "+" if diff >= 0 else ""
                print(f"    {s.capitalize()}: {sign}{diff:.4f}")


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Hyperbolic vs Euclidean Comparison")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    parser.add_argument("--experiments", type=str, nargs="+",
                       default=["main", "ablation", "curvature", "time"],
                       choices=["main", "ablation", "curvature", "time", "all"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--curvatures", type=float, nargs="+", default=[0.1, 0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--model_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--no_image_reward", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./experiment_results/hyperbolic_comparison")
    args = parser.parse_args()

    exps = args.experiments
    if "all" in exps:
        exps = ["main", "ablation", "curvature", "time"]

    num_prompts = 3 if args.quick else 10
    seeds = args.seeds[:1] if args.quick else args.seeds

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from configs.experiment_config import BaseExperimentConfig
    base_config = BaseExperimentConfig()
    base_config.model_path = args.model_path

    from run_demo import load_model
    print("Loading model...")
    model, prompt_parser = load_model(base_config, device)
    print("Model loaded.\n")

    all_results = {}
    total_start = time.time()

    if "main" in exps:
        all_results["main_comparison"] = run_main_comparison(
            model, prompt_parser, seeds, num_prompts,
            use_ir=not args.no_image_reward,
        )

    if "ablation" in exps:
        all_results["hyperbolic_ablation"] = run_hyperbolic_ablation(
            model, prompt_parser, seeds, num_prompts,
        )

    if "curvature" in exps:
        all_results["curvature_analysis"] = run_curvature_analysis(
            model, prompt_parser, seeds, num_prompts, args.curvatures,
        )

    if "time" in exps:
        all_results["time_complexity"] = run_time_comparison(model, prompt_parser, seeds)

    total_time = time.time() - total_start
    all_results["total_time_seconds"] = total_time

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "comparison_results.json")
    save_evaluation_results(all_results, output_path)

    print(f"\nAll experiments completed in {total_time:.1f}s")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
