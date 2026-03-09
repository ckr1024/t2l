"""
Ablation Study Runner (Table 2 in paper)

Tests different configurations to validate the contribution of each component:
  Config A: Baseline SDXL (no ToMe, no losses)
  Config B: ToMe + ETS only
  Config C: ToMe + ETS + Lent (entropy loss)
  Config D: Lent + Lsem only (no ToMe)
  Config E: Lent only (no ToMe, no Lsem)
  Config F: ToMe + ETS + Lsem (semantic binding loss, no entropy loss)
  Ours:     ToMe + ETS + Lent + Lsem (full method)

Usage:
    python -m experiments.run_ablation [--configs A B C Ours] [--num_prompts 20]
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.experiment_config import ABLATION_CONFIGS
from experiments.run_t2i_compbench import (
    load_t2i_compbench_prompts,
    generate_images_for_prompts,
    PROMPT_SETS,
)
from experiments.eval_metrics import BLIPVQAEvaluator, save_evaluation_results


ABLATION_DEMO_PROMPTS = [
    ("a cat wearing sunglasses and a dog wearing hat",
     [{"object": "cat", "attribute": "sunglasses"}, {"object": "dog", "attribute": "hat"}]),
    ("a white cat and a black dog",
     [{"object": "cat", "attribute": "white"}, {"object": "dog", "attribute": "black"}]),
    ("a red apple and a green pear",
     [{"object": "apple", "attribute": "red"}, {"object": "pear", "attribute": "green"}]),
    ("a blue car and a red bus",
     [{"object": "car", "attribute": "blue"}, {"object": "bus", "attribute": "red"}]),
    ("a boy wearing hat and a girl wearing sunglasses",
     [{"object": "boy", "attribute": "hat"}, {"object": "girl", "attribute": "sunglasses"}]),
]


def run_single_config(
    config_name: str,
    config,
    model,
    prompt_parser,
    prompts: List[tuple],
    subset: str,
    seeds: List[int],
) -> Dict:
    """Run evaluation for a single ablation configuration."""
    print(f"\n{'='*60}")
    print(f"Running Config {config_name}: {config.__doc__ or ''}")
    print(f"  ToMe: {getattr(config, 'use_token_merge', 'N/A')}")
    print(f"  ETS: {getattr(config, 'use_ets', 'N/A')}")
    print(f"  Lent: {config.tome_control_steps[1] > 0 if not config.run_standard_sd else False}")
    print(f"  Lsem: {config.tome_control_steps[0] > 0 if not config.run_standard_sd else False}")
    print(f"{'='*60}")

    output_dir = os.path.join(str(config.output_path), subset, "images")

    start_time = time.time()
    generation_results = generate_images_for_prompts(
        prompts, config, model, prompt_parser, seeds, output_dir
    )
    gen_time = time.time() - start_time

    blip_evaluator = BLIPVQAEvaluator()

    image_paths = [r["image_path"] for r in generation_results]
    prompts_data = []
    for r in generation_results:
        if r["attr_data"]:
            prompts_data.extend(r["attr_data"])
        else:
            prompts_data.append({
                "object": "object", "attribute": subset, "prompt": r["prompt"]
            })

    if len(prompts_data) > len(image_paths):
        prompts_data = prompts_data[:len(image_paths)]
    elif len(prompts_data) < len(image_paths):
        image_paths = image_paths[:len(prompts_data)]

    blip_results = blip_evaluator.evaluate_batch(image_paths, prompts_data, subset)

    results = {
        "config_name": config_name,
        "subset": subset,
        "num_images": len(generation_results),
        "generation_time_seconds": gen_time,
        "time_per_image": gen_time / max(len(generation_results), 1),
        "blip_vqa": blip_results,
    }

    result_path = os.path.join(str(config.output_path), subset, "results.json")
    save_evaluation_results(results, result_path)

    return results


def run_ablation_study(
    config_names: List[str],
    subsets: List[str],
    model,
    prompt_parser,
    num_prompts: int = 20,
    seeds: List[int] = None,
    data_dir: str = None,
) -> Dict:
    """
    Run complete ablation study across all configs and subsets.

    Returns:
        Nested dict: results[config_name][subset] = evaluation_results
    """
    seeds = seeds or [42]
    all_results = {}

    for config_name in config_names:
        if config_name not in ABLATION_CONFIGS:
            print(f"Warning: Unknown config '{config_name}', skipping.")
            continue

        config_cls = ABLATION_CONFIGS[config_name]
        config = config_cls()
        config.seeds = seeds

        all_results[config_name] = {}

        for subset in subsets:
            prompts = load_t2i_compbench_prompts(subset, num_prompts, data_dir)
            if not prompts:
                prompts = ABLATION_DEMO_PROMPTS

            results = run_single_config(
                config_name=config_name,
                config=config,
                model=model,
                prompt_parser=prompt_parser,
                prompts=prompts,
                subset=subset,
                seeds=seeds,
            )
            all_results[config_name][subset] = results

    return all_results


def print_ablation_table(results: Dict):
    """Print ablation results in a formatted table (similar to Table 2)."""
    print("\n" + "=" * 80)
    print("Ablation Study Results (Table 2)")
    print("=" * 80)

    header = f"{'Config':<8} {'ToMe':<6} {'Lent':<6} {'Lsem':<6}"
    subsets = set()
    for config_results in results.values():
        subsets.update(config_results.keys())
    subsets = sorted(subsets)

    for subset in subsets:
        header += f" {subset.capitalize():>10}"
    print(header)
    print("-" * len(header))

    config_info = {
        "A":    (False, False, False),
        "B":    (True,  False, False),
        "C":    (True,  True,  False),
        "D":    (False, True,  True),
        "E":    (False, True,  False),
        "F":    (True,  False, True),
        "Ours": (True,  True,  True),
    }

    for config_name in ["A", "B", "C", "D", "E", "F", "Ours"]:
        if config_name not in results:
            continue

        tome, lent, lsem = config_info.get(config_name, (None, None, None))
        check = lambda b: "✓" if b else "✗"
        line = f"{config_name:<8} {check(tome):<6} {check(lent):<6} {check(lsem):<6}"

        for subset in subsets:
            if subset in results[config_name]:
                score = results[config_name][subset]["blip_vqa"]["mean_score"]
                line += f" {score:>10.4f}"
            else:
                line += f" {'N/A':>10}"
        print(line)

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Ablation Study")
    parser.add_argument("--configs", type=str, nargs="+",
                       default=["A", "B", "C", "D", "E", "F", "Ours"],
                       help="Which configs to run")
    parser.add_argument("--subsets", type=str, nargs="+",
                       default=["color", "shape", "texture"],
                       choices=["color", "shape", "texture"])
    parser.add_argument("--num_prompts", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--model_path", type=str,
                       default="stabilityai/stable-diffusion-xl-base-1.0")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from configs.experiment_config import BaseExperimentConfig
    base_config = BaseExperimentConfig()
    base_config.model_path = args.model_path

    from run_demo import load_model
    model, prompt_parser = load_model(base_config, device)

    results = run_ablation_study(
        config_names=args.configs,
        subsets=args.subsets,
        model=model,
        prompt_parser=prompt_parser,
        num_prompts=args.num_prompts,
        seeds=args.seeds,
        data_dir=args.data_dir,
    )

    print_ablation_table(results)

    summary_path = "./experiment_results/ablation/summary.json"
    save_evaluation_results(results, summary_path)


if __name__ == "__main__":
    main()
