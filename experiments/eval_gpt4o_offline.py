"""
Offline GPT-4o Evaluation Script
=================================

Scores pre-generated images with GPT-4o API. Designed for two-stage workflow:
  Stage 1 (cloud GPU): Generate images using run_paper_experiments.py
  Stage 2 (local/API): Score images with this script

Usage:
    # Score a single method's images:
    python -m experiments.eval_gpt4o_offline \
        --image_dir paper_results/gpt4o/ToMe_Hyp/images \
        --method_name "ToMe (Hyp.)"

    # Score all methods in a directory:
    python -m experiments.eval_gpt4o_offline \
        --base_dir paper_results/gpt4o \
        --output paper_results/gpt4o_scores.json

    # Use a specific OpenAI model:
    python -m experiments.eval_gpt4o_offline \
        --base_dir paper_results/gpt4o \
        --openai_model gpt-4o-mini

Environment:
    export OPENAI_API_KEY="sk-..."
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.eval_gpt4o import GPT4oEvaluator, get_gpt4o_prompts


def discover_image_prompt_pairs(image_dir: str, prompts: List[str]) -> List[Dict]:
    """Match generated images with their prompts by filename index."""
    pairs = []
    for f in sorted(os.listdir(image_dir)):
        if not f.endswith((".png", ".jpg", ".jpeg")):
            continue

        match = re.match(r"prompt(\d+)_seed(\d+)\.", f)
        if match:
            idx = int(match.group(1))
            seed = int(match.group(2))
            if idx < len(prompts):
                pairs.append({
                    "image_path": os.path.join(image_dir, f),
                    "prompt": prompts[idx],
                    "prompt_idx": idx,
                    "seed": seed,
                })
            else:
                pairs.append({
                    "image_path": os.path.join(image_dir, f),
                    "prompt": f"Unknown prompt (idx={idx})",
                    "prompt_idx": idx,
                    "seed": seed,
                })
    return pairs


def load_generation_manifest(image_dir: str) -> Optional[List[Dict]]:
    """Try to load a generation results JSON if available."""
    for name in ["generation_results.json", "results.json"]:
        manifest_path = os.path.join(os.path.dirname(image_dir), name)
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                data = json.load(f)
            if isinstance(data, dict) and "results" in data:
                return data["results"]
            if isinstance(data, list):
                return data
    return None


def score_method(
    image_dir: str,
    method_name: str,
    prompts: List[str],
    evaluator: GPT4oEvaluator,
    openai_model: str = "gpt-4o",
) -> Dict:
    """Score all images in a directory."""
    manifest = load_generation_manifest(image_dir)
    if manifest:
        pairs = manifest
        print(f"  Loaded {len(pairs)} pairs from manifest")
    else:
        pairs = discover_image_prompt_pairs(image_dir, prompts)
        print(f"  Discovered {len(pairs)} image-prompt pairs")

    if not pairs:
        print(f"  WARNING: No images found in {image_dir}")
        return {"method": method_name, "num_images": 0, "error": "No images found"}

    valid_pairs = [p for p in pairs if os.path.exists(p["image_path"])]
    print(f"  Scoring {len(valid_pairs)} images with {openai_model}...")

    gpt4o_results = evaluator.evaluate_batch(
        [p["image_path"] for p in valid_pairs],
        [p["prompt"] for p in valid_pairs],
        model=openai_model,
    )

    return {
        "method": method_name,
        "num_images": len(valid_pairs),
        "mean_score": gpt4o_results["mean_score"],
        "std_score": gpt4o_results["std_score"],
        "num_valid": gpt4o_results["num_valid"],
        "individual_results": gpt4o_results.get("individual_results", []),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Offline GPT-4o evaluation of pre-generated images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image_dir", type=str, default=None,
                       help="Single image directory to score")
    parser.add_argument("--method_name", type=str, default="Unknown",
                       help="Method name for the single directory")
    parser.add_argument("--base_dir", type=str, default=None,
                       help="Base directory containing subdirs per method")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON path")
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--openai_model", type=str, default="gpt-4o")
    parser.add_argument("--num_prompts", type=int, default=50,
                       help="Number of GPT-4o prompts used during generation")
    args = parser.parse_args()

    api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable or use --openai_api_key")
        sys.exit(1)

    evaluator = GPT4oEvaluator(api_key=api_key)
    prompts = get_gpt4o_prompts()[:args.num_prompts]

    all_results = {}

    if args.image_dir:
        result = score_method(
            args.image_dir, args.method_name, prompts, evaluator, args.openai_model
        )
        all_results[args.method_name] = result

    elif args.base_dir:
        for subdir in sorted(os.listdir(args.base_dir)):
            subdir_path = os.path.join(args.base_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue

            img_dir = os.path.join(subdir_path, "images")
            if not os.path.isdir(img_dir):
                img_dir = subdir_path

            has_images = any(
                f.endswith((".png", ".jpg")) for f in os.listdir(img_dir)
            )
            if not has_images:
                continue

            method_name = subdir.replace("_", " ")
            print(f"\n--- Scoring: {method_name} ---")
            result = score_method(
                img_dir, method_name, prompts, evaluator, args.openai_model
            )
            all_results[method_name] = result
    else:
        print("ERROR: Provide either --image_dir or --base_dir")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("GPT-4o Evaluation Summary")
    print("=" * 60)
    print(f"{'Method':<30} {'Score':>8} {'Std':>8} {'N':>5}")
    print("-" * 55)

    for name, res in all_results.items():
        score = res.get("mean_score", 0)
        std = res.get("std_score", 0)
        n = res.get("num_valid", res.get("num_images", 0))
        print(f"{name:<30} {score:>8.4f} {std:>8.4f} {n:>5}")

    output_path = args.output or os.path.join(
        args.base_dir or os.path.dirname(args.image_dir),
        "gpt4o_offline_scores.json",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
