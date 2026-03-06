"""
T2I-CompBench Evaluation Script

Evaluates ToMe on the official T2I-CompBench benchmark for attribute binding:
  - Color binding (300 prompts)
  - Shape binding (300 prompts)
  - Texture binding (300 prompts)

Dataset: Karine-Huang/T2I-CompBench (NeurIPS 2023)
         https://github.com/Karine-Huang/T2I-CompBench

Metrics: BLIP-VQA score, ImageReward score

Usage:
    python -m experiments.run_t2i_compbench [--subset color] [--num_prompts 300] [--seeds 42]
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.experiment_config import T2ICompBenchConfig
from experiments.eval_metrics import BLIPVQAEvaluator, ImageRewardEvaluator, save_evaluation_results
from experiments.parse_compbench import parse_compbench_prompt


DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "t2i_compbench",
)


def _ensure_dataset(data_dir: str = DEFAULT_DATA_DIR) -> str:
    """Auto-download T2I-CompBench dataset if not present."""
    required = [f"{s}_val.txt" for s in ("color", "shape", "texture")]
    all_exist = all(
        os.path.exists(os.path.join(data_dir, f)) for f in required
    )
    if all_exist:
        return data_dir

    print(f"T2I-CompBench data not found at {data_dir}, downloading...")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.download_t2i_compbench import download_dataset
    return download_dataset(data_dir)


def load_t2i_compbench_prompts(
    subset: str,
    num_prompts: int = 300,
    data_dir: Optional[str] = None,
) -> List[tuple]:
    """
    Load official T2I-CompBench validation prompts with parsed attribute annotations.

    Loads from local files (auto-downloaded from GitHub if needed).
    Each prompt is automatically parsed to extract (object, attribute) pairs
    for BLIP-VQA evaluation.

    Args:
        subset: "color", "shape", or "texture"
        num_prompts: Max number of prompts to load (default 300 = full set)
        data_dir: Override directory for dataset files

    Returns:
        List of (prompt_text, attr_list) tuples where attr_list contains
        dicts with "object" and "attribute" keys.
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    data_dir = _ensure_dataset(data_dir)

    filepath = os.path.join(data_dir, f"{subset}_val.txt")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"T2I-CompBench {subset} data not found at {filepath}. "
            f"Run: python data/download_t2i_compbench.py"
        )

    with open(filepath, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    prompts = []
    for line in lines[:num_prompts]:
        attrs = parse_compbench_prompt(line, subset)
        prompts.append((line, attrs))

    n_parsed = sum(1 for _, a in prompts if a)
    print(
        f"Loaded {len(prompts)} T2I-CompBench {subset} prompts "
        f"({n_parsed}/{len(prompts)} with parsed attributes)"
    )
    return prompts


def generate_images_for_prompts(
    prompts: List[tuple],
    config,
    model,
    prompt_parser,
    seeds: List[int],
    output_dir: str,
) -> List[Dict]:
    """
    Generate images for a list of prompts using ToMe pipeline.

    Returns:
        List of dicts with image_path, prompt, and metadata
    """
    from utils.ptp_utils import AttentionStore
    from prompt_utils import PromptParser
    from run_demo import run_on_prompt, filter_text

    os.makedirs(output_dir, exist_ok=True)

    results = []

    try:
        import en_core_web_trf
        nlp = en_core_web_trf.load()
    except ImportError:
        import spacy
        nlp = spacy.load("en_core_web_sm")

    for prompt_idx, (prompt, attr_data) in enumerate(prompts):
        config.prompt = prompt

        doc = nlp(prompt)
        prompt_parser.set_doc(doc)
        token_indices = prompt_parser._get_indices(prompt)
        prompt_anchor = prompt_parser._split_prompt(doc)
        token_indices, prompt_anchor = filter_text(token_indices, prompt_anchor)

        if not token_indices:
            token_indices = config.token_indices if config.token_indices else []
        if not prompt_anchor:
            prompt_anchor = config.prompt_anchor if config.prompt_anchor else []

        fallback_standard = not token_indices or not prompt_anchor

        nouns = [chunk.text for chunk in doc.noun_chunks]
        merged_prompt = " and ".join(
            [n.split()[-1] if len(n.split()) > 1 else n for n in nouns]
        )
        if not merged_prompt:
            merged_prompt = prompt

        config.prompt_length = len(
            model.tokenizer(prompt)["input_ids"]
        ) - 2
        config.prompt_merged = merged_prompt

        orig_run_standard = config.run_standard_sd
        if fallback_standard:
            config.run_standard_sd = True

        for seed in seeds:
            g = torch.Generator("cuda").manual_seed(seed)
            controller = AttentionStore()

            try:
                image = run_on_prompt(
                    prompt=prompt,
                    model=model,
                    controller=controller,
                    token_indices=token_indices if token_indices else [[0], [0]],
                    prompt_anchor=prompt_anchor if prompt_anchor else [prompt],
                    seed=g,
                    config=config,
                )

                img_filename = f"prompt{prompt_idx:04d}_seed{seed}.png"
                img_path = os.path.join(output_dir, img_filename)
                image.save(img_path)

                results.append({
                    "image_path": img_path,
                    "prompt": prompt,
                    "seed": seed,
                    "prompt_idx": prompt_idx,
                    "attr_data": attr_data,
                })
                print(f"  [{prompt_idx+1}/{len(prompts)}] seed={seed} saved: {img_filename}")

            except Exception as e:
                print(f"  Error generating prompt {prompt_idx}: {e}")
                continue

        if fallback_standard:
            config.run_standard_sd = orig_run_standard

    return results


def evaluate_subset(
    subset: str,
    config,
    model,
    prompt_parser,
    num_prompts: int = 300,
    seeds: List[int] = None,
    data_dir: Optional[str] = None,
    use_image_reward: bool = True,
) -> Dict:
    """
    Run complete evaluation for a T2I-CompBench subset.

    Args:
        subset: "color", "shape", or "texture"
        config: Experiment configuration
        model: Loaded ToMe pipeline
        prompt_parser: PromptParser instance
        num_prompts: Number of prompts to evaluate
        seeds: Random seeds for generation
        data_dir: Optional directory with T2I-CompBench data
        use_image_reward: Whether to also compute ImageReward scores

    Returns:
        Dict with evaluation results
    """
    seeds = seeds or [42]

    print(f"\n{'='*60}")
    print(f"Evaluating T2I-CompBench: {subset}")
    print(f"{'='*60}")

    prompts = load_t2i_compbench_prompts(subset, num_prompts, data_dir)
    print(f"Loaded {len(prompts)} prompts for {subset}")

    output_dir = os.path.join(str(config.output_path), subset, "images")

    print("\n--- Generating images ---")
    start_time = time.time()
    generation_results = generate_images_for_prompts(
        prompts, config, model, prompt_parser, seeds, output_dir
    )
    gen_time = time.time() - start_time
    print(f"Generation time: {gen_time:.1f}s ({gen_time/len(generation_results):.1f}s per image)")

    print("\n--- Computing BLIP-VQA scores (T2I-CompBench protocol) ---")
    blip_evaluator = BLIPVQAEvaluator()

    eval_image_paths = [r["image_path"] for r in generation_results]
    eval_prompts = [r["prompt"] for r in generation_results]

    blip_results = blip_evaluator.evaluate_batch(eval_image_paths, eval_prompts, subset)

    results = {
        "subset": subset,
        "num_prompts": len(prompts),
        "num_images": len(generation_results),
        "generation_time_seconds": gen_time,
        "blip_vqa": blip_results,
    }

    if use_image_reward:
        print("\n--- Computing ImageReward scores ---")
        try:
            ir_evaluator = ImageRewardEvaluator()
            ir_image_paths = [r["image_path"] for r in generation_results]
            ir_prompts = [r["prompt"] for r in generation_results]
            ir_results = ir_evaluator.evaluate_batch(ir_image_paths, ir_prompts)
            results["image_reward"] = ir_results
        except Exception as e:
            print(f"ImageReward evaluation failed: {e}")
            results["image_reward"] = {"error": str(e)}

    result_path = os.path.join(str(config.output_path), subset, "results.json")
    save_evaluation_results(results, result_path)

    print(f"\n--- {subset} Results ---")
    print(f"  BLIP-VQA Score: {blip_results['mean_score']:.4f}")
    if "image_reward" in results and "mean_score" in results.get("image_reward", {}):
        print(f"  ImageReward Score: {results['image_reward']['mean_score']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="T2I-CompBench Evaluation")
    parser.add_argument("--subset", type=str, default="all",
                       choices=["color", "shape", "texture", "all"])
    parser.add_argument("--num_prompts", type=int, default=300,
                       help="Number of prompts per subset (default 300 = full T2I-CompBench)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--data_dir", type=str, default=None,
                       help="Directory with T2I-CompBench prompt files")
    parser.add_argument("--model_path", type=str,
                       default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--no_image_reward", action="store_true")
    args = parser.parse_args()

    config = T2ICompBenchConfig()
    config.seeds = args.seeds
    config.num_prompts = args.num_prompts
    config.model_path = args.model_path

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from run_demo import load_model
    model, prompt_parser = load_model(config, device)

    subsets = ["color", "shape", "texture"] if args.subset == "all" else [args.subset]

    all_results = {}
    for subset in subsets:
        results = evaluate_subset(
            subset=subset,
            config=config,
            model=model,
            prompt_parser=prompt_parser,
            num_prompts=args.num_prompts,
            seeds=args.seeds,
            data_dir=args.data_dir,
            use_image_reward=not args.no_image_reward,
        )
        all_results[subset] = results

    print("\n" + "=" * 60)
    print("T2I-CompBench Summary")
    print("=" * 60)
    print(f"{'Subset':<10} {'BLIP-VQA':>10} {'ImageReward':>12}")
    print("-" * 35)
    for subset, res in all_results.items():
        blip = res["blip_vqa"]["mean_score"]
        ir = res.get("image_reward", {}).get("mean_score", "N/A")
        ir_str = f"{ir:.4f}" if isinstance(ir, float) else ir
        print(f"{subset:<10} {blip:>10.4f} {ir_str:>12}")

    summary_path = os.path.join(str(config.output_path), "summary.json")
    save_evaluation_results(all_results, summary_path)


if __name__ == "__main__":
    main()
