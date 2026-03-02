"""
GPT-4o Object Binding Benchmark (Table 1, GPT-4o column)

Evaluates object binding using 50 prompts with the template:
  "a [objectA] with a [itemA] and a [objectB] with a [itemB]"

Images are generated and scored by GPT-4o using the 9-level rubric.

Usage:
    python -m experiments.run_gpt4o_benchmark [--num_prompts 50] [--seeds 42]
    
    Set OPENAI_API_KEY environment variable before running.
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

from configs.experiment_config import GPT4oBenchmarkConfig
from experiments.eval_gpt4o import GPT4oEvaluator, get_gpt4o_prompts
from experiments.eval_metrics import save_evaluation_results


def generate_gpt4o_benchmark_images(
    prompts: List[str],
    config,
    model,
    prompt_parser,
    seeds: List[int],
    output_dir: str,
) -> List[Dict]:
    """Generate images for GPT-4o benchmark prompts."""
    from utils.ptp_utils import AttentionStore
    from run_demo import run_on_prompt, filter_text

    os.makedirs(output_dir, exist_ok=True)
    results = []

    try:
        import en_core_web_trf
        nlp = en_core_web_trf.load()
    except ImportError:
        import spacy
        nlp = spacy.load("en_core_web_sm")

    for prompt_idx, prompt in enumerate(prompts):
        config.prompt = prompt

        doc = nlp(prompt)
        prompt_parser.set_doc(doc)
        token_indices = prompt_parser._get_indices(prompt)
        prompt_anchor = prompt_parser._split_prompt(doc)
        token_indices, prompt_anchor = filter_text(token_indices, prompt_anchor)

        nouns = [chunk.root.text for chunk in doc.noun_chunks]
        merged_parts = []
        for chunk in doc.noun_chunks:
            root = chunk.root.text
            if root not in merged_parts:
                merged_parts.append(root)
        merged_prompt = " and ".join([f"a {n}" for n in merged_parts[:2]])
        if not merged_prompt:
            merged_prompt = prompt

        config.prompt_length = len(model.tokenizer(prompt)["input_ids"]) - 2
        config.prompt_merged = merged_prompt

        for seed in seeds:
            g = torch.Generator("cuda").manual_seed(seed)
            controller = AttentionStore()

            try:
                image = run_on_prompt(
                    prompt=prompt,
                    model=model,
                    controller=controller,
                    token_indices=token_indices,
                    prompt_anchor=prompt_anchor,
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
                })
                print(f"  [{prompt_idx+1}/{len(prompts)}] seed={seed}: {prompt[:50]}...")

            except Exception as e:
                print(f"  Error: {e}")
                continue

    return results


def main():
    parser = argparse.ArgumentParser(description="GPT-4o Object Binding Benchmark")
    parser.add_argument("--num_prompts", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--model_path", type=str,
                       default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--openai_model", type=str, default="gpt-4o")
    parser.add_argument("--generate_only", action="store_true",
                       help="Only generate images, skip GPT-4o scoring")
    args = parser.parse_args()

    config = GPT4oBenchmarkConfig()
    config.seeds = args.seeds
    config.model_path = args.model_path

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from run_demo import load_model
    model, prompt_parser = load_model(config, device)

    prompts = get_gpt4o_prompts()[:args.num_prompts]
    print(f"Loaded {len(prompts)} GPT-4o benchmark prompts")

    output_dir = os.path.join(str(config.output_path), "images")

    print("\n--- Generating images ---")
    start_time = time.time()
    generation_results = generate_gpt4o_benchmark_images(
        prompts, config, model, prompt_parser, args.seeds, output_dir
    )
    gen_time = time.time() - start_time
    print(f"Generation complete: {len(generation_results)} images in {gen_time:.1f}s")

    gen_results_path = os.path.join(str(config.output_path), "generation_results.json")
    save_evaluation_results(
        {"results": generation_results, "generation_time": gen_time},
        gen_results_path,
    )

    if args.generate_only:
        print("Skipping GPT-4o scoring (--generate_only)")
        return

    api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\nNo OpenAI API key found. Set OPENAI_API_KEY to run GPT-4o scoring.")
        print("Images have been generated. Re-run with --openai_api_key to score them.")
        return

    print("\n--- Scoring with GPT-4o ---")
    evaluator = GPT4oEvaluator(api_key=api_key)

    image_paths = [r["image_path"] for r in generation_results]
    eval_prompts = [r["prompt"] for r in generation_results]

    gpt4o_results = evaluator.evaluate_batch(
        image_paths, eval_prompts, model=args.openai_model
    )

    print(f"\n--- GPT-4o Benchmark Results ---")
    print(f"  Mean Score: {gpt4o_results['mean_score']:.4f}")
    print(f"  Std Score:  {gpt4o_results['std_score']:.4f}")
    print(f"  Valid:      {gpt4o_results['num_valid']}/{gpt4o_results['num_total']}")

    final_results = {
        "generation": {"num_images": len(generation_results), "time": gen_time},
        "gpt4o_evaluation": gpt4o_results,
    }

    results_path = os.path.join(str(config.output_path), "gpt4o_results.json")
    save_evaluation_results(final_results, results_path)


if __name__ == "__main__":
    main()
