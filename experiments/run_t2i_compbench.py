"""
T2I-CompBench Evaluation Script

Evaluates ToMe on the T2I-CompBench benchmark for attribute binding:
  - Color binding (300 prompts)
  - Shape binding (300 prompts)
  - Texture binding (300 prompts)

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


# ============================================================
# T2I-CompBench Prompt Templates
# ============================================================

COLOR_PROMPTS = [
    ("a red apple and a green pear", [{"object": "apple", "attribute": "red"}, {"object": "pear", "attribute": "green"}]),
    ("a blue car and a red bus", [{"object": "car", "attribute": "blue"}, {"object": "bus", "attribute": "red"}]),
    ("a yellow cat and a black dog", [{"object": "cat", "attribute": "yellow"}, {"object": "dog", "attribute": "black"}]),
    ("a white horse and a brown cow", [{"object": "horse", "attribute": "white"}, {"object": "cow", "attribute": "brown"}]),
    ("a pink flower and a purple butterfly", [{"object": "flower", "attribute": "pink"}, {"object": "butterfly", "attribute": "purple"}]),
    ("a red hat and a blue scarf", [{"object": "hat", "attribute": "red"}, {"object": "scarf", "attribute": "blue"}]),
    ("a green frog and a yellow bird", [{"object": "frog", "attribute": "green"}, {"object": "bird", "attribute": "yellow"}]),
    ("a black cat and a white rabbit", [{"object": "cat", "attribute": "black"}, {"object": "rabbit", "attribute": "white"}]),
    ("a silver ring and a gold bracelet", [{"object": "ring", "attribute": "silver"}, {"object": "bracelet", "attribute": "gold"}]),
    ("a red rose and a white lily", [{"object": "rose", "attribute": "red"}, {"object": "lily", "attribute": "white"}]),
    ("a blue bird and a red fish", [{"object": "bird", "attribute": "blue"}, {"object": "fish", "attribute": "red"}]),
    ("a orange cat and a gray mouse", [{"object": "cat", "attribute": "orange"}, {"object": "mouse", "attribute": "gray"}]),
    ("a white swan and a black crow", [{"object": "swan", "attribute": "white"}, {"object": "crow", "attribute": "black"}]),
    ("a red ball and a blue cube", [{"object": "ball", "attribute": "red"}, {"object": "cube", "attribute": "blue"}]),
    ("a green tree and a brown fence", [{"object": "tree", "attribute": "green"}, {"object": "fence", "attribute": "brown"}]),
    ("a yellow banana and a red strawberry", [{"object": "banana", "attribute": "yellow"}, {"object": "strawberry", "attribute": "red"}]),
    ("a purple grape and a orange tangerine", [{"object": "grape", "attribute": "purple"}, {"object": "tangerine", "attribute": "orange"}]),
    ("a white cloud and a blue sky", [{"object": "cloud", "attribute": "white"}, {"object": "sky", "attribute": "blue"}]),
    ("a red fire truck and a yellow taxi", [{"object": "fire truck", "attribute": "red"}, {"object": "taxi", "attribute": "yellow"}]),
    ("a black bear and a white polar bear", [{"object": "bear", "attribute": "black"}, {"object": "polar bear", "attribute": "white"}]),
]

SHAPE_PROMPTS = [
    ("a round clock and a square frame", [{"object": "clock", "attribute": "round"}, {"object": "frame", "attribute": "square"}]),
    ("a triangular roof and a rectangular door", [{"object": "roof", "attribute": "triangular"}, {"object": "door", "attribute": "rectangular"}]),
    ("a round ball and a square box", [{"object": "ball", "attribute": "round"}, {"object": "box", "attribute": "square"}]),
    ("a cylindrical vase and a spherical lamp", [{"object": "vase", "attribute": "cylindrical"}, {"object": "lamp", "attribute": "spherical"}]),
    ("a curved bridge and a straight road", [{"object": "bridge", "attribute": "curved"}, {"object": "road", "attribute": "straight"}]),
    ("a round plate and a rectangular tray", [{"object": "plate", "attribute": "round"}, {"object": "tray", "attribute": "rectangular"}]),
    ("a oval mirror and a square window", [{"object": "mirror", "attribute": "oval"}, {"object": "window", "attribute": "square"}]),
    ("a circular rug and a rectangular carpet", [{"object": "rug", "attribute": "circular"}, {"object": "carpet", "attribute": "rectangular"}]),
    ("a round table and a long bench", [{"object": "table", "attribute": "round"}, {"object": "bench", "attribute": "long"}]),
    ("a flat screen and a curved monitor", [{"object": "screen", "attribute": "flat"}, {"object": "monitor", "attribute": "curved"}]),
    ("a round cookie and a square cracker", [{"object": "cookie", "attribute": "round"}, {"object": "cracker", "attribute": "square"}]),
    ("a tall tower and a wide bridge", [{"object": "tower", "attribute": "tall"}, {"object": "bridge", "attribute": "wide"}]),
    ("a thin pencil and a thick marker", [{"object": "pencil", "attribute": "thin"}, {"object": "marker", "attribute": "thick"}]),
    ("a round wheel and a square block", [{"object": "wheel", "attribute": "round"}, {"object": "block", "attribute": "square"}]),
    ("a star shaped cookie and a heart shaped candy", [{"object": "cookie", "attribute": "star shaped"}, {"object": "candy", "attribute": "heart shaped"}]),
    ("a spiral staircase and a straight ladder", [{"object": "staircase", "attribute": "spiral"}, {"object": "ladder", "attribute": "straight"}]),
    ("a round pizza and a triangular slice", [{"object": "pizza", "attribute": "round"}, {"object": "slice", "attribute": "triangular"}]),
    ("a flat pancake and a round donut", [{"object": "pancake", "attribute": "flat"}, {"object": "donut", "attribute": "round"}]),
    ("a square tile and a hexagonal pattern", [{"object": "tile", "attribute": "square"}, {"object": "pattern", "attribute": "hexagonal"}]),
    ("a round coin and a rectangular bill", [{"object": "coin", "attribute": "round"}, {"object": "bill", "attribute": "rectangular"}]),
]

TEXTURE_PROMPTS = [
    ("a fluffy cat and a smooth dog", [{"object": "cat", "attribute": "fluffy"}, {"object": "dog", "attribute": "smooth"}]),
    ("a wooden table and a metal chair", [{"object": "table", "attribute": "wooden"}, {"object": "chair", "attribute": "metal"}]),
    ("a glossy car and a matte truck", [{"object": "car", "attribute": "glossy"}, {"object": "truck", "attribute": "matte"}]),
    ("a rough stone and a smooth pebble", [{"object": "stone", "attribute": "rough"}, {"object": "pebble", "attribute": "smooth"}]),
    ("a furry teddy bear and a plastic robot", [{"object": "teddy bear", "attribute": "furry"}, {"object": "robot", "attribute": "plastic"}]),
    ("a leather jacket and a cotton shirt", [{"object": "jacket", "attribute": "leather"}, {"object": "shirt", "attribute": "cotton"}]),
    ("a velvet curtain and a silk pillow", [{"object": "curtain", "attribute": "velvet"}, {"object": "pillow", "attribute": "silk"}]),
    ("a glass bottle and a ceramic cup", [{"object": "bottle", "attribute": "glass"}, {"object": "cup", "attribute": "ceramic"}]),
    ("a knitted sweater and a woven basket", [{"object": "sweater", "attribute": "knitted"}, {"object": "basket", "attribute": "woven"}]),
    ("a sandy beach and a rocky cliff", [{"object": "beach", "attribute": "sandy"}, {"object": "cliff", "attribute": "rocky"}]),
    ("a metallic robot and a wooden puppet", [{"object": "robot", "attribute": "metallic"}, {"object": "puppet", "attribute": "wooden"}]),
    ("a shiny diamond and a rough coal", [{"object": "diamond", "attribute": "shiny"}, {"object": "coal", "attribute": "rough"}]),
    ("a woolen hat and a straw basket", [{"object": "hat", "attribute": "woolen"}, {"object": "basket", "attribute": "straw"}]),
    ("a rubber ball and a leather glove", [{"object": "ball", "attribute": "rubber"}, {"object": "glove", "attribute": "leather"}]),
    ("a marble statue and a bronze medal", [{"object": "statue", "attribute": "marble"}, {"object": "medal", "attribute": "bronze"}]),
    ("a feathery pillow and a stone floor", [{"object": "pillow", "attribute": "feathery"}, {"object": "floor", "attribute": "stone"}]),
    ("a silky dress and a denim jacket", [{"object": "dress", "attribute": "silky"}, {"object": "jacket", "attribute": "denim"}]),
    ("a porcelain vase and a iron gate", [{"object": "vase", "attribute": "porcelain"}, {"object": "gate", "attribute": "iron"}]),
    ("a crystal glass and a wooden mug", [{"object": "glass", "attribute": "crystal"}, {"object": "mug", "attribute": "wooden"}]),
    ("a fuzzy carpet and a polished floor", [{"object": "carpet", "attribute": "fuzzy"}, {"object": "floor", "attribute": "polished"}]),
]

PROMPT_SETS = {
    "color": COLOR_PROMPTS,
    "shape": SHAPE_PROMPTS,
    "texture": TEXTURE_PROMPTS,
}


def load_t2i_compbench_prompts(
    subset: str, num_prompts: int = 300, data_dir: Optional[str] = None
) -> List[tuple]:
    """
    Load T2I-CompBench prompts. Tries to load from:
    1. Local data directory (if provided)
    2. HuggingFace dataset
    3. Built-in prompt templates (fallback)
    """
    if data_dir:
        filepath = os.path.join(data_dir, f"{subset}_val.txt")
        if os.path.exists(filepath):
            print(f"Loading prompts from {filepath}")
            with open(filepath, "r") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            return [(line, []) for line in lines[:num_prompts]]

    try:
        from datasets import load_dataset

        print(f"Loading T2I-CompBench {subset} from HuggingFace...")
        ds = load_dataset("Zhicong/T2I-CompBench", split="validation")
        prompts = [
            (row["prompt"], [])
            for row in ds
            if row.get("category", "") == subset
        ]
        if prompts:
            return prompts[:num_prompts]
    except Exception as e:
        print(f"Could not load from HuggingFace: {e}")

    print(f"Using built-in {subset} prompts ({len(PROMPT_SETS.get(subset, []))} available)")
    return PROMPT_SETS.get(subset, [])[:num_prompts]


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
            token_indices = config.token_indices
        if not prompt_anchor:
            prompt_anchor = config.prompt_anchor

        nouns = [chunk.text for chunk in doc.noun_chunks]
        merged_prompt = " and ".join(
            [n.split()[-1] if len(n.split()) > 1 else n for n in nouns]
        )
        if not merged_prompt:
            merged_prompt = prompt

        words = prompt.split()
        config.prompt_length = len(
            model.tokenizer(prompt)["input_ids"]
        ) - 2
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
                    "attr_data": attr_data,
                })
                print(f"  [{prompt_idx+1}/{len(prompts)}] seed={seed} saved: {img_filename}")

            except Exception as e:
                print(f"  Error generating prompt {prompt_idx}: {e}")
                continue

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

    print("\n--- Computing BLIP-VQA scores ---")
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

    blip_results = blip_evaluator.evaluate_batch(
        image_paths[:len(prompts_data)], prompts_data, subset
    )

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
            ir_prompts = [r["prompt"] for r in generation_results]
            ir_results = ir_evaluator.evaluate_batch(image_paths, ir_prompts)
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
    parser.add_argument("--num_prompts", type=int, default=20,
                       help="Number of prompts per subset (default 20 for quick test, use 300 for full eval)")
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
