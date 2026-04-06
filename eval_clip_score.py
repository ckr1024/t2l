#!/usr/bin/env python
"""
CLIP Score Evaluator for T2I-CompBench
=======================================
Compute CLIP Score (ViT-L/14) between generated images and their prompts.

Expected layout:
    <image_root>/<method>/<subset>/samples/<prompt>_<idx>.png

Usage:
    python eval_clip_score.py --image_root eval_results
    python eval_clip_score.py --image_root eval_results --methods GeoBind ToMe SDXL
"""

import os
import json
import argparse

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

SUBSETS = ["color", "shape", "texture"]


def load_prompts(data_dir, subset):
    path = os.path.join(data_dir, f"{subset}_val.txt")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def detect_pairs(image_root, methods=None, subsets=None):
    pairs = []
    if not os.path.isdir(image_root):
        return pairs
    for method in sorted(os.listdir(image_root)):
        method_dir = os.path.join(image_root, method)
        if not os.path.isdir(method_dir):
            continue
        if methods and method not in methods:
            continue
        for subset in sorted(os.listdir(method_dir)):
            if subsets and subset not in subsets:
                continue
            samples_dir = os.path.join(method_dir, subset, "samples")
            if os.path.isdir(samples_dir):
                n = len([f for f in os.listdir(samples_dir) if f.endswith(".png")])
                if n > 0:
                    pairs.append((method, subset, n))
    return pairs


@torch.no_grad()
def evaluate(images_dir, prompts, model, processor, device):
    scores = []
    for k, prompt in enumerate(tqdm(prompts, desc="  CLIP Score")):
        img_path = os.path.join(images_dir, f"{prompt}_{k}.png")
        if not os.path.exists(img_path):
            continue
        image = Image.open(img_path).convert("RGB")
        inputs = processor(text=[prompt], images=[image],
                           return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        score = outputs.logits_per_image.item() / 100.0
        scores.append(score)
    mean_score = sum(scores) / len(scores) if scores else 0.0
    return mean_score, len(scores), len(prompts)


def main():
    parser = argparse.ArgumentParser(description="CLIP Score evaluator")
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--subsets", nargs="+", default=None)
    parser.add_argument("--data_dir", default="data/t2i_compbench")
    parser.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = detect_pairs(args.image_root, args.methods, args.subsets)
    if not pairs:
        print(f"No images found under {args.image_root}")
        return

    results_path = os.path.join(args.image_root, "clip_score_results.json")
    results = {} if args.force else (
        json.load(open(results_path)) if os.path.isfile(results_path) else {}
    )

    print(f"Loading CLIP model: {args.clip_model} ...")
    model = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.clip_model)

    for method, subset, n_imgs in pairs:
        results.setdefault(subset, {})
        if not args.force and results[subset].get(method) is not None:
            print(f"  [{method}/{subset}] cached: {results[subset][method]:.4f}")
            continue

        images_dir = os.path.join(args.image_root, method, subset, "samples")
        prompts = load_prompts(args.data_dir, subset)
        print(f"\nEvaluating {method}/{subset} ({n_imgs} images)")

        score, n_valid, n_total = evaluate(
            images_dir, prompts, model, processor, device)
        results[subset][method] = round(score, 4)
        print(f"  CLIP Score = {score:.4f}  ({n_valid}/{n_total} valid)")

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    print("\n" + "=" * 55)
    print("  CLIP Score Results")
    print("=" * 55)
    all_subsets = [s for s in SUBSETS if s in results]
    all_methods = sorted({m for s in all_subsets for m in results[s]})
    header = f"  {'Method':<16}" + "".join(f"{s.capitalize():<14}" for s in all_subsets)
    print(header)
    print("  " + "-" * 50)
    for m in all_methods:
        row = f"  {m:<16}"
        for s in all_subsets:
            val = results.get(s, {}).get(m)
            row += f"{val:<14.4f}" if val is not None else f"{'N/A':<14}"
        print(row)
    print("=" * 55)
    print(f"Results saved -> {results_path}")


if __name__ == "__main__":
    main()
