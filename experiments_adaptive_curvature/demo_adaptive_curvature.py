#!/usr/bin/env python
"""
Demo: Adaptive Curvature Quick Verification
============================================
Runs a handful of prompts with different curvature strategies side-by-side
and saves comparison grids for visual inspection. No heavy benchmark needed.

Usage
-----
    python -m experiments_adaptive_curvature.demo_adaptive_curvature
    python -m experiments_adaptive_curvature.demo_adaptive_curvature --prompt "a red cat and a blue dog"
"""

import os
import sys
import argparse

import torch
import spacy
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipe_tome import tomePipeline
from utils.ptp_utils import AttentionStore, register_attention_control
from prompt_utils import PromptParser

from experiments_adaptive_curvature.adaptive_curvature_utils import (
    estimate_prompt_curvature,
    estimate_entity_curvatures,
)

DEMO_PROMPTS = [
    "a red cat and a blue dog",
    "a cat wearing sunglasses and a dog wearing hat",
    "a white horse with a golden crown and a black sheep with a silver necklace",
    "a fluffy cat and a smooth dog",
]


def parse_args():
    p = argparse.ArgumentParser(description="Adaptive curvature quick demo")
    p.add_argument("--prompt", type=str, default=None,
                   help="Single prompt to test (overrides DEMO_PROMPTS)")
    p.add_argument("--model_path", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--output_dir", default="demo_output_adaptive_curvature")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_inference_steps", type=int, default=30,
                   help="Fewer steps for faster demo")
    return p.parse_args()


def generate_merged_prompt(prompt, doc):
    chunks = [
        (chunk, chunk.root.text)
        for chunk in doc.noun_chunks
        if chunk.text not in ("top", "the side", "the left", "the right")
    ]
    if not chunks:
        return prompt
    merged = prompt
    for chunk, root in reversed(chunks):
        det = ""
        for token in chunk:
            if token.dep_ == "det":
                det = token.text + " "
                break
        merged = merged[:chunk.start_char] + det + root + merged[chunk.end_char:]
    return merged


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    prompts = [args.prompt] if args.prompt else DEMO_PROMPTS
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model …")
    pipeline = tomePipeline.from_pretrained(
        args.model_path, torch_dtype=torch.float16, variant="fp16",
        safety_checker=None,
    ).to(device)
    pipeline.unet.requires_grad_(False)
    pipeline.vae.requires_grad_(False)

    nlp = spacy.load("en_core_web_trf")
    prompt_parser = PromptParser(args.model_path)
    thresholds = {i: max(26 - i, 20.5) for i in range(10)}

    strategies = [
        ("euclidean", False, None),
        ("fixed_c1.0", True, 1.0),
        ("adaptive_prompt", True, "adaptive_prompt"),
        ("adaptive_entity", True, "adaptive_entity"),
    ]

    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")

        doc = nlp(prompt)
        prompt_parser.set_doc(doc)
        try:
            ti = prompt_parser._get_indices(prompt)
            pa = prompt_parser._split_prompt(doc)
            filtered_idx, filtered_anchor = [], []
            for i, idx in enumerate(ti):
                if len(idx[1]) > 0:
                    filtered_idx.append(idx)
                    if i < len(pa):
                        filtered_anchor.append(pa[i])
            ti, pa = filtered_idx, filtered_anchor
        except Exception:
            ti, pa = [], []

        merged = generate_merged_prompt(prompt, doc)
        pl = len(pipeline.tokenizer(prompt)["input_ids"]) - 2

        if doc is not None:
            prompt_c = estimate_prompt_curvature(doc)
            entity_cs = estimate_entity_curvatures(doc, ti) if ti else [1.0]
            print(f"  Adaptive prompt curvature: {prompt_c:.2f}")
            print(f"  Adaptive entity curvatures: {[f'{c:.2f}' for c in entity_cs]}")

        images = []
        labels = []

        for label, use_hyp, c_spec in strategies:
            if c_spec == "adaptive_prompt":
                curvature = prompt_c
            elif c_spec == "adaptive_entity":
                curvature = entity_cs[0] if entity_cs else 1.0
            elif c_spec is not None:
                curvature = c_spec
            else:
                curvature = 1.0

            g = torch.Generator(device).manual_seed(args.seed)
            controller = AttentionStore()
            register_attention_control(pipeline, controller)

            out = pipeline(
                prompt=prompt,
                guidance_scale=7.5,
                generator=g,
                num_inference_steps=args.n_inference_steps,
                attention_store=controller,
                indices_to_alter=ti,
                prompt_anchor=pa,
                attention_res=32,
                run_standard_sd=not ti,
                thresholds=thresholds,
                scale_factor=3,
                scale_range=(1.0, 0.0),
                prompt3=merged,
                prompt_length=pl,
                token_refinement_steps=3,
                attention_refinement_steps=[4, 4],
                tome_control_steps=[7, 7],
                eot_replace_step=30,
                use_pose_loss=False,
                use_hyperbolic=use_hyp,
                hyperbolic_curvature=curvature,
                negative_prompt="low res, ugly, blurry, artifact, unreal",
            )
            img = out.images[0]
            images.append(img)
            labels.append(f"{label} (c={curvature:.2f})" if use_hyp else label)
            print(f"  Generated: {label}")

        # Save individual images and grid
        safe_name = prompt[:50].replace(" ", "_").replace("/", "_")
        prompt_dir = os.path.join(args.output_dir, safe_name)
        os.makedirs(prompt_dir, exist_ok=True)
        for img, label in zip(images, labels):
            img.save(os.path.join(prompt_dir, f"{label}.png"))

        # Create comparison grid
        w, h = images[0].size
        grid = Image.new("RGB", (w * len(images), h + 40), "white")
        for i, (img, label) in enumerate(zip(images, labels)):
            grid.paste(img, (i * w, 40))
        grid.save(os.path.join(prompt_dir, "comparison_grid.png"))
        print(f"  Saved to {prompt_dir}/")

    del pipeline
    torch.cuda.empty_cache()
    print("\nDemo complete.")


if __name__ == "__main__":
    main()
