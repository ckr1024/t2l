#!/usr/bin/env python
"""
GeoBind — Image Generation
============================
Generate images using the GeoBind pipeline (hyperbolic token merging +
contrastive binding loss) on T2I-CompBench attribute-binding subsets.

Usage
-----
    python run_geobind.py
    python run_geobind.py --output_dir eval_results --subsets color texture
    python run_geobind.py --seed 123
"""

import os
import sys
import logging
import argparse
from datetime import datetime

import torch
import spacy
from PIL import Image
from tqdm import tqdm

from pipe_geobind import geobindPipeline, TokenMergerWithAttnHyperspace
from utils.ptp_utils import AttentionStore, register_attention_control
from prompt_utils import PromptParser

SUBSETS = ["color", "shape", "texture"]
METHOD = "GeoBind"

# ─────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"geobind_log_{ts}.txt")

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger = logging.getLogger("geobind")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info(f"Logging to {log_path}")
    return logger

log = logging.getLogger("geobind")

# ─────────────────────────────────────────────────────────
#  Data & prompt parsing
# ─────────────────────────────────────────────────────────

def load_prompts(data_dir, subset):
    path = os.path.join(data_dir, f"{subset}_val.txt")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def generate_merged_prompt(prompt, doc):
    chunks = [
        (chunk, chunk.root.text)
        for chunk in doc.noun_chunks
        if chunk.text not in ["top", "the side", "the left", "the right"]
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
        merged = merged[: chunk.start_char] + det + root + merged[chunk.end_char :]
    return merged


def parse_prompt(prompt, nlp, prompt_parser, tokenizer):
    doc = nlp(prompt)
    prompt_parser.set_doc(doc)

    token_indices = prompt_parser._get_indices(prompt)
    prompt_anchor = prompt_parser._split_prompt(doc)

    filtered_idx, filtered_anchor = [], []
    for i, idx in enumerate(token_indices):
        if len(idx[1]) > 0:
            filtered_idx.append(idx)
            if i < len(prompt_anchor):
                filtered_anchor.append(prompt_anchor[i])

    merged = generate_merged_prompt(prompt, doc)
    prompt_length = len(tokenizer(prompt)["input_ids"]) - 2
    return filtered_idx, filtered_anchor, merged, prompt_length

# ─────────────────────────────────────────────────────────
#  Image generation
# ─────────────────────────────────────────────────────────

def generate_images(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("Loading GeoBind pipeline …")
    pipeline = geobindPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.float16, variant="fp16",
        safety_checker=None,
    ).to(device)
    pipeline.unet.requires_grad_(False)
    pipeline.vae.requires_grad_(False)

    hyper_merger = (
        TokenMergerWithAttnHyperspace(embed_dim=2048, num_heads=8)
        .to(device).eval()
    )

    log.info("Loading spaCy + PromptParser …")
    nlp = spacy.load("en_core_web_trf")
    prompt_parser = PromptParser(args.model_path)
    thresholds = {i: max(26 - i * 0.5, 21) for i in range(20)}

    total_ok, total_err = 0, 0
    for subset in args.subsets:
        prompts = load_prompts(args.data_dir, subset)
        out_dir = os.path.join(args.output_dir, METHOD, subset, "samples")
        os.makedirs(out_dir, exist_ok=True)

        existing = len([f for f in os.listdir(out_dir) if f.endswith(".png")])
        log.info(f"{'═'*55}")
        log.info(f"  {METHOD} / {subset}  ({len(prompts)} prompts, {existing} exist)")
        log.info(f"{'═'*55}")
        if existing >= len(prompts):
            log.info("  All images exist — skipping generation.")
            continue

        n_ok, n_err = 0, 0
        for idx, prompt in enumerate(tqdm(prompts, desc=f"  {METHOD}/{subset}")):
            img_path = os.path.join(out_dir, f"{prompt}_{idx}.png")
            if os.path.exists(img_path):
                continue

            g = torch.Generator(device).manual_seed(args.seed)
            try:
                ti, pa, merged, pl = parse_prompt(
                    prompt, nlp, prompt_parser, pipeline.tokenizer)
            except Exception:
                ti, pa, merged, pl = [], [], prompt, 0

            run_std = not ti
            controller = AttentionStore()
            register_attention_control(pipeline, controller)

            try:
                out = pipeline(
                    prompt=prompt,
                    guidance_scale=args.guidance_scale,
                    generator=g,
                    num_inference_steps=args.n_inference_steps,
                    attention_store=controller,
                    indices_to_alter=ti,
                    prompt_anchor=pa,
                    attention_res=32,
                    run_standard_sd=run_std,
                    thresholds=thresholds,
                    scale_factor=3,
                    scale_range=(1.0, 0.0),
                    prompt3=merged,
                    prompt_length=pl,
                    token_refinement_steps=5,
                    attention_refinement_steps=[6, 6],
                    tome_control_steps=[10, 10],
                    eot_replace_step=0,
                    use_pose_loss=False,
                    negative_prompt="low res, ugly, blurry, artifact, unreal",
                    hyper_merger=hyper_merger,
                )
                out.images[0].save(img_path)
                n_ok += 1
            except Exception as e:
                tqdm.write(f"    [ERROR] '{prompt}': {e}")
                log.error(f"  Error [{subset}] '{prompt}': {e}")
                Image.new("RGB", (1024, 1024), "gray").save(img_path)
                n_err += 1

        log.info(f"  [{subset}] done — generated={n_ok}, errors={n_err}")
        total_ok += n_ok
        total_err += n_err

    del pipeline
    torch.cuda.empty_cache()
    log.info(f"All done. total generated={total_ok}, errors={total_err}")

# ─────────────────────────────────────────────────────────
#  CLI + main
# ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GeoBind: generate images")
    p.add_argument("--output_dir", default="eval_results",
                   help="Root output directory (images → <output_dir>/GeoBind/<subset>/samples/)")
    p.add_argument("--subsets", nargs="+", default=SUBSETS)
    p.add_argument("--model_path", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--data_dir", default="data/t2i_compbench")
    p.add_argument("--n_inference_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.output_dir)

    log.info(f"Subsets: {args.subsets}")
    log.info(f"Output dir: {os.path.abspath(args.output_dir)}")
    log.info("═══  Generating GeoBind images  ═══")

    try:
        generate_images(args)
    except Exception:
        log.error(f"Generation FAILED:\n{import_traceback()}")


def import_traceback():
    import traceback
    return traceback.format_exc()


if __name__ == "__main__":
    main()
