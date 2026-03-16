#!/usr/bin/env python
"""
Adaptive Curvature Experiment — T2I-CompBench Evaluation
========================================================
Compares fixed-curvature GeoBind against adaptive-curvature variants
on the T2I-CompBench benchmark (color, shape, texture).

Methods tested:
  - ToMe           : Euclidean baseline (from run_compbench_eval_v2.py)
  - GeoBind_fixed  : Hyperbolic with fixed c=1.0
  - GeoBind_adaptive_prompt : Per-prompt adaptive curvature
  - GeoBind_adaptive_entity : Per-entity adaptive curvature

Usage
-----
    python -m experiments_adaptive_curvature.run_adaptive_curvature --phase generate
    python -m experiments_adaptive_curvature.run_adaptive_curvature --phase all
    python -m experiments_adaptive_curvature.run_adaptive_curvature --phase evaluate
"""

import os
import sys
import json
import logging
import argparse
import traceback
from datetime import datetime

import torch
import torch.nn.functional as F
import spacy
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipe_tome import tomePipeline, token_merge
from utils.ptp_utils import AttentionStore, register_attention_control
from utils.hyperbolic_utils import token_merge_hyperbolic
from prompt_utils import PromptParser
from transformers import BlipProcessor, BlipForQuestionAnswering

from experiments_adaptive_curvature.adaptive_curvature_utils import (
    estimate_prompt_curvature,
    estimate_entity_curvatures,
    token_merge_adaptive,
)

SUBSETS = ["color", "shape", "texture"]
METHODS = [
    "ToMe",
    "GeoBind_fixed",
    "GeoBind_adaptive_prompt",
    "GeoBind_adaptive_entity",
]

# ═══════════════════════════════════════════════════════════════
#  Logging
# ═══════════════════════════════════════════════════════════════

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"adaptive_log_{ts}.txt")
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger = logging.getLogger("adaptive_c")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

log = logging.getLogger("adaptive_c")


def parse_args():
    p = argparse.ArgumentParser(description="Adaptive Curvature Experiment")
    p.add_argument("--phase", choices=["generate", "evaluate", "all"], default="all")
    p.add_argument("--subsets", nargs="+", default=SUBSETS)
    p.add_argument("--methods", nargs="+", default=METHODS)
    p.add_argument("--model_path", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--blip_model", default="Salesforce/blip-vqa-base")
    p.add_argument("--output_dir", default="eval_results_adaptive_curvature")
    p.add_argument("--data_dir", default="data/t2i_compbench")
    p.add_argument("--n_inference_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--fixed_curvature", type=float, default=1.0)
    p.add_argument("--np_num", type=int, default=8)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
#  Data & prompt parsing
# ═══════════════════════════════════════════════════════════════

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
        merged = merged[:chunk.start_char] + det + root + merged[chunk.end_char:]
    return merged


def parse_prompt_for_tome(prompt, nlp, prompt_parser, tokenizer):
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
    return filtered_idx, filtered_anchor, merged, prompt_length, doc


# ═══════════════════════════════════════════════════════════════
#  Image generation
# ═══════════════════════════════════════════════════════════════

def generate_all_images(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("Loading pipeline …")
    pipeline = tomePipeline.from_pretrained(
        args.model_path, torch_dtype=torch.float16, variant="fp16",
        safety_checker=None,
    ).to(device)
    pipeline.unet.requires_grad_(False)
    pipeline.vae.requires_grad_(False)

    log.info("Loading spaCy + PromptParser …")
    nlp = spacy.load("en_core_web_trf")
    prompt_parser = PromptParser(args.model_path)
    thresholds = {
        0: 26, 1: 25, 2: 24, 3: 23, 4: 22.5,
        5: 22, 6: 21.5, 7: 20.5, 8: 20.5, 9: 20.5,
    }

    for subset in args.subsets:
        prompts = load_prompts(args.data_dir, subset)
        log.info(f"{'═'*55}")
        log.info(f"  Subset: {subset}  ({len(prompts)} prompts)")
        log.info(f"{'═'*55}")

        for method in args.methods:
            out_dir = os.path.join(args.output_dir, method, subset, "samples")
            os.makedirs(out_dir, exist_ok=True)

            existing = len([f for f in os.listdir(out_dir) if f.endswith(".png")])
            if existing >= len(prompts):
                log.info(f"  [{method}] {existing} images exist — skipping.")
                continue
            log.info(f"  [{method}] generating ({existing}/{len(prompts)} done) …")

            n_ok, n_err = 0, 0
            for idx, prompt in enumerate(tqdm(prompts, desc=f"  {method}")):
                img_path = os.path.join(out_dir, f"{prompt}_{idx}.png")
                if os.path.exists(img_path):
                    continue

                g = torch.Generator(device).manual_seed(args.seed)
                try:
                    ti, pa, merged, pl, doc = parse_prompt_for_tome(
                        prompt, nlp, prompt_parser, pipeline.tokenizer)
                except Exception:
                    ti, pa, merged, pl, doc = [], [], prompt, 0, None

                run_std = (method == "SDXL") or (not ti)

                use_hyp = method != "ToMe"
                if method == "GeoBind_fixed":
                    curvature = args.fixed_curvature
                elif method == "GeoBind_adaptive_prompt" and doc is not None:
                    curvature = estimate_prompt_curvature(doc)
                elif method == "GeoBind_adaptive_entity":
                    curvature = 1.0  # per-entity handled separately below
                else:
                    curvature = 1.0

                controller = AttentionStore()
                register_attention_control(pipeline, controller)

                try:
                    if method == "GeoBind_adaptive_entity" and ti and doc is not None:
                        entity_curvatures = estimate_entity_curvatures(doc, ti)
                        # Manually merge tokens before calling pipeline
                        kw = dict(
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
                            token_refinement_steps=4,
                            attention_refinement_steps=[5, 4],
                            tome_control_steps=[10, 10],
                            eot_replace_step=30,
                            use_pose_loss=False,
                            use_hyperbolic=True,
                            hyperbolic_curvature=entity_curvatures[0] if entity_curvatures else 1.0,
                            negative_prompt="low res, ugly, blurry, artifact, unreal",
                        )
                    else:
                        kw = dict(
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
                            token_refinement_steps=4,
                            attention_refinement_steps=[5, 4],
                            tome_control_steps=[10, 10],
                            eot_replace_step=30,
                            use_pose_loss=False,
                            use_hyperbolic=use_hyp,
                            hyperbolic_curvature=curvature,
                            negative_prompt="low res, ugly, blurry, artifact, unreal",
                        )

                    out = pipeline(**kw)
                    out.images[0].save(img_path)
                    n_ok += 1
                except Exception as e:
                    tqdm.write(f"    [ERROR] '{prompt}': {e}")
                    log.error(f"  Error [{method}/{subset}] '{prompt}': {e}")
                    Image.new("RGB", (1024, 1024), "gray").save(img_path)
                    n_err += 1

            log.info(f"  [{method}/{subset}] done — ok={n_ok}, err={n_err}")

    del pipeline
    torch.cuda.empty_cache()
    log.info("Generation phase complete.")


# ═══════════════════════════════════════════════════════════════
#  Evaluation (BLIP-VQA, same protocol as run_compbench_eval_v2)
# ═══════════════════════════════════════════════════════════════

def compute_vqa_yes_prob(model, processor, image, question, device):
    inputs = processor(image, question, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, output_scores=True, return_dict_in_generate=True,
            max_new_tokens=10,
        )
    first_logits = outputs.scores[0]
    probs = F.softmax(first_logits, dim=-1)
    yes_ids = processor.tokenizer("yes", add_special_tokens=False)["input_ids"]
    return probs[0, yes_ids[0]].item()


def evaluate_all(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_path = os.path.join(args.output_dir, "blip_vqa_results.json")
    results = {}
    if os.path.isfile(results_path):
        with open(results_path) as f:
            results = json.load(f)

    log.info(f"Loading BLIP-VQA: {args.blip_model}")
    processor = BlipProcessor.from_pretrained(args.blip_model)
    model = BlipForQuestionAnswering.from_pretrained(args.blip_model).to(device).eval()
    nlp_sm = spacy.load("en_core_web_sm")

    for subset in args.subsets:
        prompts = load_prompts(args.data_dir, subset)
        if subset not in results:
            results[subset] = {}

        for method in args.methods:
            if results[subset].get(method) is not None:
                log.info(f"  [{method}/{subset}] cached: {results[subset][method]}")
                continue

            images_dir = os.path.join(args.output_dir, method, subset, "samples")
            if not os.path.isdir(images_dir):
                log.warning(f"  [SKIP] {images_dir} not found")
                results[subset][method] = None
                continue

            log.info(f"  Evaluating {method}/{subset} …")
            n = len(prompts)
            reward = torch.ones((n, args.np_num), device=device)

            for k, prompt in enumerate(tqdm(prompts, desc=f"    {method}")):
                doc = nlp_sm(prompt)
                nps = [
                    c.text for c in doc.noun_chunks
                    if c.text not in ("top", "the side", "the left", "the right")
                ]
                img_path = os.path.join(images_dir, f"{prompt}_{k}.png")
                if not os.path.exists(img_path) or not nps:
                    for j in range(min(len(nps), args.np_num)):
                        reward[k, j] = 0.0
                    continue
                image = Image.open(img_path).convert("RGB")
                for j, np_text in enumerate(nps[:args.np_num]):
                    score = compute_vqa_yes_prob(
                        model, processor, image, f"{np_text}?", device)
                    reward[k, j] = score

            reward_final = reward[:, 0]
            for i in range(1, args.np_num):
                reward_final = reward_final * reward[:, i]
            final_score = reward_final.mean().item()

            results[subset][method] = round(final_score, 4)
            log.info(f"  ➜ [{method}/{subset}] BLIP-VQA = {final_score:.4f}")

            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

    del model, processor
    torch.cuda.empty_cache()
    return results


def print_result_table(results):
    log.info("")
    log.info("=" * 74)
    log.info("  Adaptive Curvature — BLIP-VQA Scores")
    log.info("=" * 74)
    header = f"  {'Method':<30}"
    for s in SUBSETS:
        header += f"{s.capitalize():<14}"
    log.info(header)
    log.info("  " + "-" * 68)
    for m in METHODS:
        row = f"  {m:<30}"
        for s in SUBSETS:
            val = results.get(s, {}).get(m)
            row += f"{val:<14.4f}" if val is not None else f"{'N/A':<14}"
        log.info(row)
    log.info("=" * 74)


def main():
    args = parse_args()
    setup_logging(args.output_dir)
    log.info("=== Adaptive Curvature Experiment ===")
    log.info(f"Phase: {args.phase} | Methods: {args.methods}")

    if args.phase in ("all", "generate"):
        generate_all_images(args)
    if args.phase in ("all", "evaluate"):
        results = evaluate_all(args)
        print_result_table(results)


if __name__ == "__main__":
    main()
