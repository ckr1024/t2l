#!/usr/bin/env python
"""
GeoBind — Generate + Auto-Evaluate
====================================
Generate images using the GeoBind pipeline (hyperbolic token merging +
contrastive binding loss), then automatically run BLIP-VQA evaluation.

Usage
-----
    # Generate + evaluate, default output dir
    python run_geobind.py --output_dir eval_results

    # Specific subsets only
    python run_geobind.py --output_dir eval_results --subsets color texture

    # Generate only, skip evaluation
    python run_geobind.py --output_dir eval_results --skip_eval

    # Skip generation (images exist), evaluate only
    python run_geobind.py --output_dir eval_results --eval_only
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
from transformers import BlipProcessor, BlipForQuestionAnswering

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
#  Result persistence
# ─────────────────────────────────────────────────────────

def _results_path(output_dir):
    return os.path.join(output_dir, "blip_vqa_results.json")

def load_results(output_dir):
    p = _results_path(output_dir)
    if os.path.isfile(p):
        with open(p) as f:
            return json.load(f)
    return {}

def save_results(results, output_dir):
    p = _results_path(output_dir)
    with open(p, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved → {p}")

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
#  Phase 1 — Generation (GeoBind only)
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
                    token_refinement_steps=3,
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

    del pipeline
    torch.cuda.empty_cache()
    log.info("Generation complete.")

# ─────────────────────────────────────────────────────────
#  Phase 2 — BLIP-VQA evaluation
# ─────────────────────────────────────────────────────────

def compute_vqa_yes_prob(model, processor, image, question, device):
    inputs = processor(image, question, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, output_scores=True,
            return_dict_in_generate=True, max_new_tokens=10,
        )
    probs = F.softmax(outputs.scores[0], dim=-1)
    yes_ids = processor.tokenizer("yes", add_special_tokens=False)["input_ids"]
    return probs[0, yes_ids[0]].item()


def evaluate_images(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = load_results(args.output_dir)

    log.info(f"Loading BLIP-VQA: {args.blip_model} …")
    try:
        processor = BlipProcessor.from_pretrained(args.blip_model)
        model = BlipForQuestionAnswering.from_pretrained(args.blip_model).to(device).eval()
    except Exception:
        log.error(f"Failed to load BLIP:\n{traceback.format_exc()}")
        return results

    try:
        nlp_sm = spacy.load("en_core_web_sm")
    except OSError:
        log.error("spacy en_core_web_sm not found. Run: python -m spacy download en_core_web_sm")
        return results

    for subset in args.subsets:
        if subset not in results:
            results[subset] = {}

        existing = results[subset].get(METHOD)
        if existing is not None:
            log.info(f"  [{METHOD}/{subset}] already evaluated: {existing} — skip")
            continue

        images_dir = os.path.join(args.output_dir, METHOD, subset, "samples")
        if not os.path.isdir(images_dir):
            log.warning(f"  [SKIP] {images_dir} not found")
            results[subset][METHOD] = None
            save_results(results, args.output_dir)
            continue

        prompts = load_prompts(args.data_dir, subset)
        log.info(f"{'─'*50}")
        log.info(f"  Evaluating  {METHOD} / {subset}")
        log.info(f"{'─'*50}")

        n = len(prompts)
        reward = torch.ones((n, args.np_num), device=device)

        all_nps = []
        for prompt in prompts:
            doc = nlp_sm(prompt)
            nps = [chunk.text for chunk in doc.noun_chunks
                   if chunk.text not in ["top", "the side", "the left", "the right"]]
            all_nps.append(nps)

        per_image = []
        try:
            for k, prompt in enumerate(tqdm(prompts, desc=f"    BLIP-VQA {subset}")):
                nps = all_nps[k]
                detail = {"prompt": prompt, "index": k, "noun_phrases": nps, "scores": []}
                if not nps:
                    per_image.append(detail)
                    continue

                img_path = os.path.join(images_dir, f"{prompt}_{k}.png")
                if not os.path.exists(img_path):
                    for j in range(min(len(nps), args.np_num)):
                        reward[k, j] = 0.0
                        detail["scores"].append({"np": nps[j], "score": 0.0, "note": "missing"})
                    per_image.append(detail)
                    continue

                image = Image.open(img_path).convert("RGB")
                for j, np_text in enumerate(nps[:args.np_num]):
                    score = compute_vqa_yes_prob(model, processor, image, f"{np_text}?", device)
                    reward[k, j] = score
                    detail["scores"].append({"np": np_text, "score": round(score, 6)})
                per_image.append(detail)

            reward_final = reward[:, 0]
            for i in range(1, args.np_num):
                reward_final = reward_final * reward[:, i]
            final_score = reward_final.mean().item()

            for k in range(n):
                per_image[k]["image_score"] = round(reward_final[k].item(), 6)

            detail_path = os.path.join(args.output_dir, METHOD, subset, "vqa_detail.json")
            os.makedirs(os.path.dirname(detail_path), exist_ok=True)
            with open(detail_path, "w") as f:
                json.dump({"blip_vqa_score": round(final_score, 6),
                           "per_image": per_image}, f, indent=2, ensure_ascii=False)
            log.info(f"    Detail saved → {detail_path}")

            results[subset][METHOD] = round(final_score, 4)
            log.info(f"  ➜ BLIP-VQA score = {final_score:.4f}")
        except Exception:
            log.error(f"  FAILED {METHOD}/{subset}:\n{traceback.format_exc()}")
            results[subset][METHOD] = None

        save_results(results, args.output_dir)

    del model, processor
    torch.cuda.empty_cache()
    return results

# ─────────────────────────────────────────────────────────
#  Report
# ─────────────────────────────────────────────────────────

def print_geobind_scores(results):
    subsets = [s for s in SUBSETS if s in results]
    log.info("")
    log.info("=" * 40)
    log.info("  GeoBind  BLIP-VQA Scores")
    log.info("=" * 40)
    for s in subsets:
        val = results.get(s, {}).get(METHOD)
        score_str = f"{val:.4f}" if val is not None else "N/A"
        log.info(f"  {s.capitalize():<12} {score_str}")
    log.info("=" * 40)

# ─────────────────────────────────────────────────────────
#  CLI + main
# ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GeoBind: generate + auto-evaluate")
    p.add_argument("--output_dir", default="eval_results",
                   help="Root output directory (images saved to <output_dir>/GeoBind/)")
    p.add_argument("--subsets", nargs="+", default=SUBSETS)
    p.add_argument("--model_path", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--blip_model", default="Salesforce/blip-vqa-base")
    p.add_argument("--data_dir", default="data/t2i_compbench")
    p.add_argument("--n_inference_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--np_num", type=int, default=8)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--skip_eval", action="store_true",
                   help="Only generate, do not evaluate")
    p.add_argument("--eval_only", action="store_true",
                   help="Only evaluate (images must already exist)")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.output_dir)

    log.info(f"Subsets: {args.subsets}")
    log.info(f"Output dir: {os.path.abspath(args.output_dir)}")

    if not args.eval_only:
        log.info("═══  Phase 1: Generate GeoBind images  ═══")
        try:
            generate_images(args)
        except Exception:
            log.error(f"Generation FAILED:\n{traceback.format_exc()}")

    if not args.skip_eval:
        log.info("═══  Phase 2: BLIP-VQA Evaluation  ═══")
        results = evaluate_images(args)
        print_geobind_scores(results)
        log.info(f"Results → {_results_path(args.output_dir)}")
    else:
        log.info("Evaluation skipped (--skip_eval).")


if __name__ == "__main__":
    main()
