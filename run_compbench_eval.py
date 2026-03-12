#!/usr/bin/env python
"""
T2I-CompBench Evaluation
========================
Generate images with SDXL / ToMe / GeoBind and compute BLIP-VQA scores
on the color, shape, and texture attribute-binding validation subsets.

Usage
-----
    # Full pipeline (generate + evaluate)
    python run_compbench_eval.py --output_dir eval_results_v1

    # Generate only
    python run_compbench_eval.py --phase generate --output_dir eval_results_v1

    # Evaluate only
    python run_compbench_eval.py --phase evaluate --output_dir eval_results_v1

    # Specific methods / subsets
    python run_compbench_eval.py --methods GeoBind --subsets color

Methods
-------
  SDXL     – Standard Stable Diffusion XL baseline (no token merging)
  ToMe     – Original ToMe paper: Euclidean token merging + MSE binding loss
  GeoBind  – Hyperbolic token merging + hyperbolic contrastive binding loss
             + semantic alignment (pipe_geobind.py)
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

from utils.ptp_utils import AttentionStore, register_attention_control
from prompt_utils import PromptParser
from transformers import BlipProcessor, BlipForQuestionAnswering

SUBSETS = ["color", "shape", "texture"]
METHODS = ["SDXL", "ToMe", "GeoBind"]

# Which pipeline class each method uses
_PIPELINE_FOR_METHOD = {
    "SDXL": "tome",
    "ToMe": "tome",
    "GeoBind": "geobind",
}


# ═══════════════════════════════════════════════════════════════
#  Logging — tee to file + terminal
# ═══════════════════════════════════════════════════════════════

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"eval_log_{ts}.txt")

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger = logging.getLogger("compbench")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info(f"Logging to {log_path}")
    return logger


log = logging.getLogger("compbench")


# ═══════════════════════════════════════════════════════════════
#  Incremental result persistence
# ═══════════════════════════════════════════════════════════════

def _results_path(output_dir):
    return os.path.join(output_dir, "blip_vqa_results.json")


def load_existing_results(output_dir):
    p = _results_path(output_dir)
    if os.path.isfile(p):
        with open(p) as f:
            return json.load(f)
    return {}


def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    p = _results_path(output_dir)
    with open(p, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results checkpoint saved → {p}")


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="T2I-CompBench BLIP-VQA evaluation")
    p.add_argument("--phase", choices=["generate", "evaluate", "all"], default="all")
    p.add_argument("--subsets", nargs="+", default=SUBSETS,
                   help="Attribute subsets to evaluate")
    p.add_argument("--methods", nargs="+", default=METHODS,
                   help="Methods to evaluate")
    p.add_argument("--model_path", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--blip_model", default="Salesforce/blip-vqa-base",
                   help="HuggingFace BLIP-VQA model (T2I-CompBench uses base)")
    p.add_argument("--output_dir", default="eval_results")
    p.add_argument("--data_dir", default="data/t2i_compbench")
    p.add_argument("--n_inference_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--np_num", type=int, default=8,
                   help="Max noun-phrase slots (>= actual NP count in any prompt)")
    p.add_argument("--guidance_scale", type=float, default=7.5)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
#  Data helpers
# ═══════════════════════════════════════════════════════════════

def load_prompts(data_dir, subset):
    path = os.path.join(data_dir, f"{subset}_val.txt")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


# ═══════════════════════════════════════════════════════════════
#  Prompt parsing
# ═══════════════════════════════════════════════════════════════

def generate_merged_prompt(prompt, doc):
    """Remove attribute modifiers, keep sentence structure with bare nouns."""
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
    return filtered_idx, filtered_anchor, merged, prompt_length


# ═══════════════════════════════════════════════════════════════
#  Pipeline loading helpers
# ═══════════════════════════════════════════════════════════════

def _load_pipeline(pipe_type, model_path, device):
    """Load a pipeline by type ('tome' or 'geobind')."""
    if pipe_type == "tome":
        from pipe_tome import tomePipeline
        pipe = tomePipeline.from_pretrained(
            model_path, torch_dtype=torch.float16, variant="fp16",
            safety_checker=None,
        ).to(device)
    elif pipe_type == "geobind":
        from pipe_geobind import geobindPipeline
        pipe = geobindPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16, variant="fp16",
            safety_checker=None,
        ).to(device)
    else:
        raise ValueError(f"Unknown pipe_type: {pipe_type}")

    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    return pipe


def _build_call_kwargs(method, prompt, args, ti, pa, merged, pl, controller, thresholds):
    """Build the keyword-argument dict for the pipeline __call__."""
    run_standard = (method == "SDXL") or (not ti)

    base = dict(
        prompt=prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.n_inference_steps,
        attention_store=controller,
        indices_to_alter=ti,
        prompt_anchor=pa,
        attention_res=32,
        run_standard_sd=run_standard,
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
    )

    if method == "ToMe":
        base["use_hyperbolic"] = False
        base["hyper_merger"] = None

    return base


# ═══════════════════════════════════════════════════════════════
#  Phase 1 — Image generation
# ═══════════════════════════════════════════════════════════════

def generate_all_images(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("Loading spaCy + PromptParser …")
    nlp = spacy.load("en_core_web_trf")
    prompt_parser = PromptParser(args.model_path)
    thresholds = {i: max(26 - i * 0.5, 21) for i in range(20)}

    # Group requested methods by which pipeline they need
    pipe_groups = {}
    for m in args.methods:
        pt = _PIPELINE_FOR_METHOD.get(m, "tome")
        pipe_groups.setdefault(pt, []).append(m)

    for pipe_type, methods_in_group in pipe_groups.items():
        log.info(f"Loading pipeline '{pipe_type}' for methods {methods_in_group} …")
        pipeline = _load_pipeline(pipe_type, args.model_path, device)

        for subset in args.subsets:
            prompts = load_prompts(args.data_dir, subset)
            log.info(f"{'═'*55}")
            log.info(f"  Subset: {subset}  ({len(prompts)} prompts)")
            log.info(f"{'═'*55}")

            for method in methods_in_group:
                out_dir = os.path.join(args.output_dir, method, subset, "samples")
                os.makedirs(out_dir, exist_ok=True)

                existing = len([f for f in os.listdir(out_dir) if f.endswith(".png")])
                if existing >= len(prompts):
                    log.info(f"  [{method}] {existing} images exist — skipping.")
                    continue
                log.info(f"  [{method}] generating ({existing}/{len(prompts)} done) …")

                n_generated, n_errors = 0, 0
                for idx, prompt in enumerate(tqdm(prompts, desc=f"  {method}")):
                    img_path = os.path.join(out_dir, f"{prompt}_{idx}.png")
                    if os.path.exists(img_path):
                        continue

                    g = torch.Generator(device).manual_seed(args.seed)

                    try:
                        ti, pa, merged, pl = parse_prompt_for_tome(
                            prompt, nlp, prompt_parser, pipeline.tokenizer
                        )
                    except Exception:
                        ti, pa, merged, pl = [], [], prompt, 0

                    controller = AttentionStore()
                    register_attention_control(pipeline, controller)

                    kw = _build_call_kwargs(
                        method, prompt, args, ti, pa, merged, pl,
                        controller, thresholds,
                    )
                    kw["generator"] = g

                    try:
                        out = pipeline(**kw)
                        out.images[0].save(img_path)
                        n_generated += 1
                    except Exception as e:
                        tqdm.write(f"    [ERROR] '{prompt}': {e}")
                        log.error(f"  Generation error [{method}/{subset}] "
                                  f"'{prompt}': {e}")
                        Image.new("RGB", (1024, 1024), "gray").save(img_path)
                        n_errors += 1

                log.info(f"  [{method}/{subset}] done — "
                         f"generated={n_generated}, errors={n_errors}")

        log.info(f"Unloading pipeline '{pipe_type}' …")
        del pipeline
        torch.cuda.empty_cache()

    log.info("Generation phase complete.")


# ═══════════════════════════════════════════════════════════════
#  Phase 2 — BLIP-VQA evaluation
# ═══════════════════════════════════════════════════════════════

def compute_vqa_yes_prob(model, processor, image, question, device):
    """
    Compute P(yes) as the first decoder-token probability —
    replicates T2I-CompBench's ``inference='vqa_prob'`` protocol.
    """
    inputs = processor(image, question, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=10,
        )
    first_logits = outputs.scores[0]  # (1, vocab_size)
    probs = F.softmax(first_logits, dim=-1)

    yes_ids = processor.tokenizer("yes", add_special_tokens=False)["input_ids"]
    return probs[0, yes_ids[0]].item()


def evaluate_subset_method(images_dir, prompts, model, processor, nlp_sm, device,
                           np_num, detail_save_path=None):
    """
    BLIP-VQA scoring following T2I-CompBench protocol.
    Optionally saves per-image detail to *detail_save_path*.
    """
    n = len(prompts)
    reward = torch.ones((n, np_num), device=device)

    all_nps = []
    for prompt in prompts:
        doc = nlp_sm(prompt)
        nps = [
            chunk.text
            for chunk in doc.noun_chunks
            if chunk.text not in ["top", "the side", "the left", "the right"]
        ]
        all_nps.append(nps)

    per_image_details = []

    for k, prompt in enumerate(tqdm(prompts, desc="    BLIP-VQA")):
        nps = all_nps[k]
        detail = {"prompt": prompt, "index": k, "noun_phrases": nps, "scores": []}

        if not nps:
            per_image_details.append(detail)
            continue

        img_path = os.path.join(images_dir, f"{prompt}_{k}.png")
        if not os.path.exists(img_path):
            for j in range(min(len(nps), np_num)):
                reward[k, j] = 0.0
                detail["scores"].append({"np": nps[j], "score": 0.0, "note": "image_missing"})
            per_image_details.append(detail)
            continue

        image = Image.open(img_path).convert("RGB")
        for j, np_text in enumerate(nps[:np_num]):
            score = compute_vqa_yes_prob(
                model, processor, image, f"{np_text}?", device
            )
            reward[k, j] = score
            detail["scores"].append({"np": np_text, "score": round(score, 6)})

        per_image_details.append(detail)

    max_np = max(len(nps) for nps in all_nps) if all_nps else 0
    for j in range(min(max_np, np_num)):
        slot_scores = [reward[k, j].item() for k in range(n) if len(all_nps[k]) > j]
        if slot_scores:
            log.info(f"    NP slot {j}: mean P(yes) = "
                     f"{sum(slot_scores)/len(slot_scores):.4f}")

    reward_final = reward[:, 0]
    for i in range(1, np_num):
        reward_final = reward_final * reward[:, i]

    final_score = reward_final.mean().item()

    for k in range(n):
        per_image_details[k]["image_score"] = round(reward_final[k].item(), 6)

    if detail_save_path:
        os.makedirs(os.path.dirname(detail_save_path), exist_ok=True)
        with open(detail_save_path, "w") as f:
            json.dump(
                {"blip_vqa_score": round(final_score, 6),
                 "per_image": per_image_details},
                f, indent=2, ensure_ascii=False,
            )
        log.info(f"    Per-image detail saved → {detail_save_path}")

    return final_score


def evaluate_all(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = load_existing_results(args.output_dir)

    log.info(f"Loading BLIP-VQA model: {args.blip_model} …")
    try:
        processor = BlipProcessor.from_pretrained(args.blip_model)
        model = (
            BlipForQuestionAnswering.from_pretrained(args.blip_model)
            .to(device).eval()
        )
    except Exception:
        log.error(f"Failed to load BLIP model:\n{traceback.format_exc()}")
        return results

    try:
        nlp_sm = spacy.load("en_core_web_sm")
    except OSError:
        log.error("spacy model 'en_core_web_sm' not found. "
                  "Install with: python -m spacy download en_core_web_sm")
        return results

    for subset in args.subsets:
        prompts = load_prompts(args.data_dir, subset)
        if subset not in results:
            results[subset] = {}

        for method in args.methods:
            existing_score = results[subset].get(method)
            if existing_score is not None:
                log.info(f"  [{method}/{subset}] already evaluated: "
                         f"{existing_score} — skip")
                continue

            images_dir = os.path.join(args.output_dir, method, subset, "samples")
            if not os.path.isdir(images_dir):
                log.warning(f"  [SKIP] {images_dir} not found")
                results[subset][method] = None
                save_results(results, args.output_dir)
                continue

            log.info(f"{'─'*50}")
            log.info(f"  Evaluating  {method} / {subset}")
            log.info(f"{'─'*50}")

            detail_path = os.path.join(
                args.output_dir, method, subset, "vqa_detail.json"
            )

            try:
                score = evaluate_subset_method(
                    images_dir, prompts, model, processor, nlp_sm, device,
                    args.np_num, detail_save_path=detail_path,
                )
                results[subset][method] = round(score, 4)
                log.info(f"  ➜ BLIP-VQA score = {score:.4f}")
            except Exception:
                log.error(f"  Evaluation FAILED for {method}/{subset}:\n"
                          f"{traceback.format_exc()}")
                results[subset][method] = None

            save_results(results, args.output_dir)

    del model, processor
    torch.cuda.empty_cache()
    return results


# ═══════════════════════════════════════════════════════════════
#  Phase 3 — Report
# ═══════════════════════════════════════════════════════════════

def print_result_table(results):
    subsets = [s for s in SUBSETS if s in results]
    methods_set = set()
    for s in subsets:
        methods_set.update(results[s].keys())
    methods = [m for m in METHODS if m in methods_set]

    log.info("")
    log.info("=" * 62)
    log.info("  T2I-CompBench  BLIP-VQA Scores")
    log.info("=" * 62)
    header = f"  {'Method':<16}"
    for s in subsets:
        header += f"{s.capitalize():<14}"
    log.info(header)
    log.info("  " + "-" * 56)
    for m in methods:
        row = f"  {m:<16}"
        for s in subsets:
            val = results.get(s, {}).get(m)
            row += f"{val:<14.4f}" if val is not None else f"{'N/A':<14}"
        log.info(row)
    log.info("=" * 62)


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    setup_logging(args.output_dir)

    log.info(f"Phase: {args.phase}  |  Subsets: {args.subsets}  "
             f"|  Methods: {args.methods}")
    log.info(f"Output dir: {os.path.abspath(args.output_dir)}")

    if args.phase in ("all", "generate"):
        try:
            generate_all_images(args)
        except Exception:
            log.error(f"Generation phase FAILED:\n{traceback.format_exc()}")

    if args.phase in ("all", "evaluate"):
        results = evaluate_all(args)
        print_result_table(results)
        log.info(f"Final results → {_results_path(args.output_dir)}")


if __name__ == "__main__":
    main()
