#!/usr/bin/env python
"""
BLIP-VQA Standalone Evaluator
==============================
Evaluate pre-generated images using the T2I-CompBench BLIP-VQA protocol.
No generation — only scoring.

Expected directory layout:
    <image_root>/<method>/<subset>/samples/<prompt>_<idx>.png

Usage
-----
    # Auto-detect all methods & subsets under the directory
    python eval_blip_vqa.py --image_root eval_results

    # Specify methods / subsets explicitly
    python eval_blip_vqa.py --image_root eval_results --methods SDXL ToMe --subsets color

    # Force re-evaluate (ignore cached scores)
    python eval_blip_vqa.py --image_root eval_results --force

    # Custom prompt data directory
    python eval_blip_vqa.py --image_root eval_results --data_dir data/t2i_compbench
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

SUBSETS = ["color", "shape", "texture"]

# ─────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"eval_log_{ts}.txt")

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger = logging.getLogger("eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info(f"Logging to {log_path}")
    return logger

log = logging.getLogger("eval")

# ─────────────────────────────────────────────────────────
#  Result persistence
# ─────────────────────────────────────────────────────────

def _results_path(image_root):
    return os.path.join(image_root, "blip_vqa_results.json")

def load_results(image_root):
    p = _results_path(image_root)
    if os.path.isfile(p):
        with open(p) as f:
            return json.load(f)
    return {}

def save_results(results, image_root):
    p = _results_path(image_root)
    with open(p, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved → {p}")

# ─────────────────────────────────────────────────────────
#  Data
# ─────────────────────────────────────────────────────────

def load_prompts(data_dir, subset):
    path = os.path.join(data_dir, f"{subset}_val.txt")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]

# ─────────────────────────────────────────────────────────
#  BLIP-VQA scoring
# ─────────────────────────────────────────────────────────

def compute_vqa_yes_prob(model, processor, image, question, device):
    inputs = processor(image, question, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=10,
        )
    first_logits = outputs.scores[0]
    probs = F.softmax(first_logits, dim=-1)
    yes_ids = processor.tokenizer("yes", add_special_tokens=False)["input_ids"]
    return probs[0, yes_ids[0]].item()


def evaluate_one(images_dir, prompts, model, processor, nlp_sm, device,
                 np_num, detail_save_path=None):
    n = len(prompts)
    reward = torch.ones((n, np_num), device=device)

    all_nps = []
    for prompt in prompts:
        doc = nlp_sm(prompt)
        nps = [
            chunk.text for chunk in doc.noun_chunks
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
                detail["scores"].append({"np": nps[j], "score": 0.0, "note": "missing"})
            per_image_details.append(detail)
            continue

        image = Image.open(img_path).convert("RGB")
        for j, np_text in enumerate(nps[:np_num]):
            score = compute_vqa_yes_prob(model, processor, image, f"{np_text}?", device)
            reward[k, j] = score
            detail["scores"].append({"np": np_text, "score": round(score, 6)})
        per_image_details.append(detail)

    max_np = max(len(nps) for nps in all_nps) if all_nps else 0
    for j in range(min(max_np, np_num)):
        slot_scores = [reward[k, j].item() for k in range(n) if len(all_nps[k]) > j]
        if slot_scores:
            log.info(f"    NP slot {j}: mean P(yes) = {sum(slot_scores)/len(slot_scores):.4f}")

    reward_final = reward[:, 0]
    for i in range(1, np_num):
        reward_final = reward_final * reward[:, i]
    final_score = reward_final.mean().item()

    for k in range(n):
        per_image_details[k]["image_score"] = round(reward_final[k].item(), 6)

    if detail_save_path:
        os.makedirs(os.path.dirname(detail_save_path), exist_ok=True)
        with open(detail_save_path, "w") as f:
            json.dump({"blip_vqa_score": round(final_score, 6),
                       "per_image": per_image_details},
                      f, indent=2, ensure_ascii=False)
        log.info(f"    Detail saved → {detail_save_path}")

    return final_score

# ─────────────────────────────────────────────────────────
#  Auto-detect available methods / subsets
# ─────────────────────────────────────────────────────────

def detect_available(image_root, methods_filter=None, subsets_filter=None):
    """Scan image_root and return [(method, subset)] pairs that have a samples/ dir."""
    pairs = []
    if not os.path.isdir(image_root):
        return pairs
    for method in sorted(os.listdir(image_root)):
        method_dir = os.path.join(image_root, method)
        if not os.path.isdir(method_dir):
            continue
        if methods_filter and method not in methods_filter:
            continue
        for subset in sorted(os.listdir(method_dir)):
            if subsets_filter and subset not in subsets_filter:
                continue
            samples_dir = os.path.join(method_dir, subset, "samples")
            if os.path.isdir(samples_dir):
                n_images = len([f for f in os.listdir(samples_dir) if f.endswith(".png")])
                if n_images > 0:
                    pairs.append((method, subset, n_images))
    return pairs

# ─────────────────────────────────────────────────────────
#  Report
# ─────────────────────────────────────────────────────────

def print_result_table(results):
    subsets = [s for s in SUBSETS if s in results]
    methods_set = set()
    for s in subsets:
        methods_set.update(results[s].keys())
    methods = sorted(methods_set)

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

# ─────────────────────────────────────────────────────────
#  CLI + main
# ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Standalone BLIP-VQA evaluator for T2I-CompBench images")
    p.add_argument("--image_root", required=True,
                   help="Root dir containing <method>/<subset>/samples/")
    p.add_argument("--methods", nargs="+", default=None,
                   help="Methods to evaluate (auto-detect if omitted)")
    p.add_argument("--subsets", nargs="+", default=None,
                   help="Subsets to evaluate (auto-detect if omitted)")
    p.add_argument("--data_dir", default="data/t2i_compbench",
                   help="Directory containing *_val.txt prompt files")
    p.add_argument("--blip_model", default="Salesforce/blip-vqa-base")
    p.add_argument("--np_num", type=int, default=8)
    p.add_argument("--force", action="store_true",
                   help="Re-evaluate even if cached results exist")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.image_root)

    pairs = detect_available(args.image_root, args.methods, args.subsets)
    if not pairs:
        log.error(f"No images found under {args.image_root}. "
                  f"Expected layout: <image_root>/<method>/<subset>/samples/*.png")
        return

    log.info(f"Found {len(pairs)} (method, subset) pairs to evaluate:")
    for method, subset, n in pairs:
        log.info(f"  {method}/{subset}  — {n} images")

    results = {} if args.force else load_results(args.image_root)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading BLIP-VQA: {args.blip_model} on {device} …")
    try:
        processor = BlipProcessor.from_pretrained(args.blip_model)
        model = BlipForQuestionAnswering.from_pretrained(args.blip_model).to(device).eval()
    except Exception:
        log.error(f"Failed to load BLIP:\n{traceback.format_exc()}")
        return

    try:
        nlp_sm = spacy.load("en_core_web_sm")
    except OSError:
        log.error("spacy en_core_web_sm not found. Run: python -m spacy download en_core_web_sm")
        return

    for method, subset, _ in pairs:
        if subset not in results:
            results[subset] = {}
        existing = results[subset].get(method)
        if existing is not None and not args.force:
            log.info(f"  [{method}/{subset}] cached: {existing} — skip (use --force to re-eval)")
            continue

        images_dir = os.path.join(args.image_root, method, subset, "samples")
        prompts = load_prompts(args.data_dir, subset)

        log.info(f"{'─'*50}")
        log.info(f"  Evaluating  {method} / {subset}")
        log.info(f"{'─'*50}")

        detail_path = os.path.join(args.image_root, method, subset, "vqa_detail.json")
        try:
            score = evaluate_one(
                images_dir, prompts, model, processor, nlp_sm, device,
                args.np_num, detail_save_path=detail_path,
            )
            results[subset][method] = round(score, 4)
            log.info(f"  ➜ BLIP-VQA score = {score:.4f}")
        except Exception:
            log.error(f"  FAILED {method}/{subset}:\n{traceback.format_exc()}")
            results[subset][method] = None

        save_results(results, args.image_root)

    del model, processor
    torch.cuda.empty_cache()

    print_result_table(results)
    log.info(f"Final results → {_results_path(args.image_root)}")


if __name__ == "__main__":
    main()
