#!/usr/bin/env python
"""
ImageReward Standalone Evaluator
=================================
Evaluate pre-generated images using the ImageReward model.
No generation — only scoring.

Expected directory layout:
    <image_root>/<method>/<subset>/samples/<prompt>_<idx>.png

Usage
-----
    # Auto-detect all methods & subsets under the directory
    python eval_image_reward.py --image_root eval_results

    # Specify methods / subsets explicitly
    python eval_image_reward.py --image_root eval_results --methods GeoBind --subsets color texture

    # Force re-evaluate (ignore cached scores)
    python eval_image_reward.py --image_root eval_results --force

    # Custom ImageReward checkpoint
    python eval_image_reward.py --image_root eval_results \
        --reward_path /path/to/ImageReward.pt \
        --med_config /path/to/med_config.json
"""

import os
import sys
import json
import logging
import argparse
import traceback
from datetime import datetime

import torch
from PIL import Image
from tqdm import tqdm
import ImageReward as reward

SUBSETS = ["color", "shape", "texture"]

# ─────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"ir_eval_log_{ts}.txt")

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger = logging.getLogger("ir_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info(f"Logging to {log_path}")
    return logger

log = logging.getLogger("ir_eval")

# ─────────────────────────────────────────────────────────
#  Result persistence
# ─────────────────────────────────────────────────────────

def _results_path(image_root):
    return os.path.join(image_root, "image_reward_results.json")

def load_results(image_root):
    p = _results_path(image_root)
    if os.path.isfile(p):
        with open(p) as f:
            return json.load(f)
    return {}

def save_results(results, image_root):
    p = _results_path(image_root)
    with open(p, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info(f"Results saved -> {p}")

# ─────────────────────────────────────────────────────────
#  Data
# ─────────────────────────────────────────────────────────

def load_prompts(data_dir, subset):
    path = os.path.join(data_dir, f"{subset}_val.txt")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]

# ─────────────────────────────────────────────────────────
#  ImageReward scoring
# ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_one(images_dir, prompts, model, detail_save_path=None):
    """Score all images in a (method, subset) pair and return the mean score."""
    scores = []
    per_image_details = []

    for k, prompt in enumerate(tqdm(prompts, desc="    ImageReward")):
        img_path = os.path.join(images_dir, f"{prompt}_{k}.png")
        detail = {"prompt": prompt, "index": k, "image_path": img_path}

        if not os.path.exists(img_path):
            detail["score"] = None
            detail["note"] = "missing"
            per_image_details.append(detail)
            continue

        try:
            score = model.score(prompt, str(img_path))
            detail["score"] = round(float(score), 6)
            scores.append(float(score))
        except Exception as e:
            detail["score"] = None
            detail["note"] = f"error: {e}"

        per_image_details.append(detail)

    mean_score = sum(scores) / len(scores) if scores else 0.0
    n_valid = len(scores)
    n_total = len(prompts)

    if detail_save_path:
        os.makedirs(os.path.dirname(detail_save_path), exist_ok=True)
        with open(detail_save_path, "w") as f:
            json.dump({
                "image_reward_mean": round(mean_score, 6),
                "n_valid": n_valid,
                "n_total": n_total,
                "per_image": per_image_details,
            }, f, indent=2, ensure_ascii=False)
        log.info(f"    Detail saved -> {detail_save_path}")

    return mean_score, n_valid, n_total

# ─────────────────────────────────────────────────────────
#  Auto-detect available methods / subsets
# ─────────────────────────────────────────────────────────

def detect_available(image_root, methods_filter=None, subsets_filter=None):
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
    log.info("  T2I-CompBench  ImageReward Scores")
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
            if val is not None:
                row += f"{val:<14.4f}"
            else:
                row += f"{'N/A':<14}"
        log.info(row)
    log.info("=" * 62)

# ─────────────────────────────────────────────────────────
#  CLI + main
# ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Standalone ImageReward evaluator for T2I-CompBench images")
    p.add_argument("--image_root", required=True,
                   help="Root dir containing <method>/<subset>/samples/")
    p.add_argument("--methods", nargs="+", default=None,
                   help="Methods to evaluate (auto-detect if omitted)")
    p.add_argument("--subsets", nargs="+", default=None,
                   help="Subsets to evaluate (auto-detect if omitted)")
    p.add_argument("--data_dir", default="data/t2i_compbench",
                   help="Directory containing *_val.txt prompt files")
    p.add_argument("--reward_path", default="ImageReward-v1.0",
                   help="ImageReward checkpoint name or path")
    p.add_argument("--med_config", default=None,
                   help="Path to med_config.json (optional, for local checkpoints)")
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
        log.info(f"  {method}/{subset}  -- {n} images")

    results = {} if args.force else load_results(args.image_root)

    log.info(f"Loading ImageReward model: {args.reward_path} ...")
    try:
        load_kwargs = {"name": args.reward_path}
        if args.med_config:
            load_kwargs["med_config"] = args.med_config
        ir_model = reward.load(**load_kwargs)
    except Exception:
        log.error(f"Failed to load ImageReward:\n{traceback.format_exc()}")
        return

    for method, subset, _ in pairs:
        if subset not in results:
            results[subset] = {}
        existing = results[subset].get(method)
        if existing is not None and not args.force:
            log.info(f"  [{method}/{subset}] cached: {existing} -- skip (use --force to re-eval)")
            continue

        images_dir = os.path.join(args.image_root, method, subset, "samples")
        prompts = load_prompts(args.data_dir, subset)

        log.info(f"{'─'*50}")
        log.info(f"  Evaluating  {method} / {subset}")
        log.info(f"{'─'*50}")

        detail_path = os.path.join(args.image_root, method, subset, "ir_detail.json")
        try:
            score, n_valid, n_total = evaluate_one(
                images_dir, prompts, ir_model, detail_save_path=detail_path,
            )
            results[subset][method] = round(score, 4)
            log.info(f"  -> ImageReward score = {score:.4f}  ({n_valid}/{n_total} valid)")
        except Exception:
            log.error(f"  FAILED {method}/{subset}:\n{traceback.format_exc()}")
            results[subset][method] = None

        save_results(results, args.image_root)

    del ir_model
    torch.cuda.empty_cache()

    print_result_table(results)
    log.info(f"Final results -> {_results_path(args.image_root)}")


if __name__ == "__main__":
    main()
