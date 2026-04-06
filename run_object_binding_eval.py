#!/usr/bin/env python
"""
GOB-Bench — Multi-Evaluator Object Binding Evaluation
======================================================
Evaluate generated images using GPT-4o and Qwen2-VL, following the
paper's evaluation protocol with 3 independent judges.

Scoring rubric (0-100):
  100  — All objects present with all sub-objects correctly bound to their owners
   75  — All objects present, 1 sub-object incorrectly assigned
   50  — All objects present, 2+ sub-objects incorrectly assigned
   25  — One or more objects missing
    0  — Image does not match the prompt

Usage:
    python run_object_binding_eval.py --output_dir eval_results_gob_bench
    python run_object_binding_eval.py --evaluator gpt4o --methods GeoBind ToMe SDXL
    python run_object_binding_eval.py --evaluator qwen2vl --qwen_model Qwen/Qwen2-VL-72B-Instruct
"""

import os
import sys
import json
import time
import base64
import logging
import argparse
import traceback
from datetime import datetime

from tqdm import tqdm

METHODS = ["SDXL", "ToMe", "GeoBind"]
LEVELS = ["Easy", "Medium", "Hard", "Complex"]
VALID_SCORES = {0, 25, 50, 75, 100}

SCORING_PROMPT = """\
Evaluate this image against the prompt: "{prompt}"

Score how well the image matches the prompt in terms of object binding \
(whether each object has its correct sub-objects/accessories, and no item \
is assigned to the wrong object).

Scoring criteria:
100: All objects present with all sub-objects correctly bound to their owners.
75:  All objects present, but 1 sub-object is incorrectly assigned to the wrong owner.
50:  All objects present, but 2+ sub-objects are incorrectly assigned.
25:  One or more objects are missing from the image.
0:   Image does not match the prompt at all.

First line: score only (a number from {0, 25, 50, 75, 100}).
Second line onwards: brief explanation."""


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"eval_log_{ts}.txt")
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger = logging.getLogger("gobbench_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


log = logging.getLogger("gobbench_eval")


def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_score(text):
    """Extract numeric score from first line of evaluator response."""
    import re
    first_line = text.strip().split("\n")[0].strip().rstrip(".")
    try:
        score = float(first_line)
    except ValueError:
        numbers = re.findall(r"[\d.]+", first_line)
        if numbers:
            score = float(numbers[0])
        else:
            raise ValueError(f"Cannot parse score from: '{first_line}'")
    closest = min(VALID_SCORES, key=lambda v: abs(v - score))
    return closest


# ═══════════════════════════════════════════════════════════════
#  GPT-4o Evaluator
# ═══════════════════════════════════════════════════════════════

def score_gpt4o(client, prompt, image_path, model="gpt-4o", max_retries=3):
    base64_img = encode_image_base64(image_path)
    scoring_text = SCORING_PROMPT.format(prompt=prompt)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": scoring_text},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{base64_img}"}},
                    ],
                }],
                max_tokens=500,
            )
            text = response.choices[0].message.content.strip()
            return parse_score(text), text
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                raise


# ═══════════════════════════════════════════════════════════════
#  Qwen2-VL Evaluator
# ═══════════════════════════════════════════════════════════════

def score_qwen2vl(model, processor, prompt, image_path, device="cuda"):
    from PIL import Image as PILImage

    image = PILImage.open(image_path).convert("RGB")
    scoring_text = SCORING_PROMPT.format(prompt=prompt)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": scoring_text},
        ],
    }]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text_input], images=[image],
        padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=500)

    input_len = inputs.input_ids.shape[1]
    generated = output_ids[:, input_len:]
    text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return parse_score(text), text


# ═══════════════════════════════════════════════════════════════
#  Evaluation loop
# ═══════════════════════════════════════════════════════════════

def evaluate_method(args, method, evaluator_fn):
    meta_path = os.path.join(args.output_dir, "gob_bench_meta.json")
    if not os.path.isfile(meta_path):
        log.error(f"Metadata not found: {meta_path}. Run generation first.")
        return None

    with open(meta_path) as f:
        prompts_data = json.load(f)

    image_dir = os.path.join(args.output_dir, method, "samples")
    if not os.path.isdir(image_dir):
        log.error(f"Image directory not found: {image_dir}")
        return None

    eval_path = os.path.join(
        args.output_dir, method, f"{args.evaluator}_eval.json")

    existing = {}
    if os.path.isfile(eval_path) and not args.force:
        with open(eval_path) as f:
            data = json.load(f)
        existing = {r["index"]: r for r in data.get("per_image", [])
                    if r.get("score") is not None}

    per_image = list(existing.values())
    scored_indices = set(existing.keys())

    for entry in tqdm(prompts_data, desc=f"  {method}"):
        idx = entry["index"]
        if idx in scored_indices:
            continue

        prompt = entry["prompt"]
        level = entry["level"]
        img_path = os.path.join(image_dir, f"{idx:04d}.png")

        if not os.path.isfile(img_path):
            per_image.append({"index": idx, "prompt": prompt,
                              "level": level, "score": None, "note": "missing"})
            continue

        try:
            score, explanation = evaluator_fn(prompt, img_path)
            per_image.append({
                "index": idx, "prompt": prompt, "level": level,
                "score": score, "explanation": explanation,
            })
        except Exception as e:
            log.error(f"  Scoring failed [{idx}]: {e}")
            per_image.append({"index": idx, "prompt": prompt,
                              "level": level, "score": None, "note": str(e)})

        # Checkpoint
        _save_eval(eval_path, method, args.evaluator, per_image, prompts_data)

    _save_eval(eval_path, method, args.evaluator, per_image, prompts_data)
    return _compute_scores(per_image)


def _save_eval(path, method, evaluator, per_image, prompts_data):
    scores = _compute_scores(per_image)
    result = {
        "method": method, "evaluator": evaluator,
        "n_total": len(prompts_data),
        "scores": scores, "per_image": per_image,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def _compute_scores(per_image):
    valid = [r for r in per_image if r.get("score") is not None]
    if not valid:
        return {}

    overall = sum(r["score"] for r in valid) / len(valid)
    scores = {"overall": round(overall, 2), "n_valid": len(valid)}

    for level in LEVELS:
        level_items = [r for r in valid if r.get("level") == level]
        if level_items:
            avg = sum(r["score"] for r in level_items) / len(level_items)
            scores[level] = round(avg, 2)
            scores[f"{level}_n"] = len(level_items)

    return scores


def print_results(all_results):
    log.info("\n" + "=" * 65)
    log.info("  GOB-Bench Results")
    log.info("=" * 65)
    header = f"  {'Method':<14}" + "".join(f"{l:<10}" for l in LEVELS) + f"{'Avg':<10}"
    log.info(header)
    log.info("  " + "-" * 60)
    for method, scores in all_results.items():
        if scores is None:
            continue
        row = f"  {method:<14}"
        for level in LEVELS:
            val = scores.get(level, "N/A")
            row += f"{val:<10}" if isinstance(val, str) else f"{val:<10.1f}"
        row += f"{scores.get('overall', 'N/A'):<10}"
        log.info(row)
    log.info("=" * 65)


def parse_args():
    p = argparse.ArgumentParser(description="GOB-Bench evaluation")
    p.add_argument("--output_dir", default="eval_results_gob_bench")
    p.add_argument("--methods", nargs="+", default=["GeoBind"],
                   choices=METHODS)
    p.add_argument("--evaluator", default="gpt4o",
                   choices=["gpt4o", "qwen2vl"])
    p.add_argument("--api_key", default=None,
                   help="OpenAI API key (for gpt4o evaluator)")
    p.add_argument("--gpt_model", default="gpt-4o")
    p.add_argument("--qwen_model", default="Qwen/Qwen2-VL-72B-Instruct")
    p.add_argument("--force", action="store_true")
    p.add_argument("--request_delay", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.output_dir)

    log.info(f"=== GOB-Bench Evaluation ({args.evaluator}) ===")
    log.info(f"Methods: {args.methods}")

    if args.evaluator == "gpt4o":
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            log.error("OpenAI API key required (--api_key or OPENAI_API_KEY)")
            sys.exit(1)
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        def evaluator_fn(prompt, image_path):
            result = score_gpt4o(client, prompt, image_path, args.gpt_model)
            time.sleep(args.request_delay)
            return result

    elif args.evaluator == "qwen2vl":
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        log.info(f"Loading Qwen2-VL: {args.qwen_model} ...")
        qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.qwen_model, torch_dtype=torch.float16, device_map="auto")
        qwen_processor = AutoProcessor.from_pretrained(args.qwen_model)

        def evaluator_fn(prompt, image_path):
            return score_qwen2vl(qwen_model, qwen_processor, prompt, image_path)

    all_results = {}
    for method in args.methods:
        log.info(f"\n{'─'*50}")
        log.info(f"  Evaluating: {method}")
        log.info(f"{'─'*50}")
        scores = evaluate_method(args, method, evaluator_fn)
        all_results[method] = scores
        if scores:
            log.info(f"  {method} overall: {scores.get('overall', 'N/A')}")

    print_results(all_results)

    summary_path = os.path.join(args.output_dir, f"{args.evaluator}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
