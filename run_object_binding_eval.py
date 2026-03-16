#!/usr/bin/env python
"""
Object Binding Benchmark — GPT-4o Evaluation
=============================================
Evaluates generated images using GPT-4o multimodal scoring,
following the paper's evaluation protocol (Appendix C.5, Figure 10).

Scoring rubric (9 levels, 0–100):
  100   — Both subjects only possess their own attributes
  87.5  — Both possess theirs, but one also has the other's
  75    — Both possess their own AND each other's
  62.5  — One correct, the other only has the wrong attributes
  50    — One correct, the other has nothing
  37.5  — Neither has its own, but one has the other's
  25    — Neither has any attributes
  12.5  — One subject missing
  0     — Both subjects missing

Final GPT-4o score = mean of all per-image scores (normalized to 0–1).

Usage
-----
    python run_object_binding_eval.py --output_dir eval_results_object_binding
    python run_object_binding_eval.py --methods ToMe SDXL --api_key sk-xxx
    OPENAI_API_KEY=sk-xxx python run_object_binding_eval.py
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

METHODS = ["SDXL", "ToMe", "GeoBind_v2"]

VALID_SCORES = {0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100}

GPT4O_SCORING_PROMPT = """\
Prompt: {prompt}

Based on our picture and prompt, give the score of the picture below. \
The subjects are the two subjects of the prompt words, and the attributes \
are the adjectives or nouns corresponding to the subjects in the prompt body.

The first line of the answer contains only the rating, and then the \
explanation is given starting from the second line.

The scoring criteria are as follows:
100: Both subjects only possess their own attributes, not the attributes of the other subject.
87.5: Both subjects possesses their attributes. But only one subject that possesses the attributes of another subject.
75: When two subjects possess their own attributes, they both possess the attributes of the other subject.
62.5: One subject possesses attributes of its own, without attributes of the another subject. The other subject only possesses attributes of another subject.
50: One subject possesses attributes of its own. The other subject do not possesses attributes of itself or the other party.
37.5: Both subjects not possess its own attributes. But exist one subject has the attributes of the other party.
25: Neither subject has attributes of itself or the other party.
12.5: Missing one subject
0: Missing two subject"""


# ═══════════════════════════════════════════════════════════════
#  Logging
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

    logger = logging.getLogger("objbind_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info(f"Logging to {log_path}")
    return logger


log = logging.getLogger("objbind_eval")


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Object Binding Benchmark — GPT-4o Evaluation")
    p.add_argument("--methods", nargs="+", default=["ToMe", "GeoBind_v2"],
                   choices=METHODS, help="Methods to evaluate")
    p.add_argument("--output_dir", default="eval_results_object_binding",
                   help="Root output dir (must contain prompts.json and method subdirs)")
    p.add_argument("--api_key", default=None,
                   help="OpenAI API key (or set OPENAI_API_KEY env var)")
    p.add_argument("--gpt_model", default="gpt-4o",
                   help="GPT model for evaluation")
    p.add_argument("--max_retries", type=int, default=3,
                   help="Max retries per image on API failure")
    p.add_argument("--request_delay", type=float, default=1.0,
                   help="Delay between API calls in seconds")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
#  Image encoding
# ═══════════════════════════════════════════════════════════════

def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ═══════════════════════════════════════════════════════════════
#  GPT-4o scoring
# ═══════════════════════════════════════════════════════════════

def parse_score_from_response(text):
    """Extract the numeric score from the first line of GPT-4o's response."""
    first_line = text.strip().split("\n")[0].strip()
    # Remove trailing punctuation or "points"
    first_line = first_line.rstrip(".").strip()
    for suffix in ["points", "point", "pts"]:
        if first_line.lower().endswith(suffix):
            first_line = first_line[:-len(suffix)].strip()

    try:
        score = float(first_line)
    except ValueError:
        # Try to find any number in the first line
        import re
        numbers = re.findall(r"[\d.]+", first_line)
        if numbers:
            score = float(numbers[0])
        else:
            raise ValueError(f"Cannot parse score from: '{first_line}'")

    if score not in VALID_SCORES:
        closest = min(VALID_SCORES, key=lambda v: abs(v - score))
        log.warning(f"Score {score} not in valid set, rounding to {closest}")
        score = closest

    return score


def score_single_image(client, prompt, image_path, gpt_model="gpt-4o",
                       max_retries=3):
    """Score a single image with GPT-4o and return (normalized_score, explanation)."""
    base64_img = encode_image_base64(image_path)
    scoring_text = GPT4O_SCORING_PROMPT.format(prompt=prompt)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=gpt_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": scoring_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_img}",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=500,
            )
            text = response.choices[0].message.content.strip()
            score = parse_score_from_response(text)
            return score / 100.0, text

        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                log.warning(f"  Attempt {attempt+1} failed: {e}. "
                            f"Retrying in {wait}s …")
                time.sleep(wait)
            else:
                log.error(f"  All {max_retries} attempts failed: {e}")
                raise


# ═══════════════════════════════════════════════════════════════
#  Evaluation loop
# ═══════════════════════════════════════════════════════════════

def load_existing_eval(eval_path):
    if os.path.isfile(eval_path):
        with open(eval_path) as f:
            return json.load(f)
    return None


def evaluate_method(args, method, client):
    """Evaluate all images for a single method. Returns average GPT-4o score."""
    prompts_path = os.path.join(args.output_dir, "prompts.json")
    if not os.path.isfile(prompts_path):
        log.error(f"prompts.json not found at {prompts_path}. "
                  "Run run_object_binding_generate.py first.")
        return None

    with open(prompts_path) as f:
        prompts_data = json.load(f)

    image_dir = os.path.join(args.output_dir, method, "samples")
    if not os.path.isdir(image_dir):
        log.error(f"Image directory not found: {image_dir}")
        return None

    eval_path = os.path.join(args.output_dir, method, "gpt4o_eval.json")

    # Load existing partial results for resume support
    existing = load_existing_eval(eval_path)
    if existing and "per_image" in existing:
        scored_indices = {
            r["index"] for r in existing["per_image"] if r.get("score") is not None
        }
        per_image = existing["per_image"]
    else:
        scored_indices = set()
        per_image = []

    n_scored, n_errors = 0, 0
    for entry in prompts_data:
        idx = entry["index"]
        prompt = entry["prompt"]

        if idx in scored_indices:
            continue

        img_path = os.path.join(image_dir, f"{idx:04d}.png")
        if not os.path.isfile(img_path):
            log.warning(f"  Image missing: {img_path}")
            per_image.append({
                "index": idx, "prompt": prompt,
                "score": None, "note": "image_missing",
            })
            n_errors += 1
            continue

        try:
            score, explanation = score_single_image(
                client, prompt, img_path,
                gpt_model=args.gpt_model,
                max_retries=args.max_retries,
            )
            per_image.append({
                "index": idx, "prompt": prompt,
                "score": round(score, 4),
                "raw_score": round(score * 100, 1),
                "explanation": explanation,
            })
            n_scored += 1
            log.info(f"  [{idx:02d}] score={score:.4f}  prompt='{prompt}'")

        except Exception:
            log.error(f"  Scoring failed for [{idx}] '{prompt}':\n"
                      f"{traceback.format_exc()}")
            per_image.append({
                "index": idx, "prompt": prompt,
                "score": None, "note": "scoring_failed",
            })
            n_errors += 1

        # Checkpoint after each image
        valid_scores = [r["score"] for r in per_image if r.get("score") is not None]
        partial_avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0
        checkpoint = {
            "method": method,
            "gpt_model": args.gpt_model,
            "n_scored": len(valid_scores),
            "n_total": len(prompts_data),
            "gpt4o_score": round(partial_avg, 4),
            "per_image": per_image,
        }
        os.makedirs(os.path.dirname(eval_path), exist_ok=True)
        with open(eval_path, "w") as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)

        time.sleep(args.request_delay)

    valid_scores = [r["score"] for r in per_image if r.get("score") is not None]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

    final = {
        "method": method,
        "gpt_model": args.gpt_model,
        "n_scored": len(valid_scores),
        "n_errors": n_errors,
        "n_total": len(prompts_data),
        "gpt4o_score": round(avg_score, 4),
        "per_image": per_image,
    }
    with open(eval_path, "w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    log.info(f"  Detail saved → {eval_path}")

    return avg_score


def print_result_table(results):
    log.info("")
    log.info("=" * 50)
    log.info("  Object Binding Benchmark — GPT-4o Scores")
    log.info("=" * 50)
    log.info(f"  {'Method':<16} {'GPT-4o Score':<16}")
    log.info("  " + "-" * 40)
    for method, score in results.items():
        if score is not None:
            log.info(f"  {method:<16} {score:<16.4f}")
        else:
            log.info(f"  {method:<16} {'N/A':<16}")
    log.info("=" * 50)


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    setup_logging(args.output_dir)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.error("OpenAI API key required. Provide via --api_key or "
                  "OPENAI_API_KEY environment variable.")
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    log.info("=== Object Binding Benchmark — GPT-4o Evaluation ===")
    log.info(f"Methods: {args.methods}  |  Model: {args.gpt_model}")
    log.info(f"Output dir: {os.path.abspath(args.output_dir)}")

    results = {}
    for method in args.methods:
        log.info(f"{'─'*50}")
        log.info(f"  Evaluating: {method}")
        log.info(f"{'─'*50}")
        score = evaluate_method(args, method, client)
        results[method] = score
        if score is not None:
            log.info(f"  ➜ GPT-4o score [{method}] = {score:.4f}")

    print_result_table(results)

    summary_path = os.path.join(args.output_dir, "gpt4o_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
