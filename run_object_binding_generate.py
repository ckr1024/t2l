#!/usr/bin/env python
"""
Object Binding Benchmark — Image Generation
=============================================
Generates images for the GPT-4o object binding benchmark (50 prompts).
Supports ToMe, GeoBind_v2 and SDXL baseline.

Key differences from attribute binding (run_compbench_eval_v2.py / run_geobind_v2.py):
  - Manual prompt parsing (no spaCy), since object binding follows a fixed template
  - eot_replace_step=15  (earlier ETS, Config1-style)
  - use_pose_loss=True   (widens distance between subjects)
  - attention_refinement_steps=[6, 6]

Method-specific parameters:
  ToMe:       tome_control_steps=[7,7],   token_refinement=3, merge_weights=1.0
  GeoBind_v2: tome_control_steps=[10,12], token_refinement=4, hyper_weight=0.15

Usage
-----
    python run_object_binding_generate.py
    python run_object_binding_generate.py --methods ToMe GeoBind_v2 SDXL
    python run_object_binding_generate.py --seed 42 --output_dir obj_bind_results
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

from utils.ptp_utils import AttentionStore, register_attention_control

METHODS = ["SDXL", "ToMe", "GeoBind_v2"]

_PIPELINE_FOR_METHOD = {
    "SDXL": "tome",
    "ToMe": "tome",
    "GeoBind_v2": "geobind",
}

# 50 object binding prompts following the paper's GPT-4o benchmark template:
#   "a [objectA] with [itemA] and a [objectB] with [itemB]"
OBJECT_BINDING_BENCHMARK = [
    {"objectA": "cat", "itemA": "sunglasses", "objectB": "dog", "itemB": "hat"},
    {"objectA": "dog", "itemA": "scarf", "objectB": "cat", "itemB": "hat"},
    {"objectA": "boy", "itemA": "glasses", "objectB": "girl", "itemB": "earrings"},
    {"objectA": "man", "itemA": "hat", "objectB": "woman", "itemB": "necklace"},
    {"objectA": "cat", "itemA": "scarf", "objectB": "dog", "itemB": "tie"},
    {"objectA": "fox", "itemA": "sunglasses", "objectB": "deer", "itemB": "crown"},
    {"objectA": "bear", "itemA": "hat", "objectB": "man", "itemB": "glasses"},
    {"objectA": "tiger", "itemA": "glasses", "objectB": "dog", "itemB": "hat"},
    {"objectA": "boy", "itemA": "hat", "objectB": "corgi", "itemB": "sunglasses"},
    {"objectA": "lion", "itemA": "crown", "objectB": "sheep", "itemB": "scarf"},
    {"objectA": "cat", "itemA": "hat", "objectB": "rabbit", "itemB": "glasses"},
    {"objectA": "man", "itemA": "watch", "objectB": "woman", "itemB": "earrings"},
    {"objectA": "owl", "itemA": "glasses", "objectB": "cat", "itemB": "hat"},
    {"objectA": "monkey", "itemA": "hat", "objectB": "elephant", "itemB": "glasses"},
    {"objectA": "girl", "itemA": "necklace", "objectB": "boy", "itemB": "watch"},
    {"objectA": "penguin", "itemA": "scarf", "objectB": "panda", "itemB": "hat"},
    {"objectA": "horse", "itemA": "mask", "objectB": "dog", "itemB": "cape"},
    {"objectA": "wolf", "itemA": "crown", "objectB": "fox", "itemB": "scarf"},
    {"objectA": "man", "itemA": "tie", "objectB": "woman", "itemB": "hat"},
    {"objectA": "parrot", "itemA": "crown", "objectB": "owl", "itemB": "glasses"},
    {"objectA": "cat", "itemA": "cape", "objectB": "dog", "itemB": "scarf"},
    {"objectA": "bear", "itemA": "sunglasses", "objectB": "deer", "itemB": "hat"},
    {"objectA": "girl", "itemA": "earrings", "objectB": "boy", "itemB": "glasses"},
    {"objectA": "tiger", "itemA": "crown", "objectB": "lion", "itemB": "glasses"},
    {"objectA": "rabbit", "itemA": "hat", "objectB": "squirrel", "itemB": "scarf"},
    {"objectA": "woman", "itemA": "glasses", "objectB": "man", "itemB": "scarf"},
    {"objectA": "dog", "itemA": "tie", "objectB": "cat", "itemB": "hat"},
    {"objectA": "elephant", "itemA": "hat", "objectB": "monkey", "itemB": "sunglasses"},
    {"objectA": "fox", "itemA": "glasses", "objectB": "wolf", "itemB": "hat"},
    {"objectA": "boy", "itemA": "cape", "objectB": "girl", "itemB": "crown"},
    {"objectA": "duck", "itemA": "hat", "objectB": "penguin", "itemB": "scarf"},
    {"objectA": "cat", "itemA": "crown", "objectB": "dog", "itemB": "necklace"},
    {"objectA": "man", "itemA": "helmet", "objectB": "woman", "itemB": "glasses"},
    {"objectA": "horse", "itemA": "hat", "objectB": "dog", "itemB": "vest"},
    {"objectA": "panda", "itemA": "sunglasses", "objectB": "bear", "itemB": "hat"},
    {"objectA": "girl", "itemA": "hat", "objectB": "boy", "itemB": "scarf"},
    {"objectA": "lion", "itemA": "hat", "objectB": "tiger", "itemB": "scarf"},
    {"objectA": "cat", "itemA": "glasses", "objectB": "rabbit", "itemB": "hat"},
    {"objectA": "woman", "itemA": "necklace", "objectB": "girl", "itemB": "earrings"},
    {"objectA": "owl", "itemA": "hat", "objectB": "parrot", "itemB": "scarf"},
    {"objectA": "man", "itemA": "sunglasses", "objectB": "boy", "itemB": "hat"},
    {"objectA": "dog", "itemA": "crown", "objectB": "cat", "itemB": "cape"},
    {"objectA": "deer", "itemA": "glasses", "objectB": "fox", "itemB": "hat"},
    {"objectA": "bear", "itemA": "scarf", "objectB": "wolf", "itemB": "glasses"},
    {"objectA": "squirrel", "itemA": "hat", "objectB": "monkey", "itemB": "glasses"},
    {"objectA": "woman", "itemA": "hat", "objectB": "man", "itemB": "tie"},
    {"objectA": "corgi", "itemA": "hat", "objectB": "cat", "itemB": "sunglasses"},
    {"objectA": "tiger", "itemA": "hat", "objectB": "bear", "itemB": "crown"},
    {"objectA": "girl", "itemA": "glasses", "objectB": "boy", "itemB": "tie"},
    {"objectA": "elephant", "itemA": "crown", "objectB": "horse", "itemB": "hat"},
]


# ═══════════════════════════════════════════════════════════════
#  Logging
# ═══════════════════════════════════════════════════════════════

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"generate_log_{ts}.txt")

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger = logging.getLogger("objbind")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info(f"Logging to {log_path}")
    return logger


log = logging.getLogger("objbind")


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Object Binding Benchmark — Image Generation")
    p.add_argument("--methods", nargs="+", default=["ToMe"],
                   choices=METHODS, help="Methods to generate images for")
    p.add_argument("--model_path", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--output_dir", default="eval_results_object_binding")
    p.add_argument("--n_inference_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--eot_replace_step", type=int, default=15)
    p.add_argument("--hyper_weight", type=float, default=0.15,
                   help="Hyperbolic geometry contribution weight for GeoBind_v2")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
#  Prompt helpers
# ═══════════════════════════════════════════════════════════════

def build_prompt(entry):
    """Construct the text prompt from a benchmark entry."""
    return (f"a {entry['objectA']} with {entry['itemA']} "
            f"and a {entry['objectB']} with {entry['itemB']}")


def parse_object_binding_prompt(prompt, tokenizer):
    """Parse an object binding prompt for token merging.

    Expected format: "a {obj1} with {item1} and a {obj2} with {item2}"

    Unlike attribute binding which uses spaCy for automatic NP extraction,
    object binding uses a fixed template and manual index computation.
    This mirrors RunConfig1's use_nlp=False approach.

    Returns (token_indices, prompt_anchor, prompt_merged, prompt_length).
    """
    words = prompt.split()
    and_word_idx = words.index("and")

    # Map each word to its CLIP token positions (1-based, since 0 = SOT)
    pos = 1
    word_positions = []
    for word in words:
        ids = tokenizer.encode(word)
        n_tokens = len(ids) - 2  # exclude SOT and EOT
        word_positions.append(list(range(pos, pos + n_tokens)))
        pos += n_tokens

    prompt_length = pos - 1

    # Verify against full-prompt tokenization
    full_ids = tokenizer.encode(prompt)
    expected_length = len(full_ids) - 2
    if prompt_length != expected_length:
        log.warning(
            f"Token count mismatch for '{prompt}': "
            f"word-by-word={prompt_length}, full={expected_length}. "
            f"Using full tokenization count."
        )
        prompt_length = expected_length

    # Entity 1: "a OBJ with ITEM" — noun = word[1], attrs = words[2..and)
    noun1_positions = word_positions[1]
    attr1_positions = []
    for i in range(2, and_word_idx):
        attr1_positions.extend(word_positions[i])

    # Entity 2: "a OBJ with ITEM" — noun = word[and+2], attrs = words[and+3..)
    noun2_positions = word_positions[and_word_idx + 2]
    attr2_positions = []
    for i in range(and_word_idx + 3, len(words)):
        attr2_positions.extend(word_positions[i])

    token_indices = [
        [noun1_positions, attr1_positions],
        [noun2_positions, attr2_positions],
    ]

    anchor1 = " ".join(words[:and_word_idx])
    anchor2 = " ".join(words[and_word_idx + 1:])
    prompt_anchor = [anchor1, anchor2]

    noun1_text = words[1]
    noun2_text = words[and_word_idx + 2]
    prompt_merged = f"a {noun1_text} and a {noun2_text}"

    return token_indices, prompt_anchor, prompt_merged, prompt_length


# ═══════════════════════════════════════════════════════════════
#  Pipeline loading
# ═══════════════════════════════════════════════════════════════

def _load_pipeline(pipe_type, model_path, device):
    """Load pipeline by type. Returns (pipeline, extras_dict)."""
    extras = {}
    if pipe_type == "tome":
        from pipe_tome import tomePipeline
        pipe = tomePipeline.from_pretrained(
            model_path, torch_dtype=torch.float16, variant="fp16",
            safety_checker=None,
        ).to(device)
    elif pipe_type == "geobind":
        from pipe_geobind import geobindPipeline, TokenMergerWithAttnHyperspace
        pipe = geobindPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16, variant="fp16",
            safety_checker=None,
        ).to(device)
        extras["TokenMergerClass"] = TokenMergerWithAttnHyperspace
    else:
        raise ValueError(f"Unknown pipe_type: {pipe_type}")

    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    return pipe, extras


def _build_call_kwargs(method, prompt, args, ti, pa, merged, pl,
                       controller, thresholds, extras=None):
    """Build pipeline call kwargs, adapting for method-specific parameters."""
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
        eot_replace_step=args.eot_replace_step,
        use_pose_loss=True,
        attention_refinement_steps=[6, 6],
        negative_prompt="low res, ugly, blurry, artifact, unreal",
    )

    if method in ("SDXL", "ToMe"):
        # ToMe / SDXL — use tomePipeline kwargs
        base.update(
            token_refinement_steps=3,
            tome_control_steps=[7, 7],
            merge_noun_weight=1.0,
            merge_attr_weight=1.0,
            use_hyperbolic=False,
            hyper_merger=None,
        )
    elif method == "GeoBind_v2":
        # GeoBind v2 — more aggressive control, hyperbolic geometry merger
        base.update(
            token_refinement_steps=4,
            tome_control_steps=[10, 12],
            hyper_merger=extras.get("hyper_merger") if extras else None,
        )

    return base


# ═══════════════════════════════════════════════════════════════
#  Image Generation
# ═══════════════════════════════════════════════════════════════

def generate_all_images(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Object binding thresholds — from RunConfig1
    thresholds = {
        0: 26, 1: 25, 2: 24, 3: 23, 4: 22.5,
        5: 22, 6: 21.5, 7: 21, 8: 21, 9: 21,
    }

    # Save prompts metadata for the evaluation script
    prompts_data = []
    for idx, entry in enumerate(OBJECT_BINDING_BENCHMARK):
        prompt = build_prompt(entry)
        prompts_data.append({"index": idx, "prompt": prompt, **entry})

    prompts_path = os.path.join(args.output_dir, "prompts.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(prompts_path, "w") as f:
        json.dump(prompts_data, f, indent=2, ensure_ascii=False)
    log.info(f"Saved {len(prompts_data)} prompts → {prompts_path}")

    # Group methods by pipeline type to avoid redundant model loading
    pipe_groups = {}
    for m in args.methods:
        pt = _PIPELINE_FOR_METHOD.get(m, "tome")
        pipe_groups.setdefault(pt, []).append(m)

    for pipe_type, methods_in_group in pipe_groups.items():
        log.info(f"Loading '{pipe_type}' pipeline for methods {methods_in_group} …")
        pipeline, extras = _load_pipeline(pipe_type, args.model_path, device)

        # For GeoBind_v2, create the hyperbolic merger
        if pipe_type == "geobind" and "TokenMergerClass" in extras:
            hyper_merger = extras["TokenMergerClass"](
                embed_dim=2048, num_heads=8, hyper_weight=args.hyper_weight,
            ).to(device).eval()
            extras["hyper_merger"] = hyper_merger
            log.info(f"  TokenMergerWithAttnHyperspace ready "
                     f"(hyper_weight={args.hyper_weight})")

        for method in methods_in_group:
            out_dir = os.path.join(args.output_dir, method, "samples")
            os.makedirs(out_dir, exist_ok=True)

            existing = len([f for f in os.listdir(out_dir) if f.endswith(".png")])
            if existing >= len(OBJECT_BINDING_BENCHMARK):
                log.info(f"[{method}] {existing} images exist — skipping.")
                continue
            log.info(f"[{method}] generating "
                     f"({existing}/{len(OBJECT_BINDING_BENCHMARK)} done) …")

            n_generated, n_errors = 0, 0
            for idx, entry in enumerate(
                tqdm(OBJECT_BINDING_BENCHMARK, desc=f"  {method}")
            ):
                img_path = os.path.join(out_dir, f"{idx:04d}.png")
                if os.path.exists(img_path):
                    continue

                prompt = build_prompt(entry)
                g = torch.Generator(device).manual_seed(args.seed)

                try:
                    ti, pa, merged, pl = parse_object_binding_prompt(
                        prompt, pipeline.tokenizer
                    )
                except Exception:
                    log.error(f"  Parse error for '{prompt}':\n"
                              f"{traceback.format_exc()}")
                    ti, pa, merged, pl = [], [], prompt, 0

                controller = AttentionStore()
                register_attention_control(pipeline, controller)

                kw = _build_call_kwargs(
                    method, prompt, args, ti, pa, merged, pl,
                    controller, thresholds, extras=extras,
                )
                kw["generator"] = g

                try:
                    out = pipeline(**kw)
                    out.images[0].save(img_path)
                    n_generated += 1
                except Exception as e:
                    tqdm.write(f"    [ERROR] '{prompt}': {e}")
                    log.error(f"  Generation error [{method}] "
                              f"'{prompt}': {e}\n{traceback.format_exc()}")
                    Image.new("RGB", (1024, 1024), "gray").save(img_path)
                    n_errors += 1

            log.info(f"[{method}] done — generated={n_generated}, errors={n_errors}")

        log.info(f"Unloading '{pipe_type}' pipeline …")
        del pipeline
        if "hyper_merger" in extras:
            del extras["hyper_merger"]
        torch.cuda.empty_cache()

    log.info("Generation phase complete.")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    setup_logging(args.output_dir)

    log.info("=== Object Binding Benchmark — Image Generation ===")
    log.info(f"Methods: {args.methods}  |  Seed: {args.seed}")
    log.info(f"eot_replace_step={args.eot_replace_step}  |  use_pose_loss=True")
    log.info(f"attention_refinement=[6,6]  |  n_inference_steps={args.n_inference_steps}")
    for m in args.methods:
        if m == "ToMe":
            log.info(f"  {m}: tome_control=[7,7] token_refine=3 merge=1.0/1.0")
        elif m == "GeoBind_v2":
            log.info(f"  {m}: tome_control=[10,12] token_refine=4 "
                     f"hyper_weight={args.hyper_weight}")
        elif m == "SDXL":
            log.info(f"  {m}: standard SD baseline (no ToMe)")
    log.info(f"Output dir: {os.path.abspath(args.output_dir)}")
    log.info(f"Benchmark size: {len(OBJECT_BINDING_BENCHMARK)} prompts")

    try:
        generate_all_images(args)
    except Exception:
        log.error(f"Generation FAILED:\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
