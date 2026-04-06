#!/usr/bin/env python
"""
GOB-Bench — Object Binding Image Generation
=============================================
Generate images for the GOB-Bench benchmark (200 prompts, 4 difficulty levels).
Object binding: each object has sub-objects/accessories that must be
correctly bound (e.g., "hat" belongs to "dog", not "cat").

Uses template-based parsing (not SpaCy) since object binding prompts
follow "a X with Y" patterns where "with/wearing" creates pobj
dependencies that SpaCy's amod extractor does not capture.

Usage:
    python run_object_binding_generate.py
    python run_object_binding_generate.py --methods GeoBind ToMe SDXL
    python run_object_binding_generate.py --levels Easy Medium --seed 42
"""

import os
import re
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

METHODS = ["SDXL", "ToMe", "GeoBind"]
LEVELS = ["Easy", "Medium", "Hard", "Complex"]
GOB_BENCH_FILE = os.path.join("supplementary", "gob_bench_prompts.txt")

_PIPELINE_FOR_METHOD = {
    "SDXL": "tome",
    "ToMe": "tome",
    "GeoBind": "geobind",
}


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"generate_log_{ts}.txt")
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger = logging.getLogger("gobbench")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


log = logging.getLogger("gobbench")


# ═══════════════════════════════════════════════════════════════
#  GOB-Bench prompt loading
# ═══════════════════════════════════════════════════════════════

def load_gob_bench(filepath, levels=None):
    """Load GOB-Bench prompts from file, optionally filtering by level."""
    prompts = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for level in LEVELS:
                prefix = f"[{level}]"
                if line.startswith(prefix):
                    prompt_text = line[len(prefix):].strip()
                    if levels is None or level in levels:
                        prompts.append({
                            "level": level,
                            "prompt": prompt_text,
                        })
                    break
    return prompts


# ═══════════════════════════════════════════════════════════════
#  Template-based prompt parsing for object binding
# ═══════════════════════════════════════════════════════════════

def _split_into_entities(prompt):
    """Split an object binding prompt into entity segments.

    Only starts a new entity when "a/an X with/wearing" is found,
    so items like "and a scarf" (no "with") stay with the previous entity.

    Handles patterns like:
      "a X with Y and a Z with W"
      "a X with Y and Z, and a Z with W and V"
      "a X with Y, a Z with W, and a V with U"
    """
    prompt_clean = prompt.strip()

    # Find entity anchors: "a/an [noun] with/wearing"
    # Each entity must contain "with" or "wearing" to qualify
    pattern = (
        r'(?:^|[,]?\s+(?:and\s+)?)'
        r'((?:a|an)\s+[\w\s]+?\s+(?:with|wearing)\s+)'
    )
    anchors = list(re.finditer(pattern, prompt_clean, re.IGNORECASE))

    if not anchors:
        return [prompt_clean]

    segments = []
    for i, match in enumerate(anchors):
        # Entity starts at the "a/an" part
        start = match.start(1) if match.group(1) else match.start()
        if i + 1 < len(anchors):
            end = anchors[i + 1].start()
        else:
            end = len(prompt_clean)
        segment = prompt_clean[start:end].strip().rstrip(",").strip()
        if segment:
            segments.append(segment)

    return segments


def _parse_entity_segment(segment):
    """Parse a single entity segment like 'a cat with sunglasses and a scarf'.

    Returns (object_word, [item_words]).
    """
    # Find "with" or "wearing" keyword
    for kw in ["wearing", "with"]:
        kw_idx = segment.lower().find(f" {kw} ")
        if kw_idx >= 0:
            obj_part = segment[:kw_idx].strip()
            items_part = segment[kw_idx + len(kw) + 2:].strip()

            # Remove leading article from object
            obj_word = re.sub(r'^(?:a|an)\s+', '', obj_part).strip()

            # Split items by " and "
            items = [item.strip().lstrip("a ").lstrip("an ")
                     for item in re.split(r'\s+and\s+', items_part)
                     if item.strip()]
            # Clean up articles from items
            clean_items = []
            for item in items:
                item = re.sub(r'^(?:a|an)\s+', '', item).strip()
                if item:
                    clean_items.append(item)

            return obj_word, clean_items

    # Fallback: no "with"/"wearing" found, treat whole thing as object
    obj_word = re.sub(r'^(?:a|an)\s+', '', segment).strip()
    return obj_word, []


def parse_object_binding_prompt(prompt, tokenizer):
    """Parse an object binding prompt into token indices for merging.

    Produces the same format as RunConfig1 in demo_config.py:
      token_indices = [[[noun_pos], [kw_pos, item_pos, ...]], ...]

    The "with/wearing" keyword IS included in attr group so it gets
    zeroed out after merging (same as "wearing" in RunConfig1).

    Returns (token_indices, prompt_anchor, prompt_merged, prompt_length).
    """
    segments = _split_into_entities(prompt)
    entities = [_parse_entity_segment(seg) for seg in segments]

    # Build word → CLIP token position mapping
    words = prompt.split()
    tok_pos = 1  # position 0 = SOT
    word_map = []  # [(raw_word, clean_lower, [token_positions])]
    for w in words:
        clean = w.strip(".,;:!?")
        n = len(tokenizer.encode(clean)) - 2
        word_map.append((w, clean.lower(), list(range(tok_pos, tok_pos + n))))
        tok_pos += n

    prompt_length = tok_pos - 1

    token_indices = []
    prompt_anchor = []
    entity_names = []
    used = set()  # word indices already claimed

    for seg_idx, (obj_word, item_words) in enumerate(entities):
        obj_key = obj_word.lower().split()[-1]

        # --- find noun word (first unused match after previous entities) ---
        noun_wi = None
        for wi, (_, wl, _) in enumerate(word_map):
            if wl == obj_key and wi not in used:
                noun_wi = wi
                used.add(wi)
                break
        if noun_wi is None:
            prompt_anchor.append(segments[seg_idx])
            entity_names.append(obj_key)
            continue

        noun_positions = list(word_map[noun_wi][2])

        # Handle compound noun (e.g., "polar bear" → noun = both words)
        obj_parts = obj_word.lower().split()
        if len(obj_parts) > 1:
            for part in obj_parts[:-1]:
                for wi2, (_, wl2, pos2) in enumerate(word_map):
                    if wl2 == part and wi2 not in used and wi2 < noun_wi:
                        noun_positions = list(pos2) + noun_positions
                        used.add(wi2)
                        break

        # --- find "with/wearing" keyword right after noun ---
        attr_positions = []
        for offset in range(1, 4):
            kwi = noun_wi + offset
            if kwi < len(word_map) and word_map[kwi][1] in ("with", "wearing") \
                    and kwi not in used:
                attr_positions.extend(word_map[kwi][2])
                used.add(kwi)
                break

        # --- find each item word after noun ---
        for item in item_words:
            for part in item.lower().split():
                for wi, (_, wl, positions) in enumerate(word_map):
                    if wl == part and wi not in used and wi > noun_wi:
                        attr_positions.extend(positions)
                        used.add(wi)
                        break

        if noun_positions and attr_positions:
            token_indices.append([noun_positions, sorted(attr_positions)])

        prompt_anchor.append(segments[seg_idx])
        entity_names.append(obj_key)

    # Simplified prompt for ETS (entity names only)
    if len(entity_names) == 0:
        prompt_merged = prompt
    elif len(entity_names) == 1:
        prompt_merged = f"a {entity_names[0]}"
    elif len(entity_names) == 2:
        prompt_merged = f"a {entity_names[0]} and a {entity_names[1]}"
    else:
        parts = [f"a {n}" for n in entity_names]
        prompt_merged = ", ".join(parts[:-1]) + ", and " + parts[-1]

    return token_indices, prompt_anchor, prompt_merged, prompt_length


# ═══════════════════════════════════════════════════════════════
#  Pipeline loading
# ═══════════════════════════════════════════════════════════════

def _load_pipeline(pipe_type, model_path, device):
    extras = {}
    if pipe_type == "tome":
        from pipe_tome import tomePipeline
        pipe = tomePipeline.from_pretrained(
            model_path, torch_dtype=torch.float16, variant="fp16",
            safety_checker=None,
        ).to(device)
    elif pipe_type == "geobind":
        from pipe_geobind import geobindPipeline
        from utils.hyperbolic_utils import TokenMergerWithAttnHyperspace
        pipe = geobindPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16, variant="fp16",
            safety_checker=None,
        ).to(device)
        extras["hyper_merger"] = (
            TokenMergerWithAttnHyperspace(embed_dim=2048, num_heads=8)
            .to(device).eval()
        )
    else:
        raise ValueError(f"Unknown pipe_type: {pipe_type}")

    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    return pipe, extras


def _build_call_kwargs(method, prompt, args, ti, pa, merged, pl,
                       controller, thresholds, extras=None):
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
        tome_control_steps=[7, 7],
        eot_replace_step=15,
        use_pose_loss=True,
        negative_prompt="low res, ugly, blurry, artifact, unreal",
    )

    if method == "ToMe":
        base["use_hyperbolic"] = False
        base["hyper_merger"] = None
        base["merge_noun_weight"] = 1.0
        base["merge_attr_weight"] = 1.0

    if method == "GeoBind" and extras and "hyper_merger" in extras:
        base["hyper_merger"] = extras["hyper_merger"]

    return base


# ═══════════════════════════════════════════════════════════════
#  Generation
# ═══════════════════════════════════════════════════════════════

def generate_images(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts_data = load_gob_bench(args.prompt_file, args.levels)
    if not prompts_data:
        log.error(f"No prompts loaded from {args.prompt_file}")
        return

    for i, p in enumerate(prompts_data):
        p["index"] = i
    log.info(f"Loaded {len(prompts_data)} GOB-Bench prompts")
    for level in LEVELS:
        count = sum(1 for p in prompts_data if p["level"] == level)
        if count > 0:
            log.info(f"  {level}: {count} prompts")

    meta_path = os.path.join(args.output_dir, "gob_bench_meta.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(prompts_data, f, indent=2, ensure_ascii=False)

    thresholds = {
        0: 26, 1: 25, 2: 24, 3: 23, 4: 22.5,
        5: 22, 6: 21.5, 7: 21, 8: 21, 9: 21,
    }

    pipe_groups = {}
    for m in args.methods:
        pt = _PIPELINE_FOR_METHOD.get(m, "tome")
        pipe_groups.setdefault(pt, []).append(m)

    for pipe_type, methods_in_group in pipe_groups.items():
        log.info(f"Loading '{pipe_type}' pipeline for {methods_in_group} ...")
        pipeline, extras = _load_pipeline(pipe_type, args.model_path, device)

        for method in methods_in_group:
            out_dir = os.path.join(args.output_dir, method, "samples")
            os.makedirs(out_dir, exist_ok=True)

            existing = len([f for f in os.listdir(out_dir) if f.endswith(".png")])
            log.info(f"{'='*55}")
            log.info(f"  {method} ({existing}/{len(prompts_data)} done)")
            log.info(f"{'='*55}")
            if existing >= len(prompts_data):
                log.info("  All images exist -- skipping.")
                continue

            n_ok, n_err = 0, 0
            for entry in tqdm(prompts_data, desc=f"  {method}"):
                idx = entry["index"]
                prompt = entry["prompt"]
                img_path = os.path.join(out_dir, f"{idx:04d}.png")
                if os.path.exists(img_path):
                    continue

                g = torch.Generator(device).manual_seed(args.seed)
                try:
                    ti, pa, merged, pl = parse_object_binding_prompt(
                        prompt, pipeline.tokenizer)
                except Exception:
                    log.warning(f"  Parse fallback for: {prompt}")
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
                    n_ok += 1
                except Exception as e:
                    tqdm.write(f"    [ERROR] '{prompt}': {e}")
                    log.error(f"  Error [{method}] '{prompt}': {e}")
                    Image.new("RGB", (1024, 1024), "gray").save(img_path)
                    n_err += 1

            log.info(f"  [{method}] done -- generated={n_ok}, errors={n_err}")

        log.info(f"Unloading '{pipe_type}' pipeline ...")
        del pipeline
        if "hyper_merger" in extras:
            del extras["hyper_merger"]
        torch.cuda.empty_cache()

    log.info("Generation complete.")


def parse_args():
    p = argparse.ArgumentParser(description="GOB-Bench image generation")
    p.add_argument("--methods", nargs="+", default=["GeoBind"],
                   choices=METHODS)
    p.add_argument("--levels", nargs="+", default=None,
                   choices=LEVELS, help="Filter by difficulty (default: all)")
    p.add_argument("--prompt_file", default=GOB_BENCH_FILE)
    p.add_argument("--output_dir", default="eval_results_gob_bench")
    p.add_argument("--model_path",
                   default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--n_inference_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.output_dir)
    log.info("=== GOB-Bench Image Generation ===")
    log.info(f"Methods: {args.methods}  |  Seed: {args.seed}")
    log.info(f"Levels: {args.levels or 'all'}")
    log.info(f"Output: {os.path.abspath(args.output_dir)}")

    try:
        generate_images(args)
    except Exception:
        log.error(f"Generation FAILED:\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
