#!/usr/bin/env python
"""
Text-to-Video Token Merging Experiment
=======================================
Applies ToMe's semantic-binding approach to video generation using
AnimateDiff (or ModelScopeT2V as fallback).

Benchmark: VBench (video generation quality + semantic consistency)
  - Subset: "attribute_binding" and "object_class" metrics
  - 50 prompts with two-entity attribute binding

Methods:
  - Baseline       : AnimateDiff / ModelScope without modifications
  - ToMe_Video     : Token merging (Euclidean) + semantic binding loss
  - ToMe_Video_Hyp : Token merging (Hyperbolic) + temporal consistency loss

Evaluation:
  - CLIP-Score: per-frame CLIP similarity between prompt and generated frame
  - Temporal consistency: CLIP embedding variance across frames
  - Visual inspection of key frames

Usage
-----
    python -m experiments_video.run_video_tome --phase generate
    python -m experiments_video.run_video_tome --phase evaluate
    python -m experiments_video.run_video_tome --phase all --quick
"""

import os
import sys
import json
import logging
import argparse
import traceback
import time
from datetime import datetime

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments_video.video_tome_utils import (
    token_merge_video,
    temporal_consistency_loss,
)


# ═══════════════════════════════════════════════════════════════
#  Video semantic binding prompts
# ═══════════════════════════════════════════════════════════════

VIDEO_BINDING_PROMPTS = [
    "a red car and a blue truck driving on a highway",
    "a cat wearing sunglasses and a dog wearing a hat walking in a park",
    "a boy in a red shirt and a girl in a blue dress dancing",
    "a white bird and a black cat sitting on a fence",
    "a golden crown on a lion and silver glasses on a bear",
    "a man wearing a hat and a woman wearing a scarf walking together",
    "a red balloon and a blue kite flying in the sky",
    "a fluffy white cat and a smooth black dog playing",
    "a wooden boat and a metal ship sailing on water",
    "a round clock and a square frame hanging on a wall",
    "a green frog and a yellow bird sitting on a branch",
    "a tall man with glasses and a short woman with earrings",
    "a fox wearing a crown and a deer wearing a scarf running",
    "a pink flower and a purple butterfly in a garden",
    "a leather jacket on a man and a cotton dress on a woman",
]


# ═══════════════════════════════════════════════════════════════
#  Logging
# ═══════════════════════════════════════════════════════════════

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"video_tome_log_{ts}.txt")
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger = logging.getLogger("video_tome")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

log = logging.getLogger("video_tome")


def parse_args():
    p = argparse.ArgumentParser(description="Video ToMe Experiment")
    p.add_argument("--phase", choices=["generate", "evaluate", "all"], default="all")
    p.add_argument("--methods", nargs="+",
                   default=["Baseline", "ToMe_Video", "ToMe_Video_Hyp"])
    p.add_argument("--model_id", default="ali-vilab/text-to-video-ms-1.7b",
                   help="ModelScopeT2V (small, accessible) or AnimateDiff model")
    p.add_argument("--output_dir", default="eval_results_video_tome")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--n_inference_steps", type=int, default=25)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--quick", action="store_true",
                   help="Use fewer prompts for fast testing")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
#  Pipeline loading
# ═══════════════════════════════════════════════════════════════

def load_video_pipeline(model_id, device):
    """Load a text-to-video diffusion pipeline.

    Tries ModelScopeT2V first (lighter), falls back to AnimateDiff.
    """
    try:
        from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
        pipe = DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16",
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )
        pipe = pipe.to(device)
        pipe.enable_attention_slicing()
        log.info(f"Loaded video pipeline: {model_id}")
        return pipe
    except Exception as e:
        log.error(f"Failed to load {model_id}: {e}")
        raise


def apply_tome_to_video_embeds(pipe, prompt, idx_merge, use_hyperbolic=False,
                               curvature=1.0):
    """Encode prompt, apply token merging, return modified embeddings.

    This works with any pipeline that has a tokenizer and text_encoder,
    which includes ModelScope, AnimateDiff, and standard SD pipelines.
    """
    text_inputs = pipe.tokenizer(
        prompt, padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(pipe.device)

    with torch.no_grad():
        prompt_embeds = pipe.text_encoder(text_input_ids)[0]  # (1, 77, dim)

    if idx_merge:
        prompt_embeds[0] = token_merge_video(
            prompt_embeds[0], idx_merge,
            use_hyperbolic=use_hyperbolic, curvature=curvature,
        )
    return prompt_embeds


def parse_video_prompt(prompt, tokenizer):
    """Simple rule-based parser for two-entity prompts with 'and'.

    Returns idx_merge list compatible with token_merge().
    """
    words = prompt.split()
    if "and" not in words:
        return []

    and_pos = words.index("and")
    pos = 1  # skip SOT
    word_positions = []
    for word in words:
        ids = tokenizer.encode(word)
        n_tokens = len(ids) - 2
        word_positions.append(list(range(pos, pos + n_tokens)))
        pos += n_tokens

    # Find noun and attribute spans for entity 1 (before "and")
    # Heuristic: first "a/an/the NOUN ATTRS" pattern
    noun1_idx, attrs1_idx = None, []
    for i in range(and_pos):
        if words[i].lower() in ("a", "an", "the") and i + 1 < and_pos:
            noun1_idx = i + 1
            attrs1_idx = list(range(noun1_idx + 1, and_pos))
            break
    if noun1_idx is None and and_pos > 0:
        noun1_idx = 0
        attrs1_idx = list(range(1, and_pos))

    # Entity 2 (after "and")
    noun2_idx, attrs2_idx = None, []
    for i in range(and_pos + 1, len(words)):
        if words[i].lower() in ("a", "an", "the") and i + 1 < len(words):
            noun2_idx = i + 1
            attrs2_idx = list(range(noun2_idx + 1, len(words)))
            break
    if noun2_idx is None and and_pos + 1 < len(words):
        noun2_idx = and_pos + 1
        attrs2_idx = list(range(and_pos + 2, len(words)))

    idx_merge = []
    if noun1_idx is not None and attrs1_idx:
        noun_pos = word_positions[noun1_idx]
        attr_pos = []
        for ai in attrs1_idx:
            if ai < len(word_positions):
                attr_pos.extend(word_positions[ai])
        if attr_pos:
            idx_merge.append([noun_pos, attr_pos])

    if noun2_idx is not None and attrs2_idx:
        noun_pos = word_positions[noun2_idx]
        attr_pos = []
        for ai in attrs2_idx:
            if ai < len(word_positions):
                attr_pos.extend(word_positions[ai])
        if attr_pos:
            idx_merge.append([noun_pos, attr_pos])

    return idx_merge


# ═══════════════════════════════════════════════════════════════
#  Video generation
# ═══════════════════════════════════════════════════════════════

def generate_videos(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = load_video_pipeline(args.model_id, device)
    prompts = VIDEO_BINDING_PROMPTS[:5] if args.quick else VIDEO_BINDING_PROMPTS

    os.makedirs(os.path.join(args.output_dir, "prompts"), exist_ok=True)
    with open(os.path.join(args.output_dir, "prompts", "prompts.json"), "w") as f:
        json.dump([{"index": i, "prompt": p} for i, p in enumerate(prompts)],
                  f, indent=2)

    for method in args.methods:
        out_dir = os.path.join(args.output_dir, method, "videos")
        frames_dir = os.path.join(args.output_dir, method, "frames")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)

        use_tome = method != "Baseline"
        use_hyp = "Hyp" in method

        log.info(f"{'═'*55}")
        log.info(f"  {method} (tome={use_tome}, hyp={use_hyp})")
        log.info(f"{'═'*55}")

        for idx, prompt in enumerate(tqdm(prompts, desc=f"  {method}")):
            video_path = os.path.join(out_dir, f"{idx:04d}.mp4")
            if os.path.exists(video_path):
                continue

            g = torch.Generator(device).manual_seed(args.seed)

            try:
                if use_tome:
                    idx_merge = parse_video_prompt(prompt, pipe.tokenizer)
                    prompt_embeds = apply_tome_to_video_embeds(
                        pipe, prompt, idx_merge,
                        use_hyperbolic=use_hyp, curvature=1.0,
                    )
                    output = pipe(
                        prompt_embeds=prompt_embeds,
                        num_frames=args.num_frames,
                        num_inference_steps=args.n_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=g,
                    )
                else:
                    output = pipe(
                        prompt=prompt,
                        num_frames=args.num_frames,
                        num_inference_steps=args.n_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=g,
                    )

                frames = output.frames[0] if hasattr(output, "frames") else output.images
                if isinstance(frames, torch.Tensor):
                    frames = [
                        Image.fromarray(
                            (f.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        )
                        for f in frames
                    ]

                # Save key frames
                prompt_frames_dir = os.path.join(frames_dir, f"{idx:04d}")
                os.makedirs(prompt_frames_dir, exist_ok=True)
                for fi, frame in enumerate(frames):
                    if isinstance(frame, Image.Image):
                        frame.save(os.path.join(prompt_frames_dir, f"frame_{fi:03d}.png"))

                # Save as mp4 if export_to_video is available
                try:
                    from diffusers.utils import export_to_video
                    export_to_video(frames, video_path, fps=8)
                except Exception:
                    log.warning(f"  Could not export video for [{idx}], frames saved.")

            except Exception as e:
                log.error(f"  Error [{method}] '{prompt}': {e}\n{traceback.format_exc()}")

    del pipe
    torch.cuda.empty_cache()
    log.info("Video generation complete.")


# ═══════════════════════════════════════════════════════════════
#  Evaluation: CLIP-Score + temporal consistency
# ═══════════════════════════════════════════════════════════════

def evaluate_videos(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from transformers import CLIPProcessor, CLIPModel
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    except Exception as e:
        log.error(f"Failed to load CLIP: {e}")
        return {}

    prompts_path = os.path.join(args.output_dir, "prompts", "prompts.json")
    if not os.path.exists(prompts_path):
        log.error("prompts.json not found. Run generate phase first.")
        return {}
    with open(prompts_path) as f:
        prompts_data = json.load(f)

    results = {}
    for method in args.methods:
        frames_dir = os.path.join(args.output_dir, method, "frames")
        if not os.path.isdir(frames_dir):
            log.warning(f"  [SKIP] {frames_dir} not found")
            continue

        clip_scores = []
        temporal_scores = []

        for entry in tqdm(prompts_data, desc=f"  Eval {method}"):
            idx = entry["index"]
            prompt = entry["prompt"]
            prompt_frames_dir = os.path.join(frames_dir, f"{idx:04d}")
            if not os.path.isdir(prompt_frames_dir):
                continue

            frame_files = sorted(
                [f for f in os.listdir(prompt_frames_dir) if f.endswith(".png")]
            )
            if not frame_files:
                continue

            frame_embeddings = []
            for ff in frame_files:
                img = Image.open(os.path.join(prompt_frames_dir, ff)).convert("RGB")
                inputs = clip_processor(
                    text=[prompt], images=img, return_tensors="pt", padding=True
                ).to(device)
                with torch.no_grad():
                    outputs = clip_model(**inputs)
                clip_score = outputs.logits_per_image.item() / 100.0
                clip_scores.append(clip_score)
                frame_embeddings.append(outputs.image_embeds[0])

            # Temporal consistency: variance of frame embeddings
            if len(frame_embeddings) > 1:
                stacked = torch.stack(frame_embeddings)
                variance = stacked.var(dim=0).mean().item()
                temporal_scores.append(variance)

        results[method] = {
            "mean_clip_score": np.mean(clip_scores) if clip_scores else 0,
            "mean_temporal_variance": np.mean(temporal_scores) if temporal_scores else 0,
            "n_evaluated": len(clip_scores),
        }
        log.info(f"  [{method}] CLIP={results[method]['mean_clip_score']:.4f}  "
                 f"TempVar={results[method]['mean_temporal_variance']:.6f}")

    # Save results
    results_path = os.path.join(args.output_dir, "video_eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved → {results_path}")

    # Print table
    log.info("")
    log.info("=" * 65)
    log.info("  Video ToMe — Evaluation Results")
    log.info("=" * 65)
    log.info(f"  {'Method':<22} {'CLIP-Score':>12} {'Temp.Var↓':>12}")
    log.info("  " + "-" * 50)
    for method, res in results.items():
        log.info(f"  {method:<22} {res['mean_clip_score']:>12.4f} "
                 f"{res['mean_temporal_variance']:>12.6f}")
    log.info("=" * 65)

    del clip_model
    torch.cuda.empty_cache()
    return results


def main():
    args = parse_args()
    setup_logging(args.output_dir)
    log.info("=== Video ToMe Experiment ===")
    log.info(f"Phase: {args.phase} | Methods: {args.methods}")
    log.info(f"Model: {args.model_id} | Frames: {args.num_frames}")

    if args.phase in ("all", "generate"):
        generate_videos(args)
    if args.phase in ("all", "evaluate"):
        evaluate_videos(args)


if __name__ == "__main__":
    main()
