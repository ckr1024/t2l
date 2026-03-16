#!/usr/bin/env python
"""
Demo: Video Token Merging Quick Test
=====================================
Generates a single short video with and without ToMe token merging
for side-by-side visual comparison. Uses fewer frames and steps for speed.

Usage
-----
    python -m experiments_video.demo_video_tome
    python -m experiments_video.demo_video_tome --prompt "a red car and a blue truck driving"
"""

import os
import sys
import argparse

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments_video.video_tome_utils import token_merge_video
from experiments_video.run_video_tome import (
    load_video_pipeline, parse_video_prompt, apply_tome_to_video_embeds,
)


DEMO_PROMPTS = [
    "a cat wearing sunglasses and a dog wearing a hat walking in a park",
    "a red car and a blue truck driving on a highway",
]


def parse_args():
    p = argparse.ArgumentParser(description="Video ToMe quick demo")
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--model_id", default="ali-vilab/text-to-video-ms-1.7b")
    p.add_argument("--output_dir", default="demo_output_video_tome")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_frames", type=int, default=8,
                   help="Fewer frames for fast demo")
    p.add_argument("--n_inference_steps", type=int, default=15,
                   help="Fewer steps for fast demo")
    return p.parse_args()


def save_frame_grid(frames, path, label=""):
    """Save a horizontal grid of key frames."""
    if not frames:
        return
    pil_frames = []
    for f in frames:
        if isinstance(f, Image.Image):
            pil_frames.append(f)
        elif isinstance(f, torch.Tensor):
            arr = (f.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil_frames.append(Image.fromarray(arr))
        elif isinstance(f, np.ndarray):
            pil_frames.append(Image.fromarray(f))

    if not pil_frames:
        return

    # Pick 4 evenly-spaced frames
    indices = np.linspace(0, len(pil_frames) - 1, min(4, len(pil_frames)), dtype=int)
    selected = [pil_frames[i] for i in indices]

    w, h = selected[0].size
    grid = Image.new("RGB", (w * len(selected), h), "white")
    for i, img in enumerate(selected):
        grid.paste(img, (i * w, 0))
    grid.save(path)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = [args.prompt] if args.prompt else DEMO_PROMPTS

    print("Loading video pipeline …")
    pipe = load_video_pipeline(args.model_id, device)

    methods = [
        ("baseline", False, False),
        ("tome_euclidean", True, False),
        ("tome_hyperbolic", True, True),
    ]

    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")

        safe_name = prompt[:40].replace(" ", "_")
        prompt_dir = os.path.join(args.output_dir, safe_name)
        os.makedirs(prompt_dir, exist_ok=True)

        idx_merge = parse_video_prompt(prompt, pipe.tokenizer)
        print(f"  Token indices to merge: {idx_merge}")

        for label, use_tome, use_hyp in methods:
            print(f"  Generating: {label} …")
            g = torch.Generator(device).manual_seed(args.seed)

            try:
                if use_tome and idx_merge:
                    prompt_embeds = apply_tome_to_video_embeds(
                        pipe, prompt, idx_merge,
                        use_hyperbolic=use_hyp, curvature=1.0,
                    )
                    output = pipe(
                        prompt_embeds=prompt_embeds,
                        num_frames=args.num_frames,
                        num_inference_steps=args.n_inference_steps,
                        guidance_scale=7.5,
                        generator=g,
                    )
                else:
                    output = pipe(
                        prompt=prompt,
                        num_frames=args.num_frames,
                        num_inference_steps=args.n_inference_steps,
                        guidance_scale=7.5,
                        generator=g,
                    )

                frames = output.frames[0] if hasattr(output, "frames") else output.images

                grid_path = os.path.join(prompt_dir, f"{label}_frames.png")
                save_frame_grid(frames, grid_path, label)
                print(f"    Saved frame grid → {grid_path}")

                try:
                    from diffusers.utils import export_to_video
                    video_path = os.path.join(prompt_dir, f"{label}.mp4")
                    export_to_video(frames, video_path, fps=4)
                    print(f"    Saved video → {video_path}")
                except Exception:
                    pass

            except Exception as e:
                print(f"    [ERROR] {e}")

    del pipe
    torch.cuda.empty_cache()
    print("\nDemo complete.")


if __name__ == "__main__":
    main()
