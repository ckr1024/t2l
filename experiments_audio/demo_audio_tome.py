#!/usr/bin/env python
"""
Demo: Audio Token Merging Quick Test
=====================================
Generates a few audio clips with and without ToMe token merging
for quick listening comparison. Uses shorter audio and fewer steps.

Usage
-----
    python -m experiments_audio.demo_audio_tome
    python -m experiments_audio.demo_audio_tome --prompt "a loud drum and a soft violin"
"""

import os
import sys
import argparse

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments_audio.audio_tome_utils import (
    parse_audio_prompt,
    token_merge_audio,
)


def _encode_audioldm2_prompt(pipe, prompt: str, device: str):
    """
    AudioLDM2Pipeline encode_prompt signatures vary across diffusers versions.
    This wrapper returns a dict of kwargs that can be passed into `pipe(...)`
    to avoid version-specific internal encoding bugs.
    """
    if not hasattr(pipe, "encode_prompt"):
        return {"prompt": prompt}

    # Try the most common kwarg names across versions.
    tried = []
    for kwargs in (
        {
            "prompt": prompt,
            "device": device,
            "num_waveforms_per_prompt": 1,
            "do_classifier_free_guidance": True,
        },
        {
            "prompt": prompt,
            "device": device,
            "num_waveforms_per_prompt": 1,
            "do_classifier_free_guidance": True,
            "negative_prompt": "",
        },
        {
            "prompt": prompt,
            "device": device,
            "num_waveforms_per_prompt": 1,
        },
        {"prompt": prompt, "device": device},
    ):
        try:
            out = pipe.encode_prompt(**kwargs)
            break
        except TypeError as e:
            tried.append(str(e))
            out = None
    else:
        return {"prompt": prompt}

    # Normalize outputs into kwargs for pipe.__call__.
    # Known patterns:
    # - (prompt_embeds, attention_mask, generated_prompt_embeds, generated_attention_mask)
    # - (prompt_embeds, attention_mask)
    # - prompt_embeds
    if isinstance(out, torch.Tensor):
        return {"prompt_embeds": out}

    if isinstance(out, (tuple, list)):
        result = {}
        if len(out) >= 1 and isinstance(out[0], torch.Tensor):
            result["prompt_embeds"] = out[0]
        if len(out) >= 2 and isinstance(out[1], torch.Tensor):
            result["attention_mask"] = out[1]
        if len(out) >= 3 and isinstance(out[2], torch.Tensor):
            result["generated_prompt_embeds"] = out[2]
        if len(out) >= 4 and isinstance(out[3], torch.Tensor):
            result["generated_attention_mask"] = out[3]
        if result:
            return result

    # Fallback
    return {"prompt": prompt}


def _apply_tome_to_prompt_embeds(prompt_embeds: torch.Tensor, idx_merge, use_hyp: bool):
    """
    Apply token merge to the *conditional* embedding when CFG is used.
    Supports shapes:
      - (seq, dim)
      - (batch, seq, dim)
      - (2*batch, seq, dim) where first half is unconditional.
    """
    if prompt_embeds.dim() == 2:
        return token_merge_audio(prompt_embeds, idx_merge, use_hyperbolic=use_hyp, curvature=1.0)

    if prompt_embeds.dim() == 3:
        merged = prompt_embeds.clone()
        b = merged.shape[0]
        if b >= 2:
            # Heuristic: if CFG was used, conditional often sits in the 2nd half.
            start = b // 2
            merged[start:] = token_merge_audio(
                merged[start:], idx_merge, use_hyperbolic=use_hyp, curvature=1.0
            )
        else:
            merged[:] = token_merge_audio(
                merged, idx_merge, use_hyperbolic=use_hyp, curvature=1.0
            )
        return merged

    return prompt_embeds


DEMO_PROMPTS = [
    "a loud drum and a soft violin playing",
    "a barking dog and a meowing cat",
    "a deep male voice and a high-pitched female voice singing",
]


def parse_args():
    p = argparse.ArgumentParser(description="Audio ToMe quick demo")
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--model_id", default="cvssp/audioldm2",
                   help="AudioLDM2 model")
    p.add_argument("--output_dir", default="demo_output_audio_tome")
    p.add_argument("--cache_dir", default=None,
                   help="HF cache dir (default: <output_dir>/.hf_cache)")
    p.add_argument("--offline", action="store_true",
                   help="Do not download; use local cache only")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--audio_length", type=float, default=3.0,
                   help="Shorter audio for fast demo")
    p.add_argument("--n_inference_steps", type=int, default=20,
                   help="Fewer steps for fast demo")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = [args.prompt] if args.prompt else DEMO_PROMPTS

    print("Loading AudioLDM2 pipeline …")
    try:
        from diffusers import AudioLDM2Pipeline
        cache_dir = args.cache_dir or os.path.join(args.output_dir, ".hf_cache")
        pipe = AudioLDM2Pipeline.from_pretrained(
            args.model_id, torch_dtype=torch.float16,
            cache_dir=cache_dir,
            local_files_only=bool(args.offline),
        ).to(device)
    except Exception as e:
        print(f"Failed to load AudioLDM2: {e}")
        print("Install: pip install diffusers[torch] transformers scipy")
        return

    import scipy.io.wavfile

    methods = [
        ("baseline", False, False),
        ("tome_euclidean", True, False),
        ("tome_hyperbolic", True, True),
    ]

    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")

        safe_name = prompt[:40].replace(" ", "_").replace("/", "_")
        prompt_dir = os.path.join(args.output_dir, safe_name)
        os.makedirs(prompt_dir, exist_ok=True)

        # Parse prompt for token merging
        idx_merge, prompt_anchor = parse_audio_prompt(prompt, pipe.tokenizer)
        print(f"  Token indices to merge: {idx_merge}")
        print(f"  Prompt anchors: {prompt_anchor}")

        for label, use_tome, use_hyp in methods:
            print(f"  Generating: {label} …")
            g = torch.Generator(device).manual_seed(args.seed)

            try:
                call_kwargs = _encode_audioldm2_prompt(pipe, prompt, device=device)
                if use_tome and idx_merge and "prompt_embeds" in call_kwargs:
                    call_kwargs["prompt_embeds"] = _apply_tome_to_prompt_embeds(
                        call_kwargs["prompt_embeds"], idx_merge, use_hyp=use_hyp
                    )

                output = pipe(
                    **call_kwargs,
                    audio_length_in_s=args.audio_length,
                    num_inference_steps=args.n_inference_steps,
                    guidance_scale=3.5,
                    generator=g,
                )

                audio = output.audios[0]
                audio_path = os.path.join(prompt_dir, f"{label}.wav")
                scipy.io.wavfile.write(audio_path, 16000, audio)
                print(f"    Saved → {audio_path}")

                # Print basic audio stats
                rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
                print(f"    RMS={rms:.4f}, Duration={len(audio)/16000:.1f}s, "
                      f"Range=[{audio.min():.3f}, {audio.max():.3f}]")

            except Exception as e:
                print(f"    [ERROR] {e}")

    del pipe
    torch.cuda.empty_cache()
    print("\nDemo complete. Listen to the .wav files to compare quality.")


if __name__ == "__main__":
    main()
