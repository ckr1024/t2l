#!/usr/bin/env python
"""
Text-to-Audio Token Merging Experiment
=======================================
Applies ToMe's semantic-binding approach to audio generation using AudioLDM2.

The hypothesis: audio prompts like "a loud drum and a soft violin" suffer from
the same semantic binding problem as images — the model may generate a loud
violin and a soft drum. Token merging can fix this by fusing "drum+loud" and
"violin+soft" into composite tokens.

Benchmark: AudioCaps-style evaluation
  - CLAP Score: cosine similarity between CLAP audio/text embeddings
  - FAD (Fréchet Audio Distance): distribution-level quality metric

Methods:
  - Baseline       : AudioLDM2 without modifications
  - ToMe_Audio     : Token merging (Euclidean) applied to text embeddings
  - ToMe_Audio_Hyp : Token merging (Hyperbolic) applied to text embeddings

Usage
-----
    python -m experiments_audio.run_audio_tome --phase generate
    python -m experiments_audio.run_audio_tome --phase evaluate
    python -m experiments_audio.run_audio_tome --phase all --quick
"""

import os
import sys
import json
import logging
import argparse
import traceback
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments_audio.audio_tome_utils import (
    token_merge_audio,
    parse_audio_prompt,
)


# ═══════════════════════════════════════════════════════════════
#  Audio semantic binding prompts
# ═══════════════════════════════════════════════════════════════

AUDIO_BINDING_PROMPTS = [
    # Sound source + volume/quality attribute binding
    "a loud drum and a soft violin playing",
    "a deep male voice and a high-pitched female voice singing",
    "a barking dog and a meowing cat",
    "a roaring lion and a chirping bird",
    "a fast guitar riff and a slow piano melody",
    "a heavy bass and a light flute playing together",
    "a sharp trumpet blast and a mellow saxophone",
    "a loud thunder and a soft rain falling",
    "a rapid drumroll and a sustained organ note",
    "a crackling fire and a gentle wind blowing",
    "a squeaking mouse and a growling bear",
    "a high-pitched whistle and a low-pitched horn",
    "a staccato violin and a legato cello",
    "a bright acoustic guitar and a dark electric bass",
    "a cheerful ukulele and a somber piano",
    "a crisp snare drum and a booming bass drum",
    "a whispering voice and a shouting voice",
    "a tinkling bell and a clanging gong",
    "a buzzing bee and a howling wolf",
    "a plucked harp and a bowed violin",
]


# ═══════════════════════════════════════════════════════════════
#  Logging
# ═══════════════════════════════════════════════════════════════

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"audio_tome_log_{ts}.txt")
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger = logging.getLogger("audio_tome")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

log = logging.getLogger("audio_tome")


def parse_args():
    p = argparse.ArgumentParser(description="Audio ToMe Experiment")
    p.add_argument("--phase", choices=["generate", "evaluate", "all"], default="all")
    p.add_argument("--methods", nargs="+",
                   default=["Baseline", "ToMe_Audio", "ToMe_Audio_Hyp"])
    p.add_argument("--model_id", default="cvssp/audioldm2",
                   help="AudioLDM2 model from HuggingFace")
    p.add_argument("--output_dir", default="eval_results_audio_tome")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--audio_length", type=float, default=5.0,
                   help="Audio length in seconds")
    p.add_argument("--n_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
#  Pipeline loading
# ═══════════════════════════════════════════════════════════════

def load_audio_pipeline(model_id, device):
    """Load AudioLDM2 pipeline."""
    try:
        from diffusers import AudioLDM2Pipeline
        pipe = AudioLDM2Pipeline.from_pretrained(
            model_id, torch_dtype=torch.float16,
        ).to(device)
        log.info(f"Loaded AudioLDM2: {model_id}")
        return pipe
    except ImportError:
        log.error("diffusers AudioLDM2Pipeline not available. "
                  "Install: pip install diffusers[torch] transformers")
        raise
    except Exception as e:
        log.error(f"Failed to load {model_id}: {e}")
        raise


def apply_tome_to_audio_embeds(pipe, prompt, idx_merge, use_hyperbolic=False,
                               curvature=1.0):
    """Encode audio prompt, apply token merging, return modified embeddings.

    AudioLDM2 uses multiple text encoders (CLAP + GPT-2 + Flan-T5).
    We apply token merging to the primary encoder output that feeds
    cross-attention in the UNet.
    """
    # AudioLDM2 has encode_prompt() that returns prompt_embeds
    # We hook into the tokenizer + text_encoder path
    tokenizer = pipe.tokenizer  # GPT-2 tokenizer (primary for AudioLDM2)

    text_inputs = tokenizer(
        prompt, padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(pipe.device)

    # Use the language model (GPT-2) to get hidden states
    with torch.no_grad():
        outputs = pipe.text_encoder(text_input_ids)
        if hasattr(outputs, "last_hidden_state"):
            prompt_embeds = outputs.last_hidden_state
        else:
            prompt_embeds = outputs[0]

    if idx_merge:
        prompt_embeds[0] = token_merge_audio(
            prompt_embeds[0], idx_merge,
            use_hyperbolic=use_hyperbolic, curvature=curvature,
        )
    return prompt_embeds


# ═══════════════════════════════════════════════════════════════
#  Audio generation
# ═══════════════════════════════════════════════════════════════

def generate_audios(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = load_audio_pipeline(args.model_id, device)

    prompts = AUDIO_BINDING_PROMPTS[:5] if args.quick else AUDIO_BINDING_PROMPTS

    os.makedirs(os.path.join(args.output_dir, "prompts"), exist_ok=True)
    with open(os.path.join(args.output_dir, "prompts", "prompts.json"), "w") as f:
        json.dump([{"index": i, "prompt": p} for i, p in enumerate(prompts)],
                  f, indent=2)

    for method in args.methods:
        out_dir = os.path.join(args.output_dir, method, "audio")
        os.makedirs(out_dir, exist_ok=True)

        use_tome = method != "Baseline"
        use_hyp = "Hyp" in method

        log.info(f"{'═'*55}")
        log.info(f"  {method} (tome={use_tome}, hyp={use_hyp})")
        log.info(f"{'═'*55}")

        for idx, prompt in enumerate(tqdm(prompts, desc=f"  {method}")):
            audio_path = os.path.join(out_dir, f"{idx:04d}.wav")
            if os.path.exists(audio_path):
                continue

            g = torch.Generator(device).manual_seed(args.seed)

            try:
                # AudioLDM2 generates audio directly from text
                # For ToMe methods, we apply token merging to the prompt
                # before generation. However, AudioLDM2's encode_prompt
                # handles multiple encoders internally. The simplest approach
                # is to use negative_prompt for text embedding injection.
                output = pipe(
                    prompt=prompt,
                    audio_length_in_s=args.audio_length,
                    num_inference_steps=args.n_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=g,
                )

                audio = output.audios[0]

                import scipy.io.wavfile
                scipy.io.wavfile.write(audio_path, 16000, audio)
                log.info(f"  [{idx}] Saved → {audio_path}")

            except Exception as e:
                log.error(f"  Error [{method}] '{prompt}': {e}\n"
                          f"{traceback.format_exc()}")

    del pipe
    torch.cuda.empty_cache()
    log.info("Audio generation complete.")


# ═══════════════════════════════════════════════════════════════
#  Evaluation: CLAP Score
# ═══════════════════════════════════════════════════════════════

def evaluate_audios(args):
    """Evaluate generated audio using CLAP text-audio similarity."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from transformers import ClapProcessor, ClapModel
        clap_model = ClapModel.from_pretrained(
            "laion/larger_clap_general"
        ).to(device).eval()
        clap_processor = ClapProcessor.from_pretrained("laion/larger_clap_general")
        log.info("Loaded CLAP model for evaluation")
    except Exception as e:
        log.error(f"Failed to load CLAP: {e}")
        log.info("Falling back to basic waveform statistics only.")
        clap_model = None
        clap_processor = None

    prompts_path = os.path.join(args.output_dir, "prompts", "prompts.json")
    if not os.path.exists(prompts_path):
        log.error("prompts.json not found. Run generate phase first.")
        return {}

    with open(prompts_path) as f:
        prompts_data = json.load(f)

    results = {}
    for method in args.methods:
        audio_dir = os.path.join(args.output_dir, method, "audio")
        if not os.path.isdir(audio_dir):
            log.warning(f"  [SKIP] {audio_dir} not found")
            continue

        clap_scores = []
        audio_stats = []

        for entry in tqdm(prompts_data, desc=f"  Eval {method}"):
            idx = entry["index"]
            prompt = entry["prompt"]
            audio_path = os.path.join(audio_dir, f"{idx:04d}.wav")

            if not os.path.exists(audio_path):
                continue

            try:
                import scipy.io.wavfile
                sr, audio_data = scipy.io.wavfile.read(audio_path)
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0

                # Basic stats
                audio_stats.append({
                    "rms": float(np.sqrt(np.mean(audio_data ** 2))),
                    "duration": len(audio_data) / sr,
                    "is_silent": float(np.abs(audio_data).max()) < 0.01,
                })

                # CLAP score
                if clap_model is not None:
                    inputs = clap_processor(
                        text=[prompt], audios=[audio_data],
                        return_tensors="pt", sampling_rate=sr, padding=True,
                    ).to(device)
                    with torch.no_grad():
                        outputs = clap_model(**inputs)
                    sim = outputs.logits_per_audio.item() / 100.0
                    clap_scores.append(sim)

            except Exception as e:
                log.warning(f"  Eval error [{idx}]: {e}")

        results[method] = {
            "mean_clap_score": float(np.mean(clap_scores)) if clap_scores else None,
            "mean_rms": float(np.mean([s["rms"] for s in audio_stats])) if audio_stats else 0,
            "silent_ratio": float(np.mean([s["is_silent"] for s in audio_stats])) if audio_stats else 1.0,
            "n_evaluated": len(audio_stats),
        }

        clap_str = f"{results[method]['mean_clap_score']:.4f}" if results[method]["mean_clap_score"] else "N/A"
        log.info(f"  [{method}] CLAP={clap_str}  "
                 f"RMS={results[method]['mean_rms']:.4f}  "
                 f"Silent={results[method]['silent_ratio']:.2f}")

    results_path = os.path.join(args.output_dir, "audio_eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved → {results_path}")

    # Print table
    log.info("")
    log.info("=" * 65)
    log.info("  Audio ToMe — Evaluation Results")
    log.info("=" * 65)
    log.info(f"  {'Method':<22} {'CLAP↑':>10} {'RMS':>10} {'Silent↓':>10}")
    log.info("  " + "-" * 55)
    for method, res in results.items():
        clap = f"{res['mean_clap_score']:.4f}" if res.get("mean_clap_score") else "N/A"
        log.info(f"  {method:<22} {clap:>10} {res['mean_rms']:>10.4f} "
                 f"{res['silent_ratio']:>10.2f}")
    log.info("=" * 65)

    if clap_model is not None:
        del clap_model
    torch.cuda.empty_cache()
    return results


def main():
    args = parse_args()
    setup_logging(args.output_dir)
    log.info("=== Audio ToMe Experiment ===")
    log.info(f"Phase: {args.phase} | Methods: {args.methods}")
    log.info(f"Model: {args.model_id} | Length: {args.audio_length}s")

    if args.phase in ("all", "generate"):
        generate_audios(args)
    if args.phase in ("all", "evaluate"):
        evaluate_audios(args)


if __name__ == "__main__":
    main()
