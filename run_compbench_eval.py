#!/usr/bin/env python
"""
T2I-CompBench Evaluation
========================
Generate images with SDXL / ToMe / ToMe_Hyper and compute BLIP-VQA scores
on the color, shape, and texture attribute-binding validation subsets.

Usage
-----
    # Full pipeline (generate + evaluate)
    python run_compbench_eval.py

    # Generate only (resume-safe: skips existing images)
    python run_compbench_eval.py --phase generate

    # Evaluate only (after images are generated)
    python run_compbench_eval.py --phase evaluate

BLIP-VQA model
--------------
Uses Salesforce/blip-vqa-base from HuggingFace, which corresponds to the
BLIP model_base_vqa_capfilt_large checkpoint used by T2I-CompBench (2023).
Scoring follows the original vqa_prob protocol: P(yes) as the first
decoder token probability.
"""

import os
import json
import argparse
import torch
import torch.nn.functional as F
import spacy
from PIL import Image
from tqdm import tqdm

from pipe_tome import tomePipeline
from utils.ptp_utils import AttentionStore, register_attention_control
from utils.hyperbolic_utils import TokenMergerWithAttnHyperspace
from prompt_utils import PromptParser
from transformers import BlipProcessor, BlipForQuestionAnswering

SUBSETS = ["color", "shape", "texture"]
METHODS = ["SDXL", "ToMe", "ToMe_Hyper"]


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="T2I-CompBench BLIP-VQA evaluation")
    p.add_argument("--phase", choices=["generate", "evaluate", "all"], default="all")
    p.add_argument("--subsets", nargs="+", default=SUBSETS,
                   help="Attribute subsets to evaluate")
    p.add_argument("--methods", nargs="+", default=METHODS,
                   help="Methods to evaluate")
    p.add_argument("--model_path", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--blip_model", default="Salesforce/blip-vqa-base",
                   help="HuggingFace BLIP-VQA model (T2I-CompBench uses base)")
    p.add_argument("--output_dir", default="eval_results")
    p.add_argument("--data_dir", default="data/t2i_compbench")
    p.add_argument("--n_inference_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--np_num", type=int, default=8,
                   help="Max noun-phrase slots (>= actual NP count in any prompt)")
    p.add_argument("--guidance_scale", type=float, default=7.5)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
#  Data helpers
# ═══════════════════════════════════════════════════════════════

def load_prompts(data_dir, subset):
    path = os.path.join(data_dir, f"{subset}_val.txt")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


# ═══════════════════════════════════════════════════════════════
#  Prompt parsing for ToMe
# ═══════════════════════════════════════════════════════════════

def generate_merged_prompt(prompt, doc):
    """Remove attribute modifiers, keep sentence structure with bare nouns."""
    chunks = [
        (chunk, chunk.root.text)
        for chunk in doc.noun_chunks
        if chunk.text not in ["top", "the side", "the left", "the right"]
    ]
    if not chunks:
        return prompt
    merged = prompt
    for chunk, root in reversed(chunks):
        det = ""
        for token in chunk:
            if token.dep_ == "det":
                det = token.text + " "
                break
        merged = merged[: chunk.start_char] + det + root + merged[chunk.end_char :]
    return merged


def parse_prompt_for_tome(prompt, nlp, prompt_parser, tokenizer):
    doc = nlp(prompt)
    prompt_parser.set_doc(doc)

    token_indices = prompt_parser._get_indices(prompt)
    prompt_anchor = prompt_parser._split_prompt(doc)

    filtered_idx, filtered_anchor = [], []
    for i, idx in enumerate(token_indices):
        if len(idx[1]) > 0:
            filtered_idx.append(idx)
            if i < len(prompt_anchor):
                filtered_anchor.append(prompt_anchor[i])

    merged = generate_merged_prompt(prompt, doc)
    # prompt_length counts word-piece tokens only (no SOT / EOT)
    prompt_length = len(tokenizer(prompt)["input_ids"]) - 2
    return filtered_idx, filtered_anchor, merged, prompt_length


# ═══════════════════════════════════════════════════════════════
#  Phase 1 — Image generation
# ═══════════════════════════════════════════════════════════════

def generate_all_images(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading SDXL pipeline …")
    pipeline = tomePipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None,
    ).to(device)
    pipeline.unet.requires_grad_(False)
    pipeline.vae.requires_grad_(False)

    print("Loading spaCy + PromptParser …")
    nlp = spacy.load("en_core_web_trf")
    prompt_parser = PromptParser(args.model_path)

    hyper_merger = (
        TokenMergerWithAttnHyperspace(embed_dim=2048, num_heads=8)
        .to(device)
        .eval()
    )

    thresholds = {i: max(26 - i * 0.5, 21) for i in range(20)}

    for subset in args.subsets:
        prompts = load_prompts(args.data_dir, subset)
        print(f"\n{'═'*55}")
        print(f"  Subset: {subset}  ({len(prompts)} prompts)")
        print(f"{'═'*55}")

        for method in args.methods:
            out_dir = os.path.join(args.output_dir, method, subset, "samples")
            os.makedirs(out_dir, exist_ok=True)

            run_standard = method == "SDXL"
            use_hyper = method == "ToMe_Hyper"

            existing = len([f for f in os.listdir(out_dir) if f.endswith(".png")])
            if existing >= len(prompts):
                print(f"  [{method}] {existing} images exist — skipping.")
                continue
            print(f"\n  [{method}] generating ({existing}/{len(prompts)} done) …")

            for idx, prompt in enumerate(tqdm(prompts, desc=f"  {method}")):
                img_path = os.path.join(out_dir, f"{prompt}_{idx}.png")
                if os.path.exists(img_path):
                    continue

                g = torch.Generator(device).manual_seed(args.seed)

                try:
                    ti, pa, merged, pl = parse_prompt_for_tome(
                        prompt, nlp, prompt_parser, pipeline.tokenizer
                    )
                except Exception:
                    ti, pa, merged, pl = [], [], prompt, 0

                run_std_this = run_standard or (not ti)

                controller = AttentionStore()
                register_attention_control(pipeline, controller)

                try:
                    out = pipeline(
                        prompt=prompt,
                        guidance_scale=args.guidance_scale,
                        generator=g,
                        num_inference_steps=args.n_inference_steps,
                        attention_store=controller,
                        indices_to_alter=ti,
                        prompt_anchor=pa,
                        attention_res=32,
                        run_standard_sd=run_std_this,
                        thresholds=thresholds,
                        scale_factor=3,
                        scale_range=(1.0, 0.0),
                        prompt3=merged,
                        prompt_length=pl,
                        token_refinement_steps=3,
                        attention_refinement_steps=[6, 6],
                        tome_control_steps=[10, 10],
                        eot_replace_step=0,
                        use_pose_loss=False,
                        negative_prompt="low res, ugly, blurry, artifact, unreal",
                        use_hyperbolic=use_hyper,
                        hyper_merger=hyper_merger if use_hyper else None,
                    )
                    out.images[0].save(img_path)
                except Exception as e:
                    tqdm.write(f"    [ERROR] '{prompt}': {e}")
                    Image.new("RGB", (1024, 1024), "gray").save(img_path)

    print("\nGeneration phase complete.")
    del pipeline, hyper_merger
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════
#  Phase 2 — BLIP-VQA evaluation
# ═══════════════════════════════════════════════════════════════

def compute_vqa_yes_prob(model, processor, image, question, device):
    """
    Compute P(yes) as the first decoder-token probability —
    replicates T2I-CompBench's ``inference='vqa_prob'`` protocol.
    """
    inputs = processor(image, question, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=10,
        )
    first_logits = outputs.scores[0]  # (1, vocab_size)
    probs = F.softmax(first_logits, dim=-1)

    yes_ids = processor.tokenizer("yes", add_special_tokens=False)["input_ids"]
    return probs[0, yes_ids[0]].item()


def evaluate_subset_method(images_dir, prompts, model, processor, nlp_sm, device, np_num):
    """
    BLIP-VQA scoring following T2I-CompBench protocol:
      • for each noun-phrase slot i in [0, np_num):
          – if prompt has ≥ i+1 noun-phrases → score = P(yes | image, "{NP}?")
          – otherwise → score = 1  (no penalty)
      • image score  = ∏ᵢ score_i
      • subset score = mean of image scores
    """
    n = len(prompts)
    reward = torch.ones((n, np_num), device=device)

    # Pre-compute noun phrases once
    all_nps = []
    for prompt in prompts:
        doc = nlp_sm(prompt)
        nps = [
            chunk.text
            for chunk in doc.noun_chunks
            if chunk.text not in ["top", "the side", "the left", "the right"]
        ]
        all_nps.append(nps)

    # Process each image once, score all its noun-phrase slots
    for k, prompt in enumerate(tqdm(prompts, desc="    BLIP-VQA")):
        nps = all_nps[k]
        if not nps:
            continue

        img_path = os.path.join(images_dir, f"{prompt}_{k}.png")
        if not os.path.exists(img_path):
            for j in range(min(len(nps), np_num)):
                reward[k, j] = 0.0
            continue

        image = Image.open(img_path).convert("RGB")
        for j, np_text in enumerate(nps[:np_num]):
            score = compute_vqa_yes_prob(
                model, processor, image, f"{np_text}?", device
            )
            reward[k, j] = score

    # Per-slot diagnostics
    max_np = max(len(nps) for nps in all_nps) if all_nps else 0
    for j in range(min(max_np, np_num)):
        slot_scores = [reward[k, j].item() for k in range(n) if len(all_nps[k]) > j]
        if slot_scores:
            print(f"    NP slot {j}: mean P(yes) = {sum(slot_scores)/len(slot_scores):.4f}")

    reward_final = reward[:, 0]
    for i in range(1, np_num):
        reward_final = reward_final * reward[:, i]
    return reward_final.mean().item()


def evaluate_all(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading BLIP-VQA model: {args.blip_model} …")
    processor = BlipProcessor.from_pretrained(args.blip_model)
    model = (
        BlipForQuestionAnswering.from_pretrained(args.blip_model).to(device).eval()
    )
    nlp_sm = spacy.load("en_core_web_sm")

    results = {}
    for subset in args.subsets:
        prompts = load_prompts(args.data_dir, subset)
        results[subset] = {}

        for method in args.methods:
            images_dir = os.path.join(args.output_dir, method, subset, "samples")
            if not os.path.isdir(images_dir):
                print(f"  [SKIP] {images_dir} not found")
                results[subset][method] = None
                continue
            print(f"\n{'─'*50}")
            print(f"  Evaluating  {method} / {subset}")
            print(f"{'─'*50}")
            score = evaluate_subset_method(
                images_dir, prompts, model, processor, nlp_sm, device, args.np_num
            )
            results[subset][method] = round(score, 4)
            print(f"  ➜ BLIP-VQA score = {score:.4f}")

    del model, processor
    torch.cuda.empty_cache()
    return results


# ═══════════════════════════════════════════════════════════════
#  Phase 3 — Report
# ═══════════════════════════════════════════════════════════════

def report_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "blip_vqa_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    subsets = list(results.keys())
    methods_set = set()
    for s in subsets:
        methods_set.update(results[s].keys())
    methods = [m for m in METHODS if m in methods_set]

    print("\n" + "=" * 62)
    print("  T2I-CompBench  BLIP-VQA Scores")
    print("=" * 62)
    header = f"  {'Method':<16}"
    for s in subsets:
        header += f"{s.capitalize():<14}"
    print(header)
    print("  " + "-" * 56)
    for m in methods:
        row = f"  {m:<16}"
        for s in subsets:
            val = results[s].get(m)
            row += f"{val:<14.4f}" if val is not None else f"{'N/A':<14}"
        print(row)
    print("=" * 62)
    print(f"  Results saved → {json_path}\n")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if args.phase in ("all", "generate"):
        generate_all_images(args)

    if args.phase in ("all", "evaluate"):
        results = evaluate_all(args)
        report_results(results, args.output_dir)


if __name__ == "__main__":
    main()
