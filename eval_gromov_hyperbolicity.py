#!/usr/bin/env python
"""
Gromov δ-Hyperbolicity Analysis of CLIP Token Embeddings
==========================================================
Compute Gromov δ-hyperbolicity for CLIP token embeddings to verify
that token relationships exhibit tree-like structure (Table 1 in paper).

Measures δ in three settings:
  1. Random Gaussian baseline (R^2048)
  2. CLIP tokens with Euclidean distance
  3. CLIP tokens with Poincaré ball geodesic distance

Lower δ indicates the metric space is closer to a tree metric.

Usage:
    python eval_gromov_hyperbolicity.py
    python eval_gromov_hyperbolicity.py --n_prompts 500 --data_dir data/t2i_compbench
"""

import os
import argparse
import itertools
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel


def exp_map(u, c=1.0):
    """Euclidean -> Poincaré ball (exponential map at origin)."""
    sqrt_c = c ** 0.5
    u_norm = torch.norm(u, dim=-1, keepdim=True).clamp_min(1e-15)
    return torch.tanh(sqrt_c * u_norm / 2) * u / (sqrt_c * u_norm)


def mobius_add(x, y, c=1.0):
    """Möbius addition on Poincaré ball."""
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / denom.clamp_min(1e-15)


def hyperbolic_dist(x, y, c=1.0):
    """Geodesic distance on Poincaré ball."""
    sqrt_c = c ** 0.5
    diff = mobius_add(-x, y, c)
    diff_norm = torch.norm(diff, dim=-1).clamp(min=1e-15, max=(1 - 1e-5) / sqrt_c)
    return (2.0 / sqrt_c) * torch.atanh(sqrt_c * diff_norm)


def gromov_delta_from_distances(dist_matrix):
    """Compute Gromov δ-hyperbolicity from a pairwise distance matrix.

    For a metric space (X, d), δ is the smallest value such that for all
    x, y, z, w in X:
        d(x,y) + d(z,w) <= max(d(x,z) + d(y,w), d(x,w) + d(y,z)) + 2δ
    """
    n = dist_matrix.shape[0]
    if n < 4:
        return 0.0

    max_delta = 0.0
    for i, j, k, l in itertools.combinations(range(n), 4):
        s1 = dist_matrix[i, j] + dist_matrix[k, l]
        s2 = dist_matrix[i, k] + dist_matrix[j, l]
        s3 = dist_matrix[i, l] + dist_matrix[j, k]
        sums = sorted([s1, s2, s3])
        delta = (sums[2] - sums[1]) / 2.0
        max_delta = max(max_delta, delta)

    return max_delta


def compute_euclidean_distances(embeddings):
    """Compute pairwise Euclidean distance matrix."""
    diff = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)
    return torch.norm(diff, dim=-1).cpu().numpy()


def compute_hyperbolic_distances(embeddings, c=1.0):
    """Compute pairwise geodesic distance matrix on Poincaré ball."""
    n = embeddings.shape[0]
    emb_normed = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    emb_hyp = exp_map(emb_normed, c)

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = hyperbolic_dist(emb_hyp[i], emb_hyp[j], c).item()
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    return dist_matrix


def load_prompts(data_dir, n_prompts):
    """Load compositional prompts from T2I-CompBench."""
    all_prompts = []
    for subset in ["color", "shape", "texture"]:
        path = os.path.join(data_dir, f"{subset}_val.txt")
        if os.path.isfile(path):
            with open(path) as f:
                all_prompts.extend([line.strip() for line in f if line.strip()])

    seen = set()
    unique = []
    for p in all_prompts:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique[:n_prompts]


def main():
    parser = argparse.ArgumentParser(
        description="Gromov δ-hyperbolicity analysis")
    parser.add_argument("--data_dir", default="data/t2i_compbench")
    parser.add_argument("--n_prompts", type=int, default=500)
    parser.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--curvature", type=float, default=1.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = load_prompts(args.data_dir, args.n_prompts)
    print(f"Loaded {len(prompts)} prompts")

    print(f"Loading CLIP text encoder: {args.clip_model} ...")
    tokenizer = CLIPTokenizer.from_pretrained(args.clip_model)
    text_model = CLIPTextModel.from_pretrained(args.clip_model).to(device).eval()

    deltas_euclidean = []
    deltas_hyperbolic = []
    deltas_random = []

    for prompt in tqdm(prompts, desc="Computing δ-hyperbolicity"):
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = text_model(**inputs)
        token_embs = outputs.last_hidden_state[0]

        n_tokens = inputs.attention_mask.sum().item()
        token_embs = token_embs[1:int(n_tokens) - 1].float()

        if token_embs.shape[0] < 4:
            continue

        dist_euc = compute_euclidean_distances(token_embs.cpu())
        delta_euc = gromov_delta_from_distances(dist_euc)
        deltas_euclidean.append(delta_euc)

        dist_hyp = compute_hyperbolic_distances(token_embs.cpu(), args.curvature)
        delta_hyp = gromov_delta_from_distances(dist_hyp)
        deltas_hyperbolic.append(delta_hyp)

        random_points = torch.randn_like(token_embs.cpu())
        dist_rand = compute_euclidean_distances(random_points)
        delta_rand = gromov_delta_from_distances(dist_rand)
        deltas_random.append(delta_rand)

    print("\n" + "=" * 55)
    print("  Gromov δ-Hyperbolicity Analysis")
    print(f"  ({len(deltas_euclidean)} prompts, c={args.curvature})")
    print("=" * 55)
    print(f"  {'Embedding Space':<30} {'δ_avg':<10} {'δ_max':<10}")
    print("  " + "-" * 50)

    if deltas_random:
        print(f"  {'Random Gaussian (R^2048)':<30} "
              f"{np.mean(deltas_random):<10.3f} {np.max(deltas_random):<10.3f}")
    if deltas_euclidean:
        print(f"  {'CLIP Tokens (Euclidean)':<30} "
              f"{np.mean(deltas_euclidean):<10.3f} {np.max(deltas_euclidean):<10.3f}")
    if deltas_hyperbolic:
        print(f"  {'CLIP Tokens (Poincaré ball)':<30} "
              f"{np.mean(deltas_hyperbolic):<10.3f} {np.max(deltas_hyperbolic):<10.3f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
