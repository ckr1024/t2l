"""
Hyperbolic geometry operations using the Lorentz (hyperboloid) model.

The Lorentz model represents hyperbolic space on the upper sheet of a hyperboloid
in Minkowski space R^{n+1} with the Minkowski inner product:
    <x, y>_L = -x_0 y_0 + x_1 y_1 + ... + x_n y_n

Hyperboloid: H^n_c = {x in R^{n+1} : <x,x>_L = -1/c, x_0 > 0}
Origin: o = (1/sqrt(c), 0, ..., 0)

Advantages over the Poincare ball model:
  - No boundary constraints (no need for project_to_ball clamping)
  - Gradients are stable everywhere (acosh is well-conditioned vs atanh explosion)
  - Distance computation uses inner product (simpler, no Mobius subtraction)

Mathematical references:
  - Nickel & Kiela, "Learning Continuous Hierarchies in the Lorentz Model", ICML 2018
  - Law et al., "Lorentzian Distance Learning for Hyperbolic Representations", ICML 2019
"""

import math
import torch
from typing import List


# ============================================================
# Lorentz Model Core Operations
# ============================================================

def lorentz_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Minkowski inner product: <x,y>_L = -x0*y0 + sum_{i>0} xi*yi."""
    time = -x[..., 0:1] * y[..., 0:1]
    space = (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)
    return time + space


def exp_map_lorentz(v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Exponential map at the origin of the Lorentz hyperboloid.
    Maps Euclidean vectors v in R^d to points on H^d in R^{d+1}.

    exp_o(v) = cosh(sqrt(c) ||v||) * o + sinh(sqrt(c) ||v||) / (sqrt(c) ||v||) * v_hat
    where o = (1/sqrt(c), 0,...,0) and v_hat = (0, v_1,...,v_d).
    """
    sqrt_c = math.sqrt(c)
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-7)

    t = torch.cosh(sqrt_c * v_norm) / sqrt_c
    coeff = torch.sinh(sqrt_c * v_norm) / (sqrt_c * v_norm)
    s = coeff * v

    return torch.cat([t, s], dim=-1)


def log_map_lorentz(h: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Logarithmic map at the origin. Maps hyperboloid point h to R^d.

    For h = (h_0, s) on the hyperboloid:
        d = d(o, h) = (1/sqrt(c)) * acosh(sqrt(c) * h_0)
        log_o(h) = d * s / ||s||
    """
    sqrt_c = math.sqrt(c)
    t = h[..., 0:1]
    s = h[..., 1:]

    d = torch.acosh((sqrt_c * t).clamp(min=1.0 + 1e-7)) / sqrt_c
    s_norm = s.norm(dim=-1, keepdim=True).clamp(min=1e-7)

    return (d / s_norm) * s


def lorentz_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Geodesic distance on the Lorentz hyperboloid.

    d_c(x, y) = (1/sqrt(c)) * acosh(-c * <x, y>_L)

    Numerically stable: acosh is well-conditioned for all inputs >= 1.
    Returns: (..., 1) tensor of distances.
    """
    sqrt_c = math.sqrt(c)
    inner = lorentz_inner(x, y)
    return torch.acosh((-c * inner).clamp(min=1.0 + 1e-7)) / sqrt_c


# ============================================================
# Hyperbolic Token Merging (Lorentz distance-based weighting)
# ============================================================

def token_merge_hyperbolic(
    prompt_embeds: torch.Tensor,
    idx_merge: List[List[int]],
    c: float = 1.0,
) -> torch.Tensor:
    """
    Token merging with Lorentz distance-based weighting.

    Instead of directly aggregating on the hyperboloid (numerically fragile
    for 2048-dim CLIP embeddings), we:
      1. Project tokens to 2D features (norm, cosine-to-noun)
      2. Map to the 3D Lorentz hyperboloid
      3. Compute geodesic distances (numerically stable via acosh)
      4. Convert inverse distances to attention weights
      5. Aggregate in Euclidean space using these weights

    This captures the hierarchical noun-attribute structure from hyperbolic
    geometry while keeping embeddings compatible with the pre-trained model.
    """
    for idxs in idx_merge:
        noun_idx = idxs[0][0]

        noun_tokens = prompt_embeds[idxs[0]]  # (N, dim)
        attr_tokens = prompt_embeds[idxs[1]]  # (M, dim)

        N = noun_tokens.shape[0]
        M = attr_tokens.shape[0]
        all_tokens = torch.cat([noun_tokens, attr_tokens], dim=0)

        # 2D projection: [token norm, cosine similarity to noun centroid]
        norms = all_tokens.norm(dim=-1, keepdim=True)
        noun_centroid = noun_tokens.mean(dim=0, keepdim=True)
        noun_dir = noun_centroid / noun_centroid.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        all_dir = all_tokens / norms.clamp(min=1e-8)
        cos_sim = (all_dir * noun_dir).sum(dim=-1, keepdim=True)

        features_2d = torch.cat([norms, cos_sim], dim=-1)
        feat_scale = features_2d.norm(dim=-1).max().clamp(min=1.0).item()
        features_scaled = features_2d / feat_scale * 0.5

        # Map to 3D Lorentz hyperboloid and compute distances
        features_hyp = exp_map_lorentz(features_scaled, c)
        noun_center_hyp = features_hyp[:N].mean(dim=0, keepdim=True)
        dists = lorentz_distance(
            features_hyp, noun_center_hyp.expand_as(features_hyp), c
        ).squeeze(-1)

        inv_dists = 1.0 / (dists + 0.1)
        noun_weights = torch.softmax(inv_dists[:N] * 3.0, dim=0) * N * 1.1
        attr_weights = torch.softmax(inv_dists[N:] * 3.0, dim=0) * M * 1.2

        composite = (noun_weights.unsqueeze(-1) * noun_tokens).sum(dim=0) + \
                     (attr_weights.unsqueeze(-1) * attr_tokens).sum(dim=0)

        prompt_embeds[noun_idx] = composite
        if len(idxs[0]) > 1:
            prompt_embeds[idxs[0][1:]] = 0
        prompt_embeds[idxs[1]] = 0

    return prompt_embeds


# ============================================================
# Hyperbolic Semantic Binding Loss (Lorentz geodesic distance)
# ============================================================

def hyperbolic_spatial_loss(
    pred: torch.Tensor, target: torch.Tensor, c: float = 1.0
) -> torch.Tensor:
    """
    Geodesic distance loss on the Lorentz hyperboloid, replacing MSE
    for semantic binding. Auto-calibrated to match MSE magnitude.

    Workflow:
      1. Flatten noise predictions to per-pixel 4D vectors
      2. Map to 5D Lorentz hyperboloid
      3. Compute geodesic distance (acosh-based, numerically stable)
      4. Scale to match MSE magnitude (detached ratio preserves grad direction)
    """
    B, C, H, W = pred.shape

    mse_ref = torch.nn.functional.mse_loss(pred, target)

    pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, C)
    target_flat = target.permute(0, 2, 3, 1).reshape(-1, C)

    scale = torch.cat([pred_flat, target_flat]).norm(dim=-1).max().clamp(min=1.0)
    pred_s = pred_flat / scale
    target_s = target_flat / scale

    pred_hyp = exp_map_lorentz(pred_s, c)
    target_hyp = exp_map_lorentz(target_s, c)

    dist = lorentz_distance(pred_hyp, target_hyp, c)
    raw_loss = dist.pow(2).mean()

    scale_factor = (mse_ref / raw_loss.clamp(min=1e-10)).detach()
    return raw_loss * scale_factor


# ============================================================
# Hyperbolic Entropy Loss (Lorentz Frechet variance)
# ============================================================

def compute_hyperbolic_entropy_loss(
    cross_map: torch.Tensor, c: float = 1.0
) -> torch.Tensor:
    """
    Hyperbolic Frechet variance on the Lorentz hyperboloid as a replacement
    for Shannon entropy, auto-calibrated to match Euclidean entropy magnitude.

    Workflow:
      1. Build 2D position grid for attention map pixels
      2. Map to 3D Lorentz hyperboloid
      3. Compute weighted Frechet mean in tangent space
      4. Compute weighted geodesic variance around the mean
      5. Scale to match Euclidean entropy magnitude

    Args:
        cross_map: (H, W, K) normalized attention maps (each sums to 1)
        c: Lorentz curvature
    """
    H, W, K = cross_map.shape
    device = cross_map.device

    eucl_entropy = -2.0 * (cross_map * torch.log(cross_map + 1e-5)).sum()

    grid_y = torch.linspace(-0.45, 0.45, H, device=device)
    grid_x = torch.linspace(-0.45, 0.45, W, device=device)
    gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    positions = torch.stack([gx, gy], dim=-1).reshape(-1, 2)

    pos_hyp = exp_map_lorentz(positions, c)

    raw_loss = torch.tensor(0.0, device=device)
    for k in range(K):
        weights = cross_map[:, :, k].reshape(-1)

        pos_tangent = log_map_lorentz(pos_hyp, c)
        mean_tangent = (weights.unsqueeze(-1) * pos_tangent).sum(dim=0, keepdim=True)
        mean_hyp = exp_map_lorentz(mean_tangent, c)

        dists = lorentz_distance(pos_hyp, mean_hyp.expand_as(pos_hyp), c)
        variance = (weights * dists.squeeze(-1).pow(2)).sum()
        raw_loss = raw_loss + variance

    scale_factor = (eucl_entropy.abs() / raw_loss.clamp(min=1e-10)).detach()
    return raw_loss * scale_factor


# ============================================================
# Hyperbolic Position Loss
# ============================================================

def compute_hyperbolic_pose_loss(
    pair_pos: torch.Tensor,
    pos1: torch.Tensor,
    pos2: torch.Tensor,
    grid_size: float = 32.0,
    c: float = 1.0,
) -> torch.Tensor:
    """Position loss using Lorentz geodesic distance instead of L2."""
    scale = grid_size
    p0_hyp = exp_map_lorentz((pair_pos[0] / scale).unsqueeze(0), c)
    p1_hyp = exp_map_lorentz((pair_pos[1] / scale).unsqueeze(0), c)
    t0_hyp = exp_map_lorentz((pos1 / scale).unsqueeze(0), c)
    t1_hyp = exp_map_lorentz((pos2 / scale).unsqueeze(0), c)

    loss = 0.2 * lorentz_distance(p0_hyp, t0_hyp, c).pow(2).mean()
    loss = loss + 0.2 * lorentz_distance(p1_hyp, t1_hyp, c).pow(2).mean()
    return loss
