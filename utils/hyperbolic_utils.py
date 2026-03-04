"""
Hyperbolic geometry operations in the Poincaré ball model.

Provides core operations for the hyperbolic improvement to ToMe:
  1. Token merging via Möbius addition (replacing Euclidean summation)
  2. Geodesic distance for semantic binding loss (replacing L2/MSE)
  3. Hyperbolic Fréchet variance for entropy loss (replacing Shannon entropy)

All operations work in the Poincaré ball model B_c^n = {x ∈ R^n : c||x||² < 1}
with curvature parameter c > 0.

Mathematical reference:
  - Nickel & Kiela, "Poincaré Embeddings for Learning Hierarchical Representations", NeurIPS 2017
  - Ganea et al., "Hyperbolic Neural Networks", NeurIPS 2018
"""

import torch
from typing import List


# ============================================================
# Core Poincaré Ball Operations
# ============================================================

def artanh(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable arctanh using log1p."""
    x = x.clamp(-1 + 1e-7, 1 - 1e-7)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def project_to_ball(x: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """Project points to strictly inside the Poincaré ball."""
    max_norm = (1.0 - eps) / (c ** 0.5)
    norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cond = norm > max_norm
    x_proj = x / norm * max_norm
    return torch.where(cond, x_proj, x)


def exp_map_zero(v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Exponential map at the origin of the Poincaré ball.
    Maps tangent vectors (Euclidean) → Poincaré ball.

    exp_0(v) = tanh(√c ||v||) · v / (√c ||v||)
    """
    sqrt_c = c ** 0.5
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    result = torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)
    return project_to_ball(result, c)


def log_map_zero(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Logarithmic map at the origin of the Poincaré ball.
    Maps Poincaré ball → tangent vectors (Euclidean).

    log_0(x) = arctanh(√c ||x||) · x / (√c ||x||)
    """
    sqrt_c = c ** 0.5
    x = project_to_ball(x, c)
    x_norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return artanh(sqrt_c * x_norm) * x / (sqrt_c * x_norm)


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Möbius addition in the Poincaré ball.

    x ⊕_c y = ((1 + 2c⟨x,y⟩ + c||y||²)x + (1 - c||x||²)y)
              / (1 + 2c⟨x,y⟩ + c²||x||²||y||²)
    """
    x = project_to_ball(x, c)
    y = project_to_ball(y, c)

    x_sq = (x * x).sum(dim=-1, keepdim=True)
    y_sq = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)

    num = (1 + 2 * c * xy + c * y_sq) * x + (1 - c * x_sq) * y
    den = 1 + 2 * c * xy + c ** 2 * x_sq * y_sq

    return project_to_ball(num / den.clamp(min=1e-8), c)


def mobius_scalar_mult(r: float, x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Möbius scalar multiplication.

    r ⊗_c x = (1/√c) tanh(r · arctanh(√c ||x||)) · x / ||x||
    """
    sqrt_c = c ** 0.5
    x = project_to_ball(x, c)
    x_norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    scaled_norm = torch.tanh(r * artanh(sqrt_c * x_norm)) / sqrt_c
    return project_to_ball(scaled_norm * x / x_norm, c)


def hyperbolic_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Geodesic distance in the Poincaré ball.

    d_c(x, y) = (2/√c) · arctanh(√c || (-x) ⊕_c y ||)

    Returns: (..., 1) tensor of distances.
    """
    sqrt_c = c ** 0.5
    neg_x = -x
    add_result = mobius_add(neg_x, y, c)
    add_norm = add_result.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (2.0 / sqrt_c) * artanh(sqrt_c * add_norm)


# ============================================================
# Hyperbolic Token Merging (replacing Euclidean addition)
# ============================================================

def token_merge_hyperbolic(
    prompt_embeds: torch.Tensor,
    idx_merge: List[List[int]],
    c: float = 1.0,
) -> torch.Tensor:
    """
    Token merging via Möbius addition in the Poincaré ball.

    Instead of ĉ_k = α·n_k + 1.2·Σa_k (Euclidean addition),
    we perform:
      1. Euclidean → Poincaré ball (exp map)
      2. Möbius scalar multiplication for weighting
      3. Möbius addition for aggregation
      4. Poincaré ball → Euclidean (log map)

    This respects the hyperbolic geometry of the embedding space,
    capturing hierarchical entity-attribute relationships better
    than flat Euclidean addition.
    """
    for idxs in idx_merge:
        noun_idx = idxs[0][0]

        noun_tokens = prompt_embeds[idxs[0]]  # (N, dim)
        attr_tokens = prompt_embeds[idxs[1]]  # (M, dim)

        all_tokens = torch.cat([noun_tokens, attr_tokens], dim=0)
        scale = all_tokens.norm(dim=-1).max().clamp(min=1.0).item()

        noun_scaled = noun_tokens / scale
        attr_scaled = attr_tokens / scale

        noun_hyp = exp_map_zero(noun_scaled, c)
        attr_hyp = exp_map_zero(attr_scaled, c)

        result = mobius_scalar_mult(1.1, noun_hyp[0:1], c)
        for i in range(1, noun_hyp.shape[0]):
            next_tok = mobius_scalar_mult(1.1, noun_hyp[i : i + 1], c)
            result = mobius_add(result, next_tok, c)

        for i in range(attr_hyp.shape[0]):
            next_tok = mobius_scalar_mult(1.2, attr_hyp[i : i + 1], c)
            result = mobius_add(result, next_tok, c)

        composite = log_map_zero(result, c) * scale

        prompt_embeds[noun_idx] = composite.squeeze(0)
        if len(idxs[0]) > 1:
            prompt_embeds[idxs[0][1:]] = 0
        prompt_embeds[idxs[1]] = 0

    return prompt_embeds


# ============================================================
# Hyperbolic Semantic Binding Loss (replacing MSE)
# ============================================================

def hyperbolic_spatial_loss(
    pred: torch.Tensor, target: torch.Tensor, c: float = 1.0
) -> torch.Tensor:
    """
    Geodesic distance loss replacing MSE for semantic binding.

    For noise predictions of shape (B, C, H, W), computes per-pixel
    geodesic distance in a C-dimensional Poincaré ball, then averages.
    This replaces ||ε_θ(z_t, ĉ_k, t) - ε_θ(z_t, C, t)||² with
    d_H(ε_θ(z_t, ĉ_k, t), ε_θ(z_t, C, t))².

    Using per-pixel C-dim (C=4 for latent diffusion) Poincaré ball
    keeps the dimension low for numerical stability.
    """
    B, C, H, W = pred.shape
    pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, C)     # (B*H*W, C)
    target_flat = target.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)

    scale = torch.cat([pred_flat, target_flat]).norm(dim=-1).max().clamp(min=1.0)
    pred_s = pred_flat / scale
    target_s = target_flat / scale

    pred_hyp = exp_map_zero(pred_s, c)
    target_hyp = exp_map_zero(target_s, c)

    dist = hyperbolic_distance(pred_hyp, target_hyp, c)  # (B*H*W, 1)
    return dist.pow(2).mean()


# ============================================================
# Hyperbolic Entropy Loss (replacing Shannon entropy)
# ============================================================

def compute_hyperbolic_entropy_loss(
    cross_map: torch.Tensor, c: float = 1.0
) -> torch.Tensor:
    """
    Hyperbolic Fréchet variance as a replacement for Shannon entropy.

    For each token's attention map (H×W probability distribution over
    spatial positions), computes the weighted Fréchet variance in the
    2D Poincaré ball:
      1. Map 2D grid positions to Poincaré ball
      2. Compute weighted Fréchet mean (tangent space approximation)
      3. Compute weighted sum of squared geodesic distances to mean

    Minimizing this variance concentrates the attention in hyperbolic
    space, analogous to minimizing Shannon entropy in Euclidean space,
    but respecting the curved geometry.

    Args:
        cross_map: (H, W, K) normalized attention maps (each sums to 1)
        c: Poincaré ball curvature

    Returns:
        Scalar loss (sum of Fréchet variances over tokens)
    """
    H, W, K = cross_map.shape
    device = cross_map.device

    grid_y = torch.linspace(-0.45, 0.45, H, device=device)
    grid_x = torch.linspace(-0.45, 0.45, W, device=device)
    gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    positions = torch.stack([gx, gy], dim=-1).reshape(-1, 2)  # (H*W, 2)

    pos_hyp = exp_map_zero(positions, c)  # (H*W, 2)
    pos_tangent = log_map_zero(pos_hyp, c)  # (H*W, 2) for mean computation

    loss = 0.0
    for k in range(K):
        weights = cross_map[:, :, k].reshape(-1)  # (H*W,)

        mean_tangent = (weights.unsqueeze(-1) * pos_tangent).sum(
            dim=0, keepdim=True
        )  # (1, 2)
        mean_hyp = exp_map_zero(mean_tangent, c)  # (1, 2)

        dists = hyperbolic_distance(
            pos_hyp, mean_hyp.expand_as(pos_hyp), c
        )  # (H*W, 1)
        variance = (weights * dists.squeeze(-1).pow(2)).sum()

        loss = loss + variance

    return loss


def compute_hyperbolic_pose_loss(
    pair_pos: torch.Tensor,
    pos1: torch.Tensor,
    pos2: torch.Tensor,
    grid_size: float = 32.0,
    c: float = 1.0,
) -> torch.Tensor:
    """
    Position loss using geodesic distance instead of L2.

    Maps centroid positions and target positions to the Poincaré ball,
    then computes squared geodesic distance.
    """
    scale = grid_size
    p0_hyp = exp_map_zero((pair_pos[0] / scale).unsqueeze(0), c)
    p1_hyp = exp_map_zero((pair_pos[1] / scale).unsqueeze(0), c)
    t0_hyp = exp_map_zero((pos1 / scale).unsqueeze(0), c)
    t1_hyp = exp_map_zero((pos2 / scale).unsqueeze(0), c)

    loss = 0.2 * hyperbolic_distance(p0_hyp, t0_hyp, c).pow(2).mean()
    loss = loss + 0.2 * hyperbolic_distance(p1_hyp, t1_hyp, c).pow(2).mean()
    return loss
