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
    Token merging via Möbius addition in the Poincaré ball,
    with norm preservation to match the pre-trained model's expectations.

    Strategy:
      1. Compute Euclidean composite (for target norm)
      2. Compute hyperbolic composite (for direction via Möbius ops)
      3. Rescale hyperbolic direction to match Euclidean norm

    This preserves the hierarchical structure from hyperbolic geometry
    while keeping embeddings in the distribution the model expects.
    """
    for idxs in idx_merge:
        noun_idx = idxs[0][0]

        noun_tokens = prompt_embeds[idxs[0]]  # (N, dim)
        attr_tokens = prompt_embeds[idxs[1]]  # (M, dim)

        # Euclidean reference for norm matching
        eucl_composite = 1.1 * noun_tokens.sum(dim=0) + 1.2 * attr_tokens.sum(dim=0)
        eucl_norm = eucl_composite.norm().clamp(min=1e-8)

        # Hyperbolic operations for direction
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

        hyp_direction = log_map_zero(result, c) * scale  # back to Euclidean scale
        hyp_direction = hyp_direction.squeeze(0)
        hyp_norm = hyp_direction.norm().clamp(min=1e-8)

        # Rescale to match Euclidean norm but keep hyperbolic direction
        composite = hyp_direction * (eucl_norm / hyp_norm)

        prompt_embeds[noun_idx] = composite
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
    Geodesic distance loss replacing MSE for semantic binding,
    auto-calibrated to match MSE magnitude.

    Computes both MSE (for scale reference) and geodesic distance²,
    then scales the geodesic loss to match MSE. This preserves the
    correct gradient step size while using hyperbolic geometry
    for the gradient *direction*.
    """
    B, C, H, W = pred.shape

    mse_ref = torch.nn.functional.mse_loss(pred, target)

    pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, C)
    target_flat = target.permute(0, 2, 3, 1).reshape(-1, C)

    scale = torch.cat([pred_flat, target_flat]).norm(dim=-1).max().clamp(min=1.0)
    pred_s = pred_flat / scale
    target_s = target_flat / scale

    pred_hyp = exp_map_zero(pred_s, c)
    target_hyp = exp_map_zero(target_s, c)

    dist = hyperbolic_distance(pred_hyp, target_hyp, c)
    raw_loss = dist.pow(2).mean()

    # Auto-calibrate: scale geodesic² to match MSE magnitude
    # .detach() on ratio so it doesn't affect gradient direction
    scale_factor = (mse_ref / raw_loss.clamp(min=1e-10)).detach()
    return raw_loss * scale_factor


# ============================================================
# Hyperbolic Entropy Loss (replacing Shannon entropy)
# ============================================================

def compute_hyperbolic_entropy_loss(
    cross_map: torch.Tensor, c: float = 1.0
) -> torch.Tensor:
    """
    Hyperbolic Fréchet variance as a replacement for Shannon entropy,
    auto-calibrated to match the Euclidean entropy loss magnitude.

    Computes both Shannon entropy (for scale reference) and Fréchet
    variance in the 2D Poincaré ball, then scales to match. This
    ensures the attention concentration has the same optimization
    strength while using hyperbolic geometry for gradient structure.

    Args:
        cross_map: (H, W, K) normalized attention maps (each sums to 1)
        c: Poincaré ball curvature

    Returns:
        Scalar loss (calibrated to match Euclidean entropy magnitude)
    """
    H, W, K = cross_map.shape
    device = cross_map.device

    # Euclidean entropy reference (same formula as in _entropy_loss)
    eucl_entropy = -2.0 * (cross_map * torch.log(cross_map + 1e-5)).sum()

    grid_y = torch.linspace(-0.45, 0.45, H, device=device)
    grid_x = torch.linspace(-0.45, 0.45, W, device=device)
    gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    positions = torch.stack([gx, gy], dim=-1).reshape(-1, 2)

    pos_hyp = exp_map_zero(positions, c)
    pos_tangent = log_map_zero(pos_hyp, c)

    raw_loss = torch.tensor(0.0, device=device)
    for k in range(K):
        weights = cross_map[:, :, k].reshape(-1)

        mean_tangent = (weights.unsqueeze(-1) * pos_tangent).sum(
            dim=0, keepdim=True
        )
        mean_hyp = exp_map_zero(mean_tangent, c)

        dists = hyperbolic_distance(
            pos_hyp, mean_hyp.expand_as(pos_hyp), c
        )
        variance = (weights * dists.squeeze(-1).pow(2)).sum()
        raw_loss = raw_loss + variance

    # Auto-calibrate to Euclidean entropy magnitude
    scale_factor = (eucl_entropy.abs() / raw_loss.clamp(min=1e-10)).detach()
    return raw_loss * scale_factor


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
