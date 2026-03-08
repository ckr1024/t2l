"""
Hyperbolic geometry operations for Token Merging enhancement.

Two core components:
  1. TokenMergerWithAttnHyperspace: Multi-head attention based token merging
     with Mobius addition in the Poincare ball, blended with Euclidean merging.
  2. contrastive_loss_hyperspace: InfoNCE contrastive loss using hyperbolic
     distances for semantic binding optimization.

Poincare ball operations use:
  - Custom Artanh autograd for stable backward pass
  - float64 casting in distance computation to avoid NaN
  - Per-dtype epsilon for ball projection

References:
  - Nickel & Kiela, "Poincare Embeddings", NeurIPS 2017
  - Ganea et al., "Hyperbolic Neural Networks", NeurIPS 2018
"""

import math
import torch
import torch.nn as nn
from typing import List

MIN_NORM = 1e-15
BALL_EPS = {torch.float16: 4e-3, torch.float32: 1e-5, torch.float64: 1e-10}


# ============================================================
# Numerically Stable Poincare Ball Primitives
# ============================================================

class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        dtype = x.dtype
        x = x.double()
        return (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5).to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


def artanh(x):
    return Artanh.apply(x)


def tanh(x):
    return x.clamp(-15, 15).tanh()


def project(x, c=1):
    """Project onto the open Poincare ball with per-dtype epsilon."""
    norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    eps = BALL_EPS.get(x.dtype, 1e-5)
    maxnorm = (1 - eps) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def exp_map(u, c=1):
    """Exponential map at the origin of the Poincare ball (Euclidean -> Hyperbolic)."""
    sqrt_c = c ** 0.5
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return project(gamma_1, c)


def log_map(y, c=1):
    """Logarithmic map at the origin of the Poincare ball (Hyperbolic -> Euclidean)."""
    sqrt_c = c ** 0.5
    y_norm = y.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)


def mobius_addition(x, y, c=1):
    """Mobius addition in the Poincare ball: x oplus_c y."""
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / denom.clamp_min(MIN_NORM)


def hyperbolic_distance(x, y, c=1):
    """
    Geodesic distance in the Poincare ball.
    Uses float64 internally to avoid NaN from atanh near boundary.
    """
    sqrt_c = c ** 0.5
    minus_x_oplus_y = mobius_addition(
        (-x).to(dtype=torch.float64), y.to(dtype=torch.float64), c
    )
    pairwise_norm = torch.norm(minus_x_oplus_y, p=2, dim=-1, keepdim=True).to(torch.float64)
    dist = 2 / sqrt_c * artanh(sqrt_c * pairwise_norm).to(torch.float64)
    return dist


# ============================================================
# Sinusoidal Positional Encoding for Prompt Embeddings
# ============================================================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_length: int = 128):
        super().__init__()
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_length, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.pe.to(position_ids.device)[position_ids]


# ============================================================
# Token Merging with Multi-Head Attention in Hyperbolic Space
# ============================================================

class TokenMergerWithAttnHyperspace(nn.Module):
    """
    Dual-path token merging:
      - Euclidean path: standard MHA + addition
      - Hyperbolic path: MHA on Poincare embeddings + Mobius addition
      - Final: euclidean + 0.1 * hyperbolic (gentle correction)

    Positional encoding is added via Mobius addition in hyperbolic space,
    capturing hierarchical structure of noun-attribute relationships.
    """

    def __init__(self, embed_dim: int = 2048, num_heads: int = 8,
                 max_length: int = 128, c: float = 1.0):
        super().__init__()
        self.c = c
        self.pos_encoding = SinusoidalPositionalEncoding(embed_dim, max_length)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.alpha = 1.1
        self.beta = 1.2

    def forward(self, prompt_embeds: torch.Tensor,
                idx_merge: List[List[List[int]]]) -> torch.Tensor:
        device = self.multihead_attn.in_proj_weight.device
        dtype = self.multihead_attn.in_proj_weight.dtype
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)

        needs_squeeze = prompt_embeds.dim() == 2
        if needs_squeeze:
            prompt_embeds = prompt_embeds.unsqueeze(0)

        batch_size, seq_len, embed_dim = prompt_embeds.size()
        pos = torch.arange(seq_len, device=device, dtype=torch.int)
        pos_enc = self.pos_encoding(pos).to(device=device, dtype=dtype).unsqueeze(0)

        # Hyperbolic path: positional encoding via Mobius addition
        pos_enc_hyp = exp_map(pos_enc, self.c).to(device=device, dtype=dtype)
        prompt_hyp = exp_map(prompt_embeds, self.c).to(device=device, dtype=dtype)
        prompt_hyp = mobius_addition(prompt_hyp, pos_enc_hyp, self.c)

        for idxs in idx_merge:
            noun_indices = idxs[0]
            attr_indices = idxs[1]

            # Euclidean attention
            noun_e = prompt_embeds[:, noun_indices, :]
            attr_e = prompt_embeds[:, attr_indices, :]
            attn_out_e, _ = self.multihead_attn(query=noun_e, key=attr_e, value=attr_e)
            merged_e = attn_out_e + noun_e
            merged_sum_e = self.alpha * merged_e.sum(dim=1) + self.beta * attr_e.sum(dim=1)

            # Hyperbolic attention
            noun_h = prompt_hyp[:, noun_indices, :]
            attr_h = prompt_hyp[:, attr_indices, :]
            attn_out_h, _ = self.multihead_attn(query=noun_h, key=attr_h, value=attr_h)
            merged_h = mobius_addition(attn_out_h, noun_h, self.c)
            merged_sum_h = mobius_addition(
                self.alpha * merged_h.sum(dim=1),
                self.beta * attr_h.sum(dim=1),
                self.c,
            )

            noun_main = noun_indices[0]
            prompt_embeds[:, noun_main, :] = merged_sum_e
            prompt_hyp[:, noun_main, :] = merged_sum_h
            if len(noun_indices) > 1:
                prompt_embeds[:, noun_indices[1:], :] = 0
                prompt_hyp[:, noun_indices[1:], :] = 0
            prompt_embeds[:, attr_indices, :] = 0
            prompt_hyp[:, attr_indices, :] = 0

        prompt_hyp = log_map(prompt_hyp, self.c)
        result = prompt_embeds + 0.1 * prompt_hyp.to(dtype=dtype)

        if needs_squeeze:
            result = result.squeeze(0)
        return result


# ============================================================
# Contrastive Loss in Hyperbolic Space (replaces MSE binding loss)
# ============================================================

def contrastive_loss_hyperspace(
    query_vec: torch.Tensor,
    target_anchor: torch.Tensor,
    other_anchors: List[torch.Tensor],
    temp: float = 0.07,
    c: float = 1.0,
) -> torch.Tensor:
    """
    InfoNCE contrastive loss in the Poincare ball.

    Args:
        query_vec: Noise prediction from composite token (B, C, H, W)
        target_anchor: Noise prediction from target phrase (positive)
        other_anchors: Noise predictions from other phrases (negatives)
        temp: Temperature for contrastive scaling
        c: Poincare ball curvature

    The loss encourages the composite token's prediction to be close
    to its target phrase and far from other phrases in hyperbolic space.
    """
    device = query_vec.device
    if len(other_anchors) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    query_h = exp_map(query_vec, c)
    target_h = exp_map(target_anchor, c)
    others_h = [exp_map(a, c) for a in other_anchors]

    dist_pos = hyperbolic_distance(query_h, target_h, c)
    dist_neg = torch.stack(
        [hyperbolic_distance(query_h, n, c) for n in others_h], dim=0
    )

    numerator = torch.exp(-dist_pos / temp)
    denominator = numerator + torch.exp(-dist_neg / temp).sum(dim=0)
    loss = -torch.log(numerator / (denominator + 1e-8) + 1e-8)

    return loss.mean()


# ============================================================
# Entropy Loss (kept Euclidean — attention maps are spatial, not hierarchical)
# ============================================================

def compute_hyperbolic_entropy_loss(
    cross_map: torch.Tensor, c: float = 1.0
) -> torch.Tensor:
    """Standard Shannon entropy loss for attention concentration.

    Kept Euclidean because attention maps represent spatial pixel distributions,
    not hierarchical relationships that benefit from hyperbolic geometry.
    The -2.0 scaling matches the original ToMe paper.
    """
    return -2.0 * (cross_map * torch.log(cross_map + 1e-5)).sum()


def compute_hyperbolic_pose_loss(
    pair_pos: torch.Tensor,
    pos1: torch.Tensor,
    pos2: torch.Tensor,
    grid_size: float = 32.0,
    c: float = 1.0,
) -> torch.Tensor:
    """Position loss using hyperbolic geodesic distance."""
    scale = grid_size
    p0_h = exp_map((pair_pos[0] / scale).unsqueeze(0), c)
    p1_h = exp_map((pair_pos[1] / scale).unsqueeze(0), c)
    t0_h = exp_map((pos1 / scale).unsqueeze(0), c)
    t1_h = exp_map((pos2 / scale).unsqueeze(0), c)
    loss = 0.2 * hyperbolic_distance(p0_h, t0_h, c).pow(2).mean()
    loss = loss + 0.2 * hyperbolic_distance(p1_h, t1_h, c).pow(2).mean()
    return loss


# ============================================================
# Legacy API compatibility (for non-hyperbolic token merge path)
# ============================================================

def token_merge_hyperbolic(
    prompt_embeds: torch.Tensor,
    idx_merge: List[List[int]],
    c: float = 1.0,
) -> torch.Tensor:
    """Fallback: simple weighted sum (used when TokenMergerWithAttnHyperspace
    is not available, e.g., in configs without the module initialized)."""
    for idxs in idx_merge:
        noun_idx = idxs[0][0]
        noun_tokens = prompt_embeds[idxs[0]]
        attr_tokens = prompt_embeds[idxs[1]]
        composite = 1.1 * noun_tokens.sum(dim=0) + 1.2 * attr_tokens.sum(dim=0)
        prompt_embeds[noun_idx] = composite
        if len(idxs[0]) > 1:
            prompt_embeds[idxs[0][1:]] = 0
        prompt_embeds[idxs[1]] = 0
    return prompt_embeds


def hyperbolic_spatial_loss(
    pred: torch.Tensor, target: torch.Tensor, c: float = 1.0
) -> torch.Tensor:
    """Legacy fallback: standard MSE (used when contrastive loss not available)."""
    return torch.nn.functional.mse_loss(pred, target)
