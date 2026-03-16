"""
Video Token Merging Utilities
=============================
Adapts the ToMe semantic-binding pipeline to text-to-video generation.

Architecture context:
  Video diffusion models (AnimateDiff, ModelScopeT2V) extend the image UNet
  by inserting *temporal attention* layers between spatial attention blocks.
  The text conditioning path (CLIP encoder → text embeddings → cross-attention)
  is structurally identical to text-to-image. This means:

    1. Token Merging (Sec 3.2.1) applies DIRECTLY to text embeddings
       — same code as T2I, no modification needed.
    2. End Token Substitution — same, operates on text embeddings.
    3. Semantic Binding Loss — same, compares noise predictions.
    4. Entropy Loss — needs adaptation: attention maps are per-frame,
       we aggregate across frames for consistency.
    5. NEW: Temporal Consistency Loss — encourages the composite-token's
       cross-attention map to be stable across frames.

This module provides:
  - token_merge_video(): applies standard ToMe token merging to video embeddings
  - temporal_consistency_loss(): L_temporal = Σ_t d(A_t, A_{t+1})²
  - aggregate_video_attention(): collects cross-attention across frames
"""

import torch
import torch.nn.functional as F
from typing import List, Optional


def token_merge_video(
    prompt_embeds: torch.Tensor,
    idx_merge: list,
    use_hyperbolic: bool = False,
    curvature: float = 1.0,
) -> torch.Tensor:
    """Apply token merging to video text embeddings.

    Video pipelines use the exact same text embedding format as image pipelines
    (shape: [batch, seq_len, dim]). This is a thin wrapper that dispatches
    to the appropriate merging function.
    """
    if use_hyperbolic:
        from utils.hyperbolic_utils import token_merge_hyperbolic
        return token_merge_hyperbolic(prompt_embeds, idx_merge, curvature)
    else:
        from pipe_tome import token_merge
        return token_merge(prompt_embeds, idx_merge)


def temporal_attention_entropy_loss(
    frame_attention_maps: List[torch.Tensor],
    entity_indices: List[int],
) -> torch.Tensor:
    """Entropy loss averaged across video frames.

    Args:
        frame_attention_maps: list of (H, W, seq_len) attention maps, one per frame
        entity_indices: token indices of composite tokens to regularize

    Returns:
        Scalar entropy loss aggregated over frames and entities.
    """
    loss = torch.tensor(0.0, device=frame_attention_maps[0].device)
    for attn_map in frame_attention_maps:
        cross_map = attn_map[:, :, entity_indices]
        cross_map = (cross_map - cross_map.amin(dim=(0, 1), keepdim=True)) / (
            cross_map.amax(dim=(0, 1), keepdim=True)
            - cross_map.amin(dim=(0, 1), keepdim=True)
            + 1e-8
        )
        cross_map = cross_map / (cross_map.sum(dim=(0, 1), keepdim=True) + 1e-8)
        loss = loss - 2 * (cross_map * torch.log(cross_map + 1e-5)).sum()
    return loss / max(len(frame_attention_maps), 1)


def temporal_consistency_loss(
    frame_attention_maps: List[torch.Tensor],
    entity_indices: List[int],
    use_hyperbolic: bool = False,
    curvature: float = 1.0,
) -> torch.Tensor:
    """Encourage cross-attention maps to be consistent across adjacent frames.

    L_temporal = (1/T-1) Σ_{t=0}^{T-2} ||A_t - A_{t+1}||² (Euclidean)
             or  (1/T-1) Σ_{t=0}^{T-2} d_H(A_t, A_{t+1})² (Hyperbolic)

    Args:
        frame_attention_maps: list of (H, W, seq_len) per-frame attention
        entity_indices: composite-token indices to compare
        use_hyperbolic: whether to use geodesic distance
        curvature: Poincaré ball curvature for hyperbolic mode
    """
    if len(frame_attention_maps) < 2:
        return torch.tensor(0.0, device=frame_attention_maps[0].device)

    loss = torch.tensor(0.0, device=frame_attention_maps[0].device)
    for t in range(len(frame_attention_maps) - 1):
        map_t = frame_attention_maps[t][:, :, entity_indices]      # (H, W, K)
        map_next = frame_attention_maps[t + 1][:, :, entity_indices]

        if use_hyperbolic:
            from utils.hyperbolic_utils import exp_map_zero, hyperbolic_distance
            flat_t = map_t.reshape(-1, len(entity_indices))
            flat_next = map_next.reshape(-1, len(entity_indices))
            scale = torch.cat([flat_t, flat_next]).abs().max().clamp(min=1.0)
            hyp_t = exp_map_zero(flat_t / scale, curvature)
            hyp_next = exp_map_zero(flat_next / scale, curvature)
            dist = hyperbolic_distance(hyp_t, hyp_next, curvature)
            loss = loss + dist.pow(2).mean()
        else:
            loss = loss + F.mse_loss(map_t, map_next)

    return loss / (len(frame_attention_maps) - 1)


def compute_video_semantic_binding_loss(
    noise_pred_anchor: torch.Tensor,
    noise_pred_token: torch.Tensor,
    use_hyperbolic: bool = False,
    curvature: float = 1.0,
) -> torch.Tensor:
    """Semantic binding loss for video noise predictions.

    Video noise predictions have shape (B, F, C, H, W) or (B*F, C, H, W).
    We treat each frame independently and average.
    """
    if noise_pred_anchor.dim() == 5:
        B, F, C, H, W = noise_pred_anchor.shape
        noise_pred_anchor = noise_pred_anchor.reshape(B * F, C, H, W)
        noise_pred_token = noise_pred_token.reshape(B * F, C, H, W)

    if use_hyperbolic:
        from utils.hyperbolic_utils import hyperbolic_spatial_loss
        return hyperbolic_spatial_loss(noise_pred_anchor, noise_pred_token, curvature)
    else:
        return F.mse_loss(noise_pred_anchor, noise_pred_token)
