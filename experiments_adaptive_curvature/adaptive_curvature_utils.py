"""
Adaptive Curvature Estimation for Hyperbolic Token Merging
==========================================================
Instead of a fixed curvature c=1.0 for all prompts, dynamically estimate
c based on prompt complexity (number of entities, syntactic depth,
attribute count). Training-free — uses spaCy parse tree statistics.

Two strategies:
  1. Per-prompt curvature: one c for the entire prompt
  2. Per-entity curvature: different c for each (noun, attributes) pair
"""

import torch
from typing import List, Tuple, Optional
from utils.hyperbolic_utils import (
    exp_map,
    log_map,
    mobius_addition,
    hyperbolic_distance,
)


def mobius_scalar_mult(scalar: float, x: torch.Tensor, c: float) -> torch.Tensor:
    """Hyperbolic scalar multiplication on the Poincaré ball.

    Implemented via:
      a ⊗ x = exp_0( a * log_0(x) )
    """
    return exp_map(scalar * log_map(x, c), c)


# ============================================================
# Strategy 1: Per-prompt adaptive curvature
# ============================================================

def estimate_prompt_curvature(
    doc,
    base_c: float = 0.5,
    entity_weight: float = 0.3,
    depth_weight: float = 0.2,
    attr_weight: float = 0.25,
    c_max: float = 5.0,
) -> float:
    """Estimate Poincaré ball curvature from spaCy parse tree.

    Higher curvature = more "room" for hierarchical separation.
    Complex prompts with many entities/attributes need higher c.
    """
    noun_chunks = [
        chunk for chunk in doc.noun_chunks
        if chunk.text not in ("top", "the side", "the left", "the right")
    ]
    num_entities = max(len(noun_chunks), 1)

    max_depth = 0
    for token in doc:
        depth = len(list(token.ancestors))
        max_depth = max(max_depth, depth)

    attr_deps = {"amod", "prep", "relcl", "acl", "advmod", "nummod"}
    num_attrs = sum(1 for t in doc if t.dep_ in attr_deps)

    complexity = (
        entity_weight * num_entities
        + depth_weight * max_depth
        + attr_weight * num_attrs
    )
    c = base_c * (1.0 + 0.5 * complexity)
    return min(c, c_max)


# ============================================================
# Strategy 2: Per-entity adaptive curvature
# ============================================================

def estimate_entity_curvatures(
    doc,
    token_indices: list,
    base_c: float = 0.5,
    c_max: float = 5.0,
) -> List[float]:
    """Assign different curvature to each entity based on its attribute count.

    Entities with more attributes need higher curvature for better
    hierarchical separation in the Poincaré ball.
    """
    curvatures = []
    for idxs in token_indices:
        num_attrs = len(idxs[1])
        c = base_c + 0.4 * num_attrs
        curvatures.append(min(c, c_max))
    return curvatures


# ============================================================
# Adaptive Token Merge (per-entity curvature)
# ============================================================

def token_merge_adaptive(
    prompt_embeds: torch.Tensor,
    idx_merge: list,
    curvatures: List[float],
) -> torch.Tensor:
    """Token merging where each entity uses its own curvature.

    For entity k with curvature c_k:
      1. Map noun/attr tokens to Poincaré ball B_{c_k}
      2. Aggregate via Möbius addition in B_{c_k}
      3. Map back to Euclidean
    """
    for entity_idx, idxs in enumerate(idx_merge):
        c = curvatures[entity_idx] if entity_idx < len(curvatures) else 1.0
        noun_idx = idxs[0][0]

        noun_tokens = prompt_embeds[idxs[0]]
        attr_tokens = prompt_embeds[idxs[1]]

        all_tokens = torch.cat([noun_tokens, attr_tokens], dim=0)
        scale = all_tokens.norm(dim=-1).max().clamp(min=1.0).item()

        noun_scaled = noun_tokens / scale
        attr_scaled = attr_tokens / scale

        noun_hyp = exp_map(noun_scaled, c)
        attr_hyp = exp_map(attr_scaled, c)

        result = mobius_scalar_mult(1.1, noun_hyp[0:1], c)
        for i in range(1, noun_hyp.shape[0]):
            next_tok = mobius_scalar_mult(1.1, noun_hyp[i:i+1], c)
            result = mobius_addition(result, next_tok, c)

        for i in range(attr_hyp.shape[0]):
            next_tok = mobius_scalar_mult(1.2, attr_hyp[i:i+1], c)
            result = mobius_addition(result, next_tok, c)

        composite = log_map(result, c) * scale

        prompt_embeds[noun_idx] = composite.squeeze(0)
        if len(idxs[0]) > 1:
            prompt_embeds[idxs[0][1:]] = 0
        prompt_embeds[idxs[1]] = 0

    return prompt_embeds
