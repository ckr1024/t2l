"""
Audio Token Merging Utilities
=============================
Adapts ToMe's semantic-binding approach to text-to-audio generation.

Architecture context:
  Audio diffusion models (AudioLDM2, MusicGen-style, Tango) share the same
  fundamental structure as image diffusion:
    - Text encoder (CLAP / Flan-T5 / GPT-2) → text embeddings
    - UNet denoiser operating on mel-spectrogram latents
    - Cross-attention between audio latents and text embeddings
    - VAE decoder: latent → mel-spectrogram → waveform (via vocoder)

  The cross-attention mechanism is structurally identical to T2I.
  Therefore, ToMe's three techniques apply naturally:
    1. Token Merging: fuse "sound source + attribute" tokens
    2. End Token Substitution: clean up CLAP/T5 EOT tokens
    3. Semantic Binding Loss: align noise predictions for composite tokens

Audio-specific considerations:
  - Audio prompts describe sound sources and their attributes:
    "a loud drum and a soft violin" → drum* = drum + loud, violin* = violin + soft
  - Cross-attention maps are 1D (time) instead of 2D (spatial),
    but entropy regularization applies identically.
  - The "layout" in audio = temporal arrangement of sounds.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional


def token_merge_audio(
    prompt_embeds: torch.Tensor,
    idx_merge: list,
    use_hyperbolic: bool = False,
    curvature: float = 1.0,
) -> torch.Tensor:
    """Apply token merging to audio text embeddings.

    Audio text encoders (CLAP, Flan-T5) produce embeddings of shape
    (batch, seq_len, dim), identical to CLIP. Token merging applies directly.
    """
    if use_hyperbolic:
        from utils.hyperbolic_utils import token_merge_hyperbolic
        return token_merge_hyperbolic(prompt_embeds, idx_merge, curvature)
    else:
        from pipe_tome import token_merge
        return token_merge(prompt_embeds, idx_merge)


def audio_entropy_loss(
    cross_attention_map: torch.Tensor,
    entity_indices: List[int],
) -> torch.Tensor:
    """Entropy loss for audio cross-attention maps.

    Audio attention maps are typically 1D (time × seq_len) or
    2D (freq × time × seq_len) for mel-spectrogram latents.
    We flatten spatial dims and compute per-entity Shannon entropy.

    Args:
        cross_attention_map: (..., seq_len) attention weights
        entity_indices: indices of composite tokens

    Returns:
        Scalar entropy loss.
    """
    if cross_attention_map.dim() == 3:
        # (freq, time, seq_len) → flatten to (freq*time, seq_len)
        H, W, S = cross_attention_map.shape
        attn = cross_attention_map.reshape(-1, S)
    elif cross_attention_map.dim() == 2:
        attn = cross_attention_map
    else:
        attn = cross_attention_map.reshape(-1, cross_attention_map.shape[-1])

    cross_map = attn[:, entity_indices]  # (N, K)
    cross_map = (cross_map - cross_map.amin(dim=0, keepdim=True)) / (
        cross_map.amax(dim=0, keepdim=True)
        - cross_map.amin(dim=0, keepdim=True) + 1e-8
    )
    cross_map = cross_map / (cross_map.sum(dim=0, keepdim=True) + 1e-8)

    entropy = -(cross_map * torch.log(cross_map + 1e-5)).sum()
    return -2 * entropy


def audio_semantic_binding_loss(
    noise_pred_anchor: torch.Tensor,
    noise_pred_token: torch.Tensor,
    use_hyperbolic: bool = False,
    curvature: float = 1.0,
) -> torch.Tensor:
    """Semantic binding loss for audio noise predictions.

    Audio latents have shape (B, C, T) or (B, C, F, T) depending on the model.
    For AudioLDM2: (B, 8, freq, time) — 2D mel-spectrogram latent.
    """
    if use_hyperbolic:
        if noise_pred_anchor.dim() == 3:
            # (B, C, T) → treat as (B, C, 1, T) for the spatial loss
            noise_pred_anchor = noise_pred_anchor.unsqueeze(2)
            noise_pred_token = noise_pred_token.unsqueeze(2)
        from utils.hyperbolic_utils import hyperbolic_spatial_loss
        return hyperbolic_spatial_loss(noise_pred_anchor, noise_pred_token, curvature)
    else:
        return F.mse_loss(noise_pred_anchor, noise_pred_token)


def parse_audio_prompt(prompt: str, tokenizer) -> tuple:
    """Parse an audio prompt for token merging.

    Expected patterns:
      "a loud drum and a soft violin"
      "a barking dog and a meowing cat"

    Returns (idx_merge, prompt_anchor) or empty lists if parsing fails.
    """
    words = prompt.lower().split()
    if "and" not in words:
        return [], []

    and_pos = words.index("and")

    pos = 1
    word_positions = []
    for word in words:
        ids = tokenizer.encode(word)
        n_tokens = len(ids) - 2
        n_tokens = max(n_tokens, 1)
        word_positions.append(list(range(pos, pos + n_tokens)))
        pos += n_tokens

    # Entity 1: find first content word after determiner
    noun1_idx, attrs1_idx = None, []
    for i in range(and_pos):
        if words[i] in ("a", "an", "the"):
            if i + 1 < and_pos:
                # Last word before "and" (or before next determiner) is likely the noun
                # Attributes are everything between determiner and noun
                noun_candidates = list(range(i + 1, and_pos))
                if noun_candidates:
                    noun1_idx = noun_candidates[-1]
                    attrs1_idx = noun_candidates[:-1]
                break

    # Entity 2
    noun2_idx, attrs2_idx = None, []
    for i in range(and_pos + 1, len(words)):
        if words[i] in ("a", "an", "the"):
            if i + 1 < len(words):
                noun_candidates = list(range(i + 1, len(words)))
                if noun_candidates:
                    noun2_idx = noun_candidates[-1]
                    attrs2_idx = noun_candidates[:-1]
                break

    idx_merge = []
    prompt_anchor = []

    if noun1_idx is not None and attrs1_idx:
        noun_pos = word_positions[noun1_idx]
        attr_pos = []
        for ai in attrs1_idx:
            attr_pos.extend(word_positions[ai])
        if attr_pos:
            idx_merge.append([noun_pos, attr_pos])
            anchor = " ".join(words[:and_pos])
            prompt_anchor.append(anchor)

    if noun2_idx is not None and attrs2_idx:
        noun_pos = word_positions[noun2_idx]
        attr_pos = []
        for ai in attrs2_idx:
            attr_pos.extend(word_positions[ai])
        if attr_pos:
            idx_merge.append([noun_pos, attr_pos])
            anchor = " ".join(words[and_pos + 1:])
            prompt_anchor.append(anchor)

    return idx_merge, prompt_anchor
