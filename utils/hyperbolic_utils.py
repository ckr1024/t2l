"""
Hyperbolic space operations for Token Merging.

Two improvements over original ToMe:
1. TokenMergerWithAttnHyperspace: dual-path (Euclidean + Hyperbolic) token
   merging with multi-head attention and Möbius addition.
2. contrastive_loss_hyperspace: InfoNCE contrastive loss computed in
   hyperbolic (Poincaré ball) space for semantic binding.
"""

import math
from typing import List

import torch
import torch.nn as nn

MIN_NORM = 1e-15
BALL_EPS = {torch.float16: 4e-3, torch.float32: 1e-5}


# -------------------- Mapping functions --------------------

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
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


def artanh(x):
    return Artanh.apply(x)


def tanh(x):
    return x.clamp(-15, 15).tanh()


class Artan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        dtype = x.dtype
        x = x.double()
        return x.atan().to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 + input ** 2)


def artan(x):
    return Artan.apply(x)


def tan(x):
    return x.clamp(-15, 15).tan()


# -------------------- Euclidean -> Hyperbolic --------------------

def exp_map(u):
    c = 1
    sqrt_c = c ** 0.5
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return project(gamma_1, c)


# -------------------- Hyperbolic -> Euclidean --------------------

def log_map(y):
    c = 1
    sqrt_c = c ** 0.5
    y_norm = y.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)


def project(x, c=1):
    norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    eps = BALL_EPS[x.dtype]
    maxnorm = (1 - eps) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


# -------------------- Möbius addition --------------------

def mobius_addition(x, y):
    c = 1
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / denom.clamp_min(MIN_NORM)


# -------------------- Hyperbolic distance --------------------

def hyperbolic_distance(x, y, c=1, eval_mode=False):
    sqrt_c = c ** 0.5
    mobius_minus_x = -x
    minus_x_oplus_y = mobius_addition(
        mobius_minus_x.to(dtype=torch.float64),
        y.to(dtype=torch.float64),
    )
    minus_x_oplus_y = minus_x_oplus_y.to(dtype=torch.float64)
    if eval_mode:
        pairwise_norm = torch.norm(minus_x_oplus_y, p=2, dim=-1)
    else:
        pairwise_norm = torch.norm(minus_x_oplus_y, p=2, dim=-1, keepdim=True)
        pairwise_norm = pairwise_norm.to(dtype=torch.float64)
    dist = 2 / sqrt_c * (torch.atanh(sqrt_c * pairwise_norm).to(dtype=torch.float64))
    return dist.to(dtype=torch.float64)


# -------------------- Sinusoidal positional encoding --------------------

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_length: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length

        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_length, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        pe_on_device = self.pe.to(position_ids.device)
        return pe_on_device[position_ids]


# -------------------- Token Merging with MHA in Hyperbolic space --------------------

class TokenMergerWithAttnHyperspace(nn.Module):
    def __init__(
        self,
        embed_dim: int = 2048,
        num_heads: int = 8,
        max_length: int = 128,
        c: float = 1.0,
    ):
        super().__init__()
        self.c = c
        self.pos_encoding = SinusoidalPositionalEncoding(embed_dim, max_length)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.alpha = 1.1
        self.beta = 1.2

        # Identity initialization: in a training-free setting the MHA weights
        # must be meaningful without any training. Setting Q/K/V projections
        # to identity makes attention operate on raw cosine similarity of the
        # input token embeddings.
        with torch.no_grad():
            nn.init.eye_(self.multihead_attn.in_proj_weight[:embed_dim])
            nn.init.eye_(self.multihead_attn.in_proj_weight[embed_dim:2 * embed_dim])
            nn.init.eye_(self.multihead_attn.in_proj_weight[2 * embed_dim:])
            nn.init.zeros_(self.multihead_attn.in_proj_bias)
            nn.init.eye_(self.multihead_attn.out_proj.weight)
            nn.init.zeros_(self.multihead_attn.out_proj.bias)

    def forward(
        self, prompt_embeds: torch.Tensor, idx_merge: List[List[List[int]]]
    ) -> torch.Tensor:
        device = self.multihead_attn.in_proj_weight.device
        dtype = self.multihead_attn.in_proj_weight.dtype
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype).unsqueeze(0)

        batch_size, seq_len, embed_dim = prompt_embeds.size()
        pos = torch.arange(seq_len, device=device, dtype=torch.int)
        pos_encoding = self.pos_encoding(pos).to(device=device, dtype=dtype).unsqueeze(0)

        pos_encoding_hyper = exp_map(pos_encoding).to(device=device, dtype=dtype)
        prompt_embeds_hyper = exp_map(prompt_embeds).to(device=device, dtype=dtype)
        prompt_embeds_hyper = mobius_addition(prompt_embeds_hyper, pos_encoding_hyper)

        for idxs in idx_merge:
            noun_indices = idxs[0]
            attr_indices = idxs[1]

            noun_vectors = prompt_embeds_hyper[:, noun_indices, :]
            attr_vectors = prompt_embeds_hyper[:, attr_indices, :]
            noun_vectors_2 = prompt_embeds[:, noun_indices, :]
            attr_vectors_2 = prompt_embeds[:, attr_indices, :]

            attn_output_2, _ = self.multihead_attn(
                query=noun_vectors_2,
                key=attr_vectors_2,
                value=attr_vectors_2,
            )

            attn_output, _ = self.multihead_attn(
                query=noun_vectors,
                key=attr_vectors,
                value=attr_vectors,
            )

            merged_vector = mobius_addition(attn_output, noun_vectors)
            merged_vector_2 = attn_output_2 + noun_vectors_2
            merged_sum = mobius_addition(
                self.alpha * merged_vector.sum(dim=1),
                self.beta * attr_vectors.sum(dim=1),
            )
            merged_sum_2 = (
                self.alpha * merged_vector_2.sum(dim=1)
                + self.beta * attr_vectors_2.sum(dim=1)
            )

            noun_main_idx = noun_indices[0]
            prompt_embeds_hyper[:, noun_main_idx, :] = merged_sum
            prompt_embeds[:, noun_main_idx, :] = merged_sum_2
            if len(noun_indices) > 1:
                prompt_embeds_hyper[:, noun_indices[1:], :] = 0
                prompt_embeds[:, noun_indices[1:], :] = 0
            prompt_embeds_hyper[:, attr_indices, :] = 0
            prompt_embeds[:, attr_indices, :] = 0

        prompt_embeds_hyper = log_map(prompt_embeds_hyper)
        prompt_embeds_hyper = prompt_embeds_hyper.squeeze(0)
        prompt_embeds = prompt_embeds.squeeze(0)
        return prompt_embeds + 0.1 * prompt_embeds_hyper


# -------------------- Contrastive loss in Hyperbolic space --------------------

def contrastive_loss_hyperspace(
    query_vec: torch.Tensor,
    target_anchor: torch.Tensor,
    other_anchors: List[torch.Tensor],
    temp: float = 0.07,
) -> torch.Tensor:
    """InfoNCE contrastive loss computed in the Poincaré ball."""
    device = query_vec.device
    query_vec = exp_map(query_vec)
    target_anchor = exp_map(target_anchor)
    other_anchors = [exp_map(anchor) for anchor in other_anchors]
    if len(other_anchors) == 0:
        return torch.tensor(0.0, device=device)
    dist_pos = hyperbolic_distance(query_vec, target_anchor, c=1)
    dist_neg_list = []
    for n_vec in other_anchors:
        dist_neg_list.append(hyperbolic_distance(query_vec, n_vec, c=1))
    dist_neg = torch.stack(dist_neg_list, dim=0)
    numerator = torch.exp(-dist_pos / temp)
    denominator = numerator + torch.exp(-dist_neg / temp).sum(dim=0)
    loss_cont = -torch.log(numerator / (denominator + 1e-8) + 1e-8)
    return loss_cont.mean()
