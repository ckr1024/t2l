import torch
import torch.nn as nn
import math

MIN_NORM = 1e-15
BALL_EPS = {torch.float16: 4e-3, torch.float32: 1e-5, torch.float64: 1e-10}


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
    norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    eps = BALL_EPS.get(x.dtype, 1e-5)
    maxnorm = (1 - eps) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def exp_map(u, c=1):
    """Euclidean -> Poincaré ball (exponential map at the origin)."""
    sqrt_c = c ** 0.5
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return project(gamma_1, c)


def log_map(y, c=1):
    """Poincaré ball -> Euclidean (logarithmic map at the origin)."""
    sqrt_c = c ** 0.5
    y_norm = y.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)


def mobius_addition(x, y, c=1):
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / denom.clamp_min(MIN_NORM)


def hyperbolic_distance(x, y, c=1):
    sqrt_c = c ** 0.5
    neg_x = -x
    diff = mobius_addition(
        neg_x.to(dtype=torch.float64), y.to(dtype=torch.float64), c
    )
    diff_norm = diff.norm(dim=-1, p=2, keepdim=True).to(dtype=torch.float64)
    diff_norm = diff_norm.clamp_min(MIN_NORM)
    safe_arg = (sqrt_c * diff_norm).clamp(max=(1 - 1e-5) / sqrt_c)
    dist = 2.0 / sqrt_c * torch.atanh(safe_arg)
    return dist.to(dtype=torch.float64)


def contrastive_loss_hyperspace(
    query_vec: torch.Tensor,
    target_anchor: torch.Tensor,
    other_anchors: list,
    temp: float = 0.07,
) -> torch.Tensor:
    """Contrastive loss in hyperbolic space between query and positive/negative anchors."""
    device = query_vec.device
    orig_dtype = query_vec.dtype

    query_h = exp_map(query_vec.float())
    target_h = exp_map(target_anchor.float())
    others_h = [exp_map(a.float()) for a in other_anchors]

    if len(others_h) == 0:
        return torch.tensor(0.0, device=device, dtype=orig_dtype)

    dist_pos = hyperbolic_distance(query_h, target_h, c=1)
    dist_neg = torch.stack(
        [hyperbolic_distance(query_h, n, c=1) for n in others_h], dim=0
    )

    numerator = torch.exp(-dist_pos / temp)
    denominator = numerator + torch.exp(-dist_neg / temp).sum(dim=0)
    loss = -torch.log(numerator / (denominator + 1e-8) + 1e-8)
    return loss.mean().to(orig_dtype)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_length: int = 128):
        super().__init__()
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_length, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.pe.to(position_ids.device)[position_ids]


class TokenMergerWithAttnHyperspace(nn.Module):
    """
    Hyperbolic-space token merger: maps embeddings to the Poincaré ball,
    applies positional encoding via Möbius addition, merges with multi-head
    attention in both hyperbolic and Euclidean spaces, and fuses the results.
    """

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

    def forward(
        self, prompt_embeds: torch.Tensor, idx_merge: list
    ) -> torch.Tensor:
        """
        Args:
            prompt_embeds: (seq_len, embed_dim) — single prompt embedding.
            idx_merge: list of [[noun_indices], [attr_indices]].
        Returns:
            (seq_len, embed_dim) — merged embedding (Euclidean + 0.1 * Hyperbolic).
        """
        device = self.multihead_attn.in_proj_weight.device
        input_dtype = prompt_embeds.dtype
        work_dtype = torch.float32

        prompt_embeds = prompt_embeds.to(device=device, dtype=work_dtype).unsqueeze(0)
        batch_size, seq_len, embed_dim = prompt_embeds.size()

        pos = torch.arange(seq_len, device=device, dtype=torch.int)
        pos_enc = (
            self.pos_encoding(pos).to(device=device, dtype=work_dtype).unsqueeze(0)
        )

        pos_enc_hyper = exp_map(pos_enc, self.c).to(device=device, dtype=work_dtype)
        prompt_embeds_hyper = exp_map(prompt_embeds, self.c).to(
            device=device, dtype=work_dtype
        )
        prompt_embeds_hyper = mobius_addition(
            prompt_embeds_hyper, pos_enc_hyper, self.c
        )

        self.multihead_attn.to(device=device, dtype=work_dtype)

        for idxs in idx_merge:
            noun_indices = idxs[0]
            attr_indices = idxs[1]

            noun_h = prompt_embeds_hyper[:, noun_indices, :]
            attr_h = prompt_embeds_hyper[:, attr_indices, :]
            noun_e = prompt_embeds[:, noun_indices, :]
            attr_e = prompt_embeds[:, attr_indices, :]

            attn_out_e, _ = self.multihead_attn(
                query=noun_e, key=attr_e, value=attr_e
            )
            attn_out_h, _ = self.multihead_attn(
                query=noun_h, key=attr_h, value=attr_h
            )

            merged_h = mobius_addition(attn_out_h, noun_h, self.c)
            merged_sum_h = mobius_addition(
                self.alpha * merged_h.sum(dim=1),
                self.beta * attr_h.sum(dim=1),
                self.c,
            )

            merged_e = attn_out_e + noun_e
            merged_sum_e = (
                self.alpha * merged_e.sum(dim=1) + self.beta * attr_e.sum(dim=1)
            )

            noun_main = noun_indices[0]
            prompt_embeds_hyper[:, noun_main, :] = merged_sum_h
            prompt_embeds[:, noun_main, :] = merged_sum_e

            if len(noun_indices) > 1:
                prompt_embeds_hyper[:, noun_indices[1:], :] = 0
                prompt_embeds[:, noun_indices[1:], :] = 0
            prompt_embeds_hyper[:, attr_indices, :] = 0
            prompt_embeds[:, attr_indices, :] = 0

        prompt_embeds_hyper = log_map(prompt_embeds_hyper, self.c)

        result = prompt_embeds.squeeze(0) + 0.1 * prompt_embeds_hyper.squeeze(0)
        return result.to(input_dtype)
