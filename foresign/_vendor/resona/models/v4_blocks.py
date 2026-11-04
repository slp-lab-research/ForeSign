"""DSTformer building blocks for RESONA v4 (MAE-asymmetric encoder).

Source: ported with minor adaptation from MotionBERT
  reference: references_sota/MotionBERT/lib/model/DSTformer.py (NeurIPS 2022)

Key components:
  - Attention (single-axis: spatial OR temporal)
  - Block (dual mode: 'st' = Spatial→Temporal sequential; 'ts' = reverse)
  - DSTformerBody: stacks Block_st and Block_ts in PARALLEL with learned
    per-layer α-fusion. This is the "dual-stream" design that MotionBERT showed
    converges faster and gives stronger representations than sequential factored.

Differences from MotionBERT original:
  - Uses RMSNorm by default (LLaMA-style) instead of LayerNorm
  - Uses SwiGLU FFN instead of plain MLP
  - Operates on tokens of shape (BF, N, dim) where BF = B*T_patch, N = V (per-frame V joints)
    The temporal axis is folded into the batch dim via seqlen argument
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from typing import Optional


# ---------------------------------------------------------------------------
# Norms / activations (LLaMA-style, matches our v3 conventions)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        scale = var.add(self.eps).rsqrt().to(in_dtype)
        return x * scale * self.weight.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, ffn_dim, bias=False)
        self.w_up = nn.Linear(d_model, ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(self.dropout(F.silu(self.w_gate(x)) * self.w_up(x)))


# ---------------------------------------------------------------------------
# Drop path (stochastic depth, from timm)
# ---------------------------------------------------------------------------

def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False) -> torch.Tensor:
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# ---------------------------------------------------------------------------
# Attention (single-axis)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Single-axis attention. Operates on tokens of shape (BF, N, dim).

    When mode='spatial' (over V=joints): treats BF as batch, attention along N.
    When mode='temporal' (over T=frames): reshapes (BF, N, dim) where BF = B*N
                                          and the actual temporal axis L is N.
    """

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 attn_drop: float = 0., proj_drop: float = 0., mode: str = "spatial"):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by heads {num_heads}"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_p = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mode = mode

    def forward(self, x: torch.Tensor, seqlen: int = 1) -> torch.Tensor:
        # x: (BF, N, dim)  — where BF = B * F (or B * V if temporal)
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (BF, H, N, d_head)

        if self.mode == "spatial":
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop_p if self.training else 0.0, is_causal=False
            )
            out = out.transpose(1, 2).reshape(B_, N, C)
        elif self.mode == "temporal":
            # q,k,v: (BF, H, N, d_head). Reshape so attention is along T dimension.
            # If x represents (B*F frames, V joints, dim), we need to attend ACROSS frames
            # for each (B, V) pair. So we reshape to (B*V, H, F, d_head).
            # caller has already arranged so that BF = B*F and N = V (or vice versa).
            # mode='temporal' here just means we permute differently.
            d_head = C // self.num_heads
            B = B_ // seqlen
            # q,k,v: (BF=B*F, H, N, d_head). We must attend ACROSS F frames for each
            # (B, head, joint). Fold to (B, F, H, N, d) → permute to (B, H, N, F, d),
            # then MAKE CONTIGUOUS before collapsing so the B/H/N axes don't scramble
            # into each other. (The previous reshape skipped .contiguous() and mixed the
            # N axis into the leading and head dims for num_heads>1 — verified bug.)
            def _fold(t):
                return (t.reshape(B, seqlen, self.num_heads, N, d_head)
                         .permute(0, 2, 3, 1, 4)            # (B, H, N, F, d)
                         .contiguous()
                         .reshape(B * self.num_heads * N, seqlen, d_head))
            qt, kt, vt = _fold(q), _fold(k), _fold(v)
            out = F.scaled_dot_product_attention(
                qt, kt, vt, dropout_p=self.attn_drop_p if self.training else 0.0, is_causal=False
            )  # (B*H*N, F, d)
            out = (out.reshape(B, self.num_heads, N, seqlen, d_head)
                      .permute(0, 3, 1, 2, 4)              # (B, F, H, N, d)
                      .contiguous()
                      .reshape(B_, self.num_heads, N, d_head))
            out = out.transpose(1, 2).reshape(B_, N, C)
        else:
            raise ValueError(f"unknown mode {self.mode!r}")
        return self.proj_drop(self.proj(out))


# ---------------------------------------------------------------------------
# Block — DSTformer-style sequential factored
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """Single DSTformer block in either 'st' or 'ts' mode (= S→T or T→S sequence).
    Per-token forward: each axis attention + FFN, pre-norm residuals.
    """

    def __init__(self, dim: int, num_heads: int = 8, ffn_dim: int = 2048,
                 qkv_bias: bool = True, attn_drop: float = 0., drop: float = 0.,
                 drop_path: float = 0., st_order: str = "st"):
        super().__init__()
        assert st_order in ("st", "ts")
        self.st_order = st_order
        self.norm1_s = RMSNorm(dim)
        self.norm1_t = RMSNorm(dim)
        self.attn_s = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                attn_drop=attn_drop, proj_drop=drop, mode="spatial")
        self.attn_t = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                attn_drop=attn_drop, proj_drop=drop, mode="temporal")
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2_s = RMSNorm(dim)
        self.norm2_t = RMSNorm(dim)
        self.ffn_s = SwiGLU(dim, ffn_dim, dropout=drop)
        self.ffn_t = SwiGLU(dim, ffn_dim, dropout=drop)

    def forward(self, x: torch.Tensor, seqlen: int = 1) -> torch.Tensor:
        if self.st_order == "st":
            x = x + self.drop_path(self.attn_s(self.norm1_s(x), seqlen))
            x = x + self.drop_path(self.ffn_s(self.norm2_s(x)))
            x = x + self.drop_path(self.attn_t(self.norm1_t(x), seqlen))
            x = x + self.drop_path(self.ffn_t(self.norm2_t(x)))
        else:  # "ts"
            x = x + self.drop_path(self.attn_t(self.norm1_t(x), seqlen))
            x = x + self.drop_path(self.ffn_t(self.norm2_t(x)))
            x = x + self.drop_path(self.attn_s(self.norm1_s(x), seqlen))
            x = x + self.drop_path(self.ffn_s(self.norm2_s(x)))
        return x


# ---------------------------------------------------------------------------
# DSTformer Body — dual-stream with per-layer learned α fusion
# ---------------------------------------------------------------------------

class DSTformerBody(nn.Module):
    """Stacks `depth` pairs of (Block_st, Block_ts) and fuses per-layer with
    learned α. This is the MotionBERT dual-stream body.

    Input/output shape: (B, F, V, dim) — F = temporal positions (after patching),
    V = spatial positions (joints).
    """

    def __init__(self, dim: int, depth: int = 8, num_heads: int = 8, ffn_dim: int = 2048,
                 qkv_bias: bool = True, attn_drop: float = 0., drop: float = 0.,
                 drop_path_rate: float = 0.1, grad_checkpoint: bool = False):
        super().__init__()
        self.depth = depth
        self.grad_checkpoint = grad_checkpoint
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks_st = nn.ModuleList([
            Block(dim, num_heads=num_heads, ffn_dim=ffn_dim, qkv_bias=qkv_bias,
                  attn_drop=attn_drop, drop=drop, drop_path=dpr[i], st_order="st")
            for i in range(depth)
        ])
        self.blocks_ts = nn.ModuleList([
            Block(dim, num_heads=num_heads, ffn_dim=ffn_dim, qkv_bias=qkv_bias,
                  attn_drop=attn_drop, drop=drop, drop_path=dpr[i], st_order="ts")
            for i in range(depth)
        ])
        # per-layer learned α (2-way softmax over concat features)
        # Init bias to 0 so both streams start equally weighted
        self.ts_attn = nn.ModuleList([nn.Linear(dim * 2, 2) for _ in range(depth)])
        for lin in self.ts_attn:
            nn.init.zeros_(lin.weight)
            nn.init.constant_(lin.bias, 0.5)
        self.norm_final = RMSNorm(dim)

    def _layer_forward(self, x, F_, blk_st, blk_ts, att):
        x_st = blk_st(x, F_)
        x_ts = blk_ts(x, F_)
        alpha = att(torch.cat([x_st, x_ts], dim=-1)).softmax(dim=-1)
        return x_st * alpha[..., 0:1] + x_ts * alpha[..., 1:2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, V, dim)
        B, F_, V_, dim = x.shape
        x = x.reshape(B * F_, V_, dim)
        if self.grad_checkpoint and self.training:
            for blk_st, blk_ts, att in zip(self.blocks_st, self.blocks_ts, self.ts_attn):
                x = torch.utils.checkpoint.checkpoint(
                    self._layer_forward, x, F_, blk_st, blk_ts, att, use_reentrant=False,
                )
        else:
            for blk_st, blk_ts, att in zip(self.blocks_st, self.blocks_ts, self.ts_attn):
                x = self._layer_forward(x, F_, blk_st, blk_ts, att)
        x = self.norm_final(x)
        x = x.reshape(B, F_, V_, dim)
        return x
