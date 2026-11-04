"""Speed knobs with the same maths (kernel rounding only):
- `fast_attn`: temporal attention on the efficient SDPA kernel (below).
- `fast_norm`: RMSNorm via the fused F.rms_norm (one kernel instead of ~6; |diff| <= 1 bf16 ulp).

Temporal attention:
The vendored v4_blocks.Attention (mode "temporal") folds q/k/v to 3-D (B*H*N, F, d) before
F.scaled_dot_product_attention. 3-D inputs are not eligible for the flash / memory-efficient
kernels, so PyTorch runs the math fallback (fp32 upcast, separate matmul + softmax): ~20% of the
encoder forward in our profile. Here the same attention (per batch, head and joint, over
frames) is computed with 4-D inputs (B, H*N, F, d), which the efficient kernels accept. The result
equals the original up to kernel rounding (bf16). The vendored file is not modified:
`enable(model)` swaps the class of every temporal Attention module.
"""
from __future__ import annotations

import torch.nn.functional as F
from foresign._vendor.resona.models.v4_blocks import Attention, RMSNorm


class FastTemporalAttention(Attention):
    def forward(self, x, seqlen: int = 1):
        if self.mode != "temporal":
            return super().forward(x, seqlen)
        B_, N, C = x.shape                              # (B*F, N, C)
        H = self.num_heads
        d = C // H
        B = B_ // seqlen
        qkv = self.qkv(x).reshape(B, seqlen, N, 3, H, d)
        q, k, v = qkv.permute(3, 0, 4, 2, 1, 5).unbind(0)   # each (B, H, N, F, d)
        q, k, v = (t.reshape(B, H * N, seqlen, d) for t in (q, k, v))
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop_p if self.training else 0.0, is_causal=False)
        out = (out.reshape(B, H, N, seqlen, d).permute(0, 3, 2, 1, 4)   # (B, F, N, H, d)
               .reshape(B_, N, C))
        return self.proj_drop(self.proj(out))


class FastRMSNorm(RMSNorm):
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight.to(x.dtype), self.eps)


def enable(module) -> int:
    """Swap every temporal Attention in `module` to FastTemporalAttention; returns the count."""
    n = 0
    for m in module.modules():
        if type(m) is Attention and m.mode == "temporal":
            m.__class__ = FastTemporalAttention
            n += 1
    return n


def enable_norm(module) -> int:
    """Swap every RMSNorm in `module` to FastRMSNorm; returns the count (0 and a warning when
    this torch has no F.rms_norm, i.e. < 2.4: the original RMSNorm is kept)."""
    if not hasattr(F, "rms_norm"):
        import warnings
        warnings.warn("torch has no F.rms_norm; fast_norm ignored (original RMSNorm kept)")
        return 0
    n = 0
    for m in module.modules():
        if type(m) is RMSNorm:
            m.__class__ = FastRMSNorm
            n += 1
    return n
