"""Supervised-contrastive loss used by the optional SSL-time auxiliary terms of
foresign.jepa (language / content SupCon; off in all Phase-2 configs)."""
from __future__ import annotations

import torch


def supcon(z, labels, dom, temp, cross_only=False):
    sim = (z @ z.t()) / temp
    n = z.size(0)
    diag = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim - sim.max(1, keepdim=True).values.detach()
    exp = torch.exp(sim).masked_fill(diag, 0.0)
    logp = sim - torch.log(exp.sum(1, keepdim=True) + 1e-12)
    pos = (labels[:, None] == labels[None, :]) & ~diag
    if cross_only:
        pos = pos & (dom[:, None] != dom[None, :])
    cnt = pos.sum(1); valid = cnt > 0
    if not valid.any():
        return z.new_tensor(0.0)
    return -((pos * logp).sum(1)[valid] / cnt[valid].clamp(min=1)).mean()
