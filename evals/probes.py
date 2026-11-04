"""Frozen-feature probes shared by the canon ISLR evaluation (evals.islr_canon): full-batch
linear probe (top-1, top-5, per-class recall), similarity-weighted kNN, and a V-JEPA-style
attentive probe over joint-pooled time tokens."""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F


def make_adapter(device="cuda"):
    """FS_ADAPTER=foresign:<ckpt.pt> -> (ForeSignAdapter, tag). FS_WHICH=ema|online selects the
    EMA target (default) or the online encoder; FS_TAG overrides the tag."""
    from foresign.adapters import ForeSignAdapter
    spec = os.environ.get("FS_ADAPTER", "")
    if not spec.startswith("foresign:"):
        raise ValueError("set FS_ADAPTER=foresign:<checkpoint.pt>")
    ckpt = spec.split(":", 1)[1]
    which = os.environ.get("FS_WHICH", "ema")
    tag = os.environ.get("FS_TAG") or (
        os.path.basename(os.path.dirname(ckpt)) + "_" +
        os.path.basename(ckpt).replace(".pt", ""))
    return ForeSignAdapter(ckpt, device=device, which=which), tag


def linear_probe_full(ftr, ytr, fte, yte, n_cls, device, seed, epochs=120):
    """Linear probe (AdamW, cosine schedule, standardised features) that also returns per-class
    recall. Returns (top1, top5, per_class) with per_class (n_cls,) float tensor,
    NaN for classes absent from the test set (use nanmean for P-C)."""
    torch.manual_seed(seed)
    mu, sd = ftr.mean(0, keepdim=True), ftr.std(0, keepdim=True).clamp(min=1e-6)
    Xtr = ((ftr - mu) / sd).to(device); Xte = ((fte - mu) / sd).to(device)
    ytr = ytr.to(device); yte = yte.to(device)
    lin = torch.nn.Linear(Xtr.size(1), n_cls).to(device)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-2, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = Xtr.size(0)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 4096):
            idx = perm[i:i + 4096]
            opt.zero_grad()
            F.cross_entropy(lin(Xtr[idx]), ytr[idx]).backward(); opt.step()
        sched.step()
    with torch.no_grad():
        logits = lin(Xte)
        hit = (logits.argmax(1) == yte).float()
        top1 = hit.mean().item()
        top5 = (logits.topk(min(5, n_cls), 1).indices == yte[:, None]).any(1).float().mean().item()
        cnt = torch.zeros(n_cls, device=device).index_add_(0, yte, torch.ones_like(hit))
        cor = torch.zeros(n_cls, device=device).index_add_(0, yte, hit)
        pc = torch.where(cnt > 0, cor / cnt.clamp(min=1), torch.full_like(cnt, float("nan")))
    return top1, top5, pc.cpu()


@torch.no_grad()
def knn_probe(ftr, ytr, fte, yte, n_cls, device, k=5, chunk=2048):
    a = F.normalize(ftr, dim=-1).to(device)
    b = F.normalize(fte, dim=-1).to(device)
    ytr_d = ytr.to(device)
    top1 = []
    for i in range(0, b.size(0), chunk):
        sim = b[i:i + chunk] @ a.T                              # (c, Ntr)
        v, idx = sim.topk(k, dim=1)
        votes = torch.zeros(idx.size(0), n_cls, device=device)
        votes.scatter_add_(1, ytr_d[idx], v.clamp(min=0))       # sim-weighted
        top1.append(votes.argmax(1))
    pred = torch.cat(top1)
    return (pred == yte.to(device)).float().mean().item()


class AttentiveHead(torch.nn.Module):
    """V-JEPA-style attentive probe over (cached, joint-pooled) time tokens."""

    def __init__(self, d, n_cls, n_heads=8):
        super().__init__()
        self.q = torch.nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = torch.nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm = torch.nn.LayerNorm(d)
        self.fc = torch.nn.Linear(d, n_cls)

    def forward(self, t, m):                  # t (B,TP,d), m (B,TP) True=valid
        q = self.q.expand(t.size(0), 1, t.size(2))
        o, _ = self.attn(q, t, t, key_padding_mask=~m)
        return self.fc(self.norm(o.squeeze(1)))


def attentive_probe(ttr, mtr, ytr, tte, mte, yte, n_cls, device, seed,
                    epochs=40, bs=256, lr=1e-3):
    torch.manual_seed(seed)
    head = AttentiveHead(ttr.size(2), n_cls).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = ttr.size(0)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = F.cross_entropy(
                head(ttr[idx].to(device), mtr[idx].to(device)), ytr[idx].to(device))
            loss.backward(); opt.step()
        sched.step()
    head.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, tte.size(0), bs):
            logit = head(tte[i:i + bs].to(device), mte[i:i + bs].to(device))
            correct += (logit.argmax(1) == yte[i:i + bs].to(device)).sum().item()
            total += logit.size(0)
    return correct / total
