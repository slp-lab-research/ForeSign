"""Low-label fine-tuning ISLR on C77-3D canon data (Phase 2; protocol fixed in the plan addendum
2026-10-31 of docs/design/2026-10-30-phase2-c77-3d-adaptation.md).

EMA encoder of the checkpoint (tap FS_TAP_DEPTH, 0 = full depth = L0) -> joint-mean time tokens ->
attentive-pooling head; every encoder weight trainable. AdamW (wd 0.05, none on 1-D params), head lr
1e-3, encoder 5e-4 with layer-wise decay 0.75, warmup 2 epochs + cosine, batch 64, label smoothing
0.1, bf16 autocast. Budget 100% (30 epochs) or class-stratified 10% (60 epochs). Train: random crop;
test: centre crop. Final-epoch test top-1/top-5. Identical for every checkpoint.

  FS_CANON_ROOT=<dir> FS_FT_CKPT=<ckpt> FS_TAG=<tag> FS_FT_DS=autsl FS_FT_BUDGET=10 FS_FT_SEED=0 \
      python -m evals.finetune_islr_canon      -> results/ft_<tag>_<ds>_<budget>pct_s<seed>.json
Smoke-only knobs: FS_FT_MAX_STEPS (stop after n optimiser steps), FS_FT_EPOCHS (override).
Per-epoch state in $FS_RESULTS_DIR/ft_state/ (resume after a crash; removed when done).
"""
from __future__ import annotations

import math
import os
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from foresign import paths

HEAD_LR, BASE_LR, LLRD, WD = 1e-3, 5e-4, 0.75, 0.05
BATCH, WARMUP_EP, SMOOTH = 64, 2, 0.1
EPOCHS = {100: 30, 10: 60}
SEEDS = {100: (0,), 10: (0, 1, 2)}


def result_name(tag, ds, budget, seed):
    return f"ft_{tag}_{ds}_{budget}pct_s{seed}"


def stratified_subset(items, budget, seed):
    """items [(file_idx, clip_id, class)] -> class-stratified subset: per class
    max(1, round(budget/100 * n_c)) clips, drawn by default_rng(seed) from the clip-id sorted list."""
    if budget == 100:
        return list(items)
    by = {}
    for it in sorted(items, key=lambda r: r[1]):
        by.setdefault(it[2], []).append(it)
    rng = np.random.default_rng(seed)
    out = []
    for c in sorted(by):
        rows = by[c]
        k = max(1, int(round(budget / 100 * len(rows))))
        out.extend(rows[i] for i in sorted(rng.choice(len(rows), size=k, replace=False)))
    return out


def load_clips(root, files, items):
    """-> list of (x (T,77,3) f16, mask (T,77) u8) in RAM, labels (N,)."""
    handles, clips = {}, []
    for fi, cid, _ in items:
        if fi not in handles:
            handles[fi] = h5py.File(os.path.join(root, files[fi]), "r")
        g = handles[fi][cid]
        clips.append((g["x"][:], g["mask"][:]))
    for h in handles.values():
        h.close()
    return clips, np.array([it[2] for it in items], np.int64)


def pack_batch(clips, idx, L, in_dim, rng=None, pad_to=None):
    """Random crop (rng given, train) or centre crop (test); zero/mask-0 padding.
    pad_to: append empty rows (all zero, n_valid 1) up to a fixed batch size, for a compiled
    model with static shapes; callers drop those rows from the loss / predictions.
    -> pose (B, L, 77*(in_dim+1)) float32, n_valid (B,)."""
    from foresign.inputs import pack_c77
    n = max(len(idx), pad_to or 0)
    out = np.zeros((n, L, 77 * (in_dim + 1)), np.float32)
    nv = np.ones(n, np.int64)
    for b, i in enumerate(idx):
        x, m = clips[i]
        T = x.shape[0]
        s = (rng.integers(0, T - L + 1) if rng is not None else (T - L) // 2) if T > L else 0
        p = pack_c77(x[s:s + L], m[s:s + L], in_dim)
        out[b, :p.shape[0]] = p
        nv[b] = p.shape[0]
    return torch.from_numpy(out), torch.from_numpy(nv)


class AttnPoolHead(nn.Module):
    def __init__(self, d, n_cls, n_heads=8):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.fc = nn.Linear(d, n_cls)

    def forward(self, t, m):                       # t (B,TP,d), m (B,TP) True = valid
        o, _ = self.attn(self.q.expand(t.size(0), 1, t.size(2)), t, t, key_padding_mask=~m)
        return self.fc(self.norm(o.squeeze(1)))


class FTModel(nn.Module):
    """EMA tokenizer + body (all trainable) -> time tokens -> attentive head."""

    def __init__(self, ckpt, n_cls, tap_depth=0):
        super().__init__()
        from foresign.build import build_model
        from foresign.inputs import C77, layout_for
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        if layout_for(cfg).fmt != C77:
            raise ValueError("finetune_islr_canon needs a c77 checkpoint")
        m = build_model(cfg)
        m.load_state_dict(ck["model"], strict=False)
        # fast kernels follow build_model (default on for c77; FS_FAST_OPS=0 disables)
        from foresign.build import speed_flags
        self.fast = speed_flags(cfg)["fast_attn"] and speed_flags(cfg)["fast_norm"]
        self.tok, self.body = m.target_tok, m.target_ctx
        for p in self.parameters():
            p.requires_grad_(True)
        # activation checkpointing as in pretraining (same maths, identical results in the
        # smoke). Measured at batch 64 x clip 128: 4.5 GiB peak with it, 30.7 GiB without
        # (does not fit a 24 GB GPU). Keep on; FS_FT_GRAD_CKPT=0 disables.
        self.body.grad_checkpoint = os.environ.get("FS_FT_GRAD_CKPT", "1") == "1"
        # FS_FT_CKPT_LAYERS=n: checkpoint only the first n layers (more memory, less recompute)
        cl = os.environ.get("FS_FT_CKPT_LAYERS")
        self.body.ckpt_layers = int(cl) if cl not in (None, "") else None
        self.depth = self.body.depth
        self.tap = tap_depth if 0 < tap_depth < self.depth else None
        self.t_patch, self.clip_len = cfg["t_patch"], cfg["clip_len"]
        self.in_dim = int(cfg.get("in_dim", 3))
        self.d_model = cfg["d_model"]
        self.step = int(ck.get("step", 0))
        self.head = AttnPoolHead(self.d_model, n_cls)
        self.compiled = False

    def encode(self, pose):
        h = self.tok(pose)                                    # (B,TP,V,d)
        if self.tap is None:
            return self.body(h)
        N, TP, V, d = h.shape
        x = h.reshape(N * TP, V, d)
        b = self.body
        for li, (st, ts, att) in enumerate(zip(b.blocks_st, b.blocks_ts, b.ts_attn), 1):
            x = b._layer_forward(x, TP, st, ts, att)
            if li == self.tap:
                break
        return x.reshape(N, TP, V, d)

    def forward(self, pose, nv):
        z = self.encode(pose).mean(2)                         # (B,TP,d)
        tp = z.size(1)
        vtp = torch.div(nv + self.t_patch - 1, self.t_patch, rounding_mode="floor").clamp(1, tp)
        m = torch.arange(tp, device=z.device)[None] < vtp[:, None]
        return self.head(z, m)


def _layer_id(name, depth):
    if name.startswith("tok."):
        return 0
    if name.startswith("body."):
        parts = name.split(".")
        if parts[1] in ("blocks_st", "blocks_ts", "ts_attn"):
            return int(parts[2]) + 1
        return depth + 1                                      # norm_final
    raise KeyError(name)


def param_groups(model, base_lr=BASE_LR, head_lr=HEAD_LR, decay=LLRD, wd=WD):
    """Layer-wise lr decay groups; no weight decay on 1-D params (biases, norms, embeddings)."""
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("head."):
            key, lr = ("head", -1), head_lr
        else:
            lid = _layer_id(n, model.depth)
            key, lr = ("enc", lid), base_lr * decay ** (model.depth + 1 - lid)
        nd = p.ndim <= 1 or n.endswith("_embed") or n.endswith(".absent")
        g = groups.setdefault(key + (nd,), {"params": [], "lr": lr, "base_lr": lr,
                                            "weight_decay": 0.0 if nd else wd,
                                            "name": f"{key[0]}{key[1]}{'_nd' if nd else ''}"})
        g["params"].append(p)
    return list(groups.values())


def lr_factor(step, warm, total):
    if step < warm:
        return (step + 1) / warm
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))


@torch.no_grad()
def evaluate(model, clips, y, device, bs=128):
    model.eval()
    hit1 = hit5 = 0
    pad = bs if getattr(model, "compiled", False) else None
    for i in range(0, len(clips), bs):
        idx = list(range(i, min(i + bs, len(clips))))
        x, nv = pack_batch(clips, idx, model.clip_len, model.in_dim, pad_to=pad)
        with torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
            logit = model(x.to(device), nv.to(device)).float()[:len(idx)]
        t = torch.from_numpy(y[idx]).to(device)
        top = logit.topk(min(5, logit.size(1)), 1).indices
        hit1 += (top[:, 0] == t).sum().item()
        hit5 += (top == t[:, None]).any(1).sum().item()
    return hit1 / len(clips), hit5 / len(clips)


def run(ckpt, tag, ds, budget, seed, device="cuda", tap_depth=0, epochs=None, max_steps=0,
        root=None, cache_dir=None):
    from evals.islr_canon import collect_canon
    from foresign.results import write_result
    if budget not in EPOCHS:
        raise ValueError(f"budget must be 10 or 100, got {budget}")
    name = result_name(tag, ds, budget, seed)
    res_dir = os.environ.get("FS_RESULTS_DIR", paths.RESULTS)
    state_p = os.path.join(res_dir, "ft_state", name + ".pt")
    root = root or paths.canon_root()
    tr_items, te_items, n_cls, files = collect_canon(ds, root, cache_dir=cache_dir)
    sub = stratified_subset(tr_items, budget, seed)
    t0 = time.time()
    tr_clips, ytr = load_clips(root, files, sub)
    te_clips, yte = load_clips(root, files, te_items)
    print(f"[ft] {name}: train {len(sub)}/{len(tr_items)} test {len(te_items)} cls {n_cls} "
          f"(load {time.time() - t0:.0f}s)", flush=True)

    torch.manual_seed(seed)
    model = FTModel(ckpt, n_cls, tap_depth).to(device)
    from foresign.build import compile_flag
    model.compiled = compile_flag({}, "FS_FT_COMPILE")
    if model.compiled:          # in place: state_dict keys unchanged; static shapes (padding)
        model.compile(mode="default", dynamic=False)
        print("[ft] torch.compile on (mode default, dynamic=False)", flush=True)
    groups = param_groups(model)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    n_ep = epochs or EPOCHS[budget]
    spe = math.ceil(len(sub) / BATCH)
    total, warm = n_ep * spe, WARMUP_EP * spe
    start_ep = 0
    if os.path.exists(state_p):
        st = torch.load(state_p, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        start_ep = st["epoch"]
        print(f"[ft] resumed {name} at epoch {start_ep}", flush=True)
    step, n_done, t_train, dtype = start_ep * spe, 0, time.time(), device.split(":")[0]
    t_warm = None                                  # time after 10 steps (compile warmup)
    stopped = False
    for ep in range(start_ep, n_ep):
        model.train()
        rng = np.random.default_rng([seed, ep])   # per-epoch stream: resume == uninterrupted
        perm = rng.permutation(len(sub))
        tot = 0.0
        for i in range(0, len(sub), BATCH):
            idx = perm[i:i + BATCH].tolist()
            x, nv = pack_batch(tr_clips, idx, model.clip_len, model.in_dim, rng,
                               pad_to=BATCH if model.compiled else None)
            f = lr_factor(step, warm, total)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * f
            with torch.autocast(dtype, dtype=torch.bfloat16):
                logit = model(x.to(device), nv.to(device))[:len(idx)]
            loss = F.cross_entropy(logit.float(), torch.from_numpy(ytr[idx]).to(device),
                                   label_smoothing=SMOOTH)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1; n_done += 1; tot += loss.item() * len(idx)     # .item() syncs
            if n_done == 10:
                t_warm = time.time()
            if max_steps and n_done >= max_steps:
                stopped = True
                break
        rate = n_done / max(1e-6, time.time() - t_train)
        print(f"[ft] {name} ep{ep + 1}/{n_ep} loss {tot / len(sub):.4f} {rate:.2f} it/s", flush=True)
        if stopped:
            break
        os.makedirs(os.path.dirname(state_p), exist_ok=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1},
                   state_p + ".tmp")
        os.replace(state_p + ".tmp", state_p)
    t_end = time.time()
    top1, top5 = evaluate(model, te_clips, yte, device)
    print(f"[ft] {name} FINAL test top1 {top1:.4f} top5 {top5:.4f}", flush=True)
    out = {"task": "islr_finetune_canon", "ckpt": ckpt, "ckpt_step": model.step, "tag": tag,
           "dataset": ds, "budget_pct": budget, "seed": seed, "n_train": len(sub),
           "n_train_full": len(tr_items), "n_test": len(te_items), "n_cls": n_cls,
           "epochs": n_ep, "steps": step, "max_steps": max_steps, "complete": not stopped,
           "tap_depth": tap_depth, "top1": top1, "top5": top5, "fast_ops": model.fast,
           "ckpt_layers": model.body.ckpt_layers,
           "train_it_per_s": n_done / max(1e-6, time.time() - t_train),
           "train_it_per_s_after10": ((n_done - 10) / max(1e-6, t_end - t_warm)
                                      if t_warm and n_done > 10 else None),
           "epoch_loss_last": tot / len(sub), "compile": model.compiled,
           "recipe": {"head_lr": HEAD_LR, "base_lr": BASE_LR, "llrd": LLRD, "wd": WD,
                      "batch": BATCH, "warmup_epochs": WARMUP_EP, "label_smoothing": SMOOTH}}
    write_result(name, out)
    if os.path.exists(state_p):
        os.remove(state_p)
    return out


def main():
    run(os.environ["FS_FT_CKPT"], os.environ["FS_TAG"], os.environ["FS_FT_DS"],
        int(os.environ.get("FS_FT_BUDGET", "100")), int(os.environ.get("FS_FT_SEED", "0")),
        device=os.environ.get("FS_DEVICE", "cuda"),
        tap_depth=int(os.environ.get("FS_TAP_DEPTH", "0")),
        epochs=int(os.environ["FS_FT_EPOCHS"]) if os.environ.get("FS_FT_EPOCHS") else None,
        max_steps=int(os.environ.get("FS_FT_MAX_STEPS", "0")))


if __name__ == "__main__":
    main()
