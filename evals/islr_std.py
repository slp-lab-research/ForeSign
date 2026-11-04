"""ISLR fine-tuning, standard recipe `islr_std` (MASA / Zhao lineage; docs/design/2026-10-30-phase2-c77-3d-adaptation.md,
addendum 2026-11-03).

32 frames (train: one random frame per equal segment; test: segment centres) -> EMA encoder
(full depth, all weights trainable) -> masked mean over time tokens and joint groups ->
BatchNorm1d -> Linear. SGD momentum 0.9, lr 0.01, wd 1e-4, x0.1 after epochs 20 and 40, 60 epochs,
batch 64 (drop last), plain cross-entropy, bf16. Train augmentation per clip: mirror (p 0.5),
in-plane rotation U(+-15 deg), scale U(0.8, 1.2), x/y shift U(+-0.1). Single centre view at test.
Metrics: P-I and P-C top-1/top-5.

Datasets: autsl, wlasl, asl_citizen (official split attrs), lsa64_signer (signers 1-8 / 9-10),
lsa64_rand (class-stratified random 80/20, split seed = run seed).
Budgets: 100 or 20 (random, unstratified subset of the train split; Zhao TIP Tab. VII).

  FS_CANON_ROOT=<dir> FS_STD_CKPT=<ckpt> FS_TAG=<tag> FS_STD_DS=wlasl FS_STD_BUDGET=20 \
      FS_STD_SEED=0 [FS_STD_COMPILE=1] python -m evals.islr_std
  -> results/std_<tag>_<ds>_<budget>pct_s<seed>.json
Smoke knobs: FS_STD_EPOCHS, FS_STD_MAX_STEPS, FS_STD_LIMIT (clips per split).
"""
from __future__ import annotations

import math
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from foresign import paths

N_FRAMES, BATCH, EPOCHS, LR, MOMENTUM, WD, STEPS = 32, 64, 60, 0.01, 0.9, 1e-4, (20, 40)
DATASETS = ("autsl", "wlasl", "asl_citizen", "lsa64_signer", "lsa64_rand")


def result_name(tag, ds, budget, seed):
    return f"std_{tag}_{ds}_{budget}pct_s{seed}"


def sample_indices(T: int, n: int = N_FRAMES, rng=None) -> np.ndarray:
    """n equal segments of [0, T): a uniform random frame in each (rng) or the segment centre."""
    edges = np.linspace(0, T, n + 1)
    if rng is None:
        idx = (edges[:-1] + edges[1:]) / 2
    else:
        idx = edges[:-1] + rng.random(n) * (edges[1:] - edges[:-1])
    return np.clip(np.floor(idx).astype(np.int64), 0, T - 1)


# ---- splits and labels ----------------------------------------------------------------------

def clip_label(ds, cid):
    from evals.islr_canon import clip_label as base
    if ds == "asl_citizen":
        name = cid.split("__")[-1]
        return name.split("-", 1)[1].lower() if "-" in name else None
    return base("lsa64" if ds.startswith("lsa64") else ds, cid)


def collect(ds, root, seed=0, cache_dir=None):
    """-> (train [(file_idx, clip_id, label)], test [...], classes, files)."""
    from foresign.canon_data import load_or_build_canon_index
    from foresign.canon_splits import canon_split
    base = "lsa64" if ds.startswith("lsa64") else ds
    idx = load_or_build_canon_index(root, [base], min_frames=1, cache_dir=cache_dir)
    rows = sorted(idx["clips"][base], key=lambda r: r[1])
    tr, te = [], []
    if ds == "lsa64_rand":
        by = defaultdict(list)
        for fi, cid, _T, _sp in rows:
            by[clip_label(ds, cid)].append((fi, cid))
        rng = np.random.default_rng(seed)
        for lab in sorted(by):
            items = by[lab]
            perm = rng.permutation(len(items))
            k = int(round(0.8 * len(items)))
            tr += [(*items[i], lab) for i in sorted(perm[:k])]
            te += [(*items[i], lab) for i in sorted(perm[k:])]
    else:
        for fi, cid, _T, sp in rows:
            if ds == "asl_citizen":
                s = {"train": "train", "test": "test"}.get(sp, "other")
            else:
                s = canon_split(base, cid, sp)
            lab = clip_label(ds, cid)
            if lab is None or s not in ("train", "test"):
                continue
            (tr if s == "train" else te).append((fi, cid, lab))
    classes = sorted({r[2] for r in tr})
    cmap = {c: i for i, c in enumerate(classes)}
    train = [(fi, cid, cmap[l]) for fi, cid, l in tr]
    test = [(fi, cid, cmap[l]) for fi, cid, l in te if l in cmap]
    return train, test, classes, idx["files"][base]


def label_subset(items, budget, seed):
    """budget 100 -> all; else a random (unstratified) budget% of the train clips."""
    if budget == 100:
        return list(items)
    rng = np.random.default_rng(seed)
    k = int(round(budget / 100 * len(items)))
    return [items[i] for i in sorted(rng.choice(len(items), size=k, replace=False))]


# ---- data ---------------------------------------------------------------------------------

def load_clips(root, files, items):
    from evals.finetune_islr_canon import load_clips as lc
    return lc(root, files, items)


def make_batch(clips, idx, in_dim, rng=None, augment=False, pad_to=None, n=N_FRAMES):
    from foresign.augment import islr_augment
    from foresign.inputs import pack_c77
    B = max(len(idx), pad_to or 0)
    out = np.zeros((B, n, 77 * (in_dim + 1)), np.float32)
    for b, i in enumerate(idx):
        x, m = clips[i]
        sel = sample_indices(x.shape[0], n, rng)
        xs, ms = x[sel], m[sel]
        if augment:
            xs, ms = islr_augment(xs, ms, rng)
        out[b] = pack_c77(xs, ms, in_dim)
    nv = torch.full((B,), n, dtype=torch.long)
    return torch.from_numpy(out), nv


# ---- model --------------------------------------------------------------------------------

class StdModel(nn.Module):
    def __init__(self, ckpt, n_cls):
        super().__init__()
        from foresign.build import build_model
        from foresign.inputs import C77, layout_for
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        if layout_for(cfg).fmt != C77:
            raise ValueError("islr_std needs a c77 checkpoint")
        m = build_model(cfg)
        m.load_state_dict(ck["model"], strict=False)
        self.tok, self.body = m.target_tok, m.target_ctx
        for p in self.parameters():
            p.requires_grad_(True)
        # 32 frames = 16 tokens: activations fit without checkpointing (exact maths either way)
        self.body.grad_checkpoint = os.environ.get("FS_STD_GRAD_CKPT", "0") == "1"
        self.in_dim = int(cfg.get("in_dim", 3))
        self.d_model = cfg["d_model"]
        self.step = int(ck.get("step", 0))
        self.bn = nn.BatchNorm1d(self.d_model)
        self.fc = nn.Linear(self.d_model, n_cls)
        self.compiled = False

    def forward(self, x, nv):
        z = self.body(self.tok(x)).mean(2)                     # (B, TP, d)
        tp = z.size(1)
        t_patch = x.size(1) // tp
        vtp = torch.div(nv + t_patch - 1, t_patch, rounding_mode="floor").clamp(1, tp)
        m = (torch.arange(tp, device=z.device)[None] < vtp[:, None]).to(z.dtype)[..., None]
        pooled = (z * m).sum(1) / m.sum(1)
        return self.fc(self.bn(pooled.float()))


@torch.no_grad()
def evaluate(model, clips, y, n_cls, device, bs=128):
    model.eval()
    pad = bs if model.compiled else None
    top1 = np.zeros(len(clips), bool)
    top5 = np.zeros(len(clips), bool)
    for i in range(0, len(clips), bs):
        idx = list(range(i, min(i + bs, len(clips))))
        x, nv = make_batch(clips, idx, model.in_dim, pad_to=pad)
        with torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
            logit = model(x.to(device), nv.to(device)).float()[:len(idx)]
        t = torch.from_numpy(y[idx]).to(device)
        top = logit.topk(min(5, n_cls), 1).indices
        top1[idx] = (top[:, 0] == t).cpu().numpy()
        top5[idx] = (top == t[:, None]).any(1).cpu().numpy()
    per_cls1, per_cls5 = [], []
    for c in np.unique(y):
        sel = y == c
        per_cls1.append(top1[sel].mean()); per_cls5.append(top5[sel].mean())
    return {"pi_top1": float(top1.mean()), "pi_top5": float(top5.mean()),
            "pc_top1": float(np.mean(per_cls1)), "pc_top5": float(np.mean(per_cls5))}


def run(ckpt, tag, ds, budget, seed, device="cuda", epochs=None, max_steps=0, limit=0,
        root=None, cache_dir=None):
    from foresign.results import write_result
    if ds not in DATASETS:
        raise ValueError(f"unknown dataset {ds}")
    name = result_name(tag, ds, budget, seed)
    res_dir = os.environ.get("FS_RESULTS_DIR", paths.RESULTS)
    state_p = os.path.join(res_dir, "std_state", name + ".pt")
    root = root or paths.canon_root()
    tr_items, te_items, classes, files = collect(ds, root, seed=seed, cache_dir=cache_dir)
    sub = label_subset(tr_items, budget, seed)
    if limit:
        sub, te_items = sub[:limit], te_items[:limit]
    t0 = time.time()
    tr_clips, ytr = load_clips(root, files, sub)
    te_clips, yte = load_clips(root, files, te_items)
    n_cls = len(classes)
    print(f"[std] {name}: train {len(sub)}/{len(tr_items)} test {len(te_items)} cls {n_cls} "
          f"(load {time.time() - t0:.0f}s)", flush=True)

    torch.manual_seed(seed)
    model = StdModel(ckpt, n_cls).to(device)
    from foresign.build import compile_flag
    model.compiled = compile_flag({}, "FS_STD_COMPILE")
    if model.compiled:
        model.compile(mode="default", dynamic=False)
        print("[std] torch.compile on", flush=True)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD)
    n_ep = epochs or EPOCHS
    spe = len(sub) // BATCH if len(sub) >= BATCH else 1        # drop last (keep >= 1 step)
    start_ep = 0
    if os.path.exists(state_p):
        st = torch.load(state_p, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        start_ep = st["epoch"]
        print(f"[std] resumed {name} at epoch {start_ep}", flush=True)
    dtype = device.split(":")[0]
    n_done, t_train, t_warm, stopped, last_loss = 0, time.time(), None, False, float("nan")
    for ep in range(start_ep, n_ep):
        lr = LR * (0.1 ** sum(ep >= s for s in STEPS))
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        rng = np.random.default_rng([seed, ep])
        perm = rng.permutation(len(sub))
        tot, cnt = 0.0, 0
        for b in range(spe):
            idx = perm[b * BATCH:(b + 1) * BATCH].tolist()
            x, nv = make_batch(tr_clips, idx, model.in_dim, rng, augment=True,
                               pad_to=BATCH if model.compiled else None)
            with torch.autocast(dtype, dtype=torch.bfloat16):
                logit = model(x.to(device), nv.to(device))[:len(idx)]
            loss = F.cross_entropy(logit.float(), torch.from_numpy(ytr[idx]).to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            n_done += 1
            tot += loss.item() * len(idx); cnt += len(idx)
            if n_done == 10:
                t_warm = time.time()
            if max_steps and n_done >= max_steps:
                stopped = True
                break
        last_loss = tot / max(1, cnt)
        print(f"[std] {name} ep{ep + 1}/{n_ep} lr {lr:.0e} loss {last_loss:.4f} "
              f"{n_done / max(1e-6, time.time() - t_train):.2f} it/s", flush=True)
        if stopped:
            break
        os.makedirs(os.path.dirname(state_p), exist_ok=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1},
                   state_p + ".tmp")
        os.replace(state_p + ".tmp", state_p)
    t_end = time.time()
    met = evaluate(model, te_clips, yte, n_cls, device)
    print(f"[std] {name} FINAL P-I {met['pi_top1']:.4f}/{met['pi_top5']:.4f} "
          f"P-C {met['pc_top1']:.4f}/{met['pc_top5']:.4f}", flush=True)
    from foresign.build import speed_flags
    out = {"task": "islr_std", "recipe": "islr_std", "ckpt": ckpt, "ckpt_step": model.step,
           "tag": tag, "dataset": ds, "budget_pct": budget, "seed": seed,
           "n_train": len(sub), "n_train_full": len(tr_items), "n_test": len(te_items),
           "n_cls": n_cls, "epochs": n_ep, "max_steps": max_steps, "limit": limit,
           "complete": not stopped and not limit, **met, "epoch_loss_last": last_loss,
           "compile": model.compiled, "fast_ops": all(speed_flags({"input_format": "c77"}).values()),
           "train_it_per_s_after10": ((n_done - 10) / max(1e-6, t_end - t_warm)
                                      if t_warm and n_done > 10 else None),
           "hparams": {"frames": N_FRAMES, "batch": BATCH, "lr": LR, "momentum": MOMENTUM,
                       "wd": WD, "steps": list(STEPS)}}
    write_result(name, out)
    if os.path.exists(state_p):
        os.remove(state_p)
    return out


def main():
    run(os.environ["FS_STD_CKPT"], os.environ["FS_TAG"], os.environ["FS_STD_DS"],
        int(os.environ.get("FS_STD_BUDGET", "100")), int(os.environ.get("FS_STD_SEED", "0")),
        device=os.environ.get("FS_DEVICE", "cuda"),
        epochs=int(os.environ["FS_STD_EPOCHS"]) if os.environ.get("FS_STD_EPOCHS") else None,
        max_steps=int(os.environ.get("FS_STD_MAX_STEPS", "0")),
        limit=int(os.environ.get("FS_STD_LIMIT", "0")))


if __name__ == "__main__":
    main()
