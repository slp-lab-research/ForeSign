"""CSLR fine-tuning, CoSign-style recipe `cslr_cosign` (docs/design/2026-10-30-phase2-c77-3d-adaptation.md, addendum 2026-11-03).

EMA encoder (all weights trainable, full depth) on non-overlapping 128-frame windows (tokens at
12.5 Hz, concatenated) -> conv1d(k5, 1024)+BN+ReLU -> maxpool(2) -> conv1d(k5, 1024)+BN+ReLU
[overall 4x from 25 fps, as CoSign] -> auxiliary linear classifier; -> 2-layer BiLSTM (2x512) ->
primary linear classifier. Loss = CTC(primary) + CTC(auxiliary) (VAC/CoSign reduction: per-sample
loss, batch mean). AdamW lr 4e-4 (encoder: layer-wise decay 0.75), wd 1e-4, /10 after epochs 20
and 35, 40 epochs, batch 8, bf16, grad clip 1.0. Augmentation: random temporal scaling +-20%.
Decoding: CTC prefix beam search (width 10, top-40 token cutoff, no LM) and greedy. Dev WER every
epoch (official protocol, evals/phoenix_wer.py); test at the best-dev epoch.

  FS_CANON_ROOT=<dir> FS_CC_CKPT=<ckpt> FS_TAG=<tag> FS_CC_DS=ph14t FS_CC_SEED=0 \
      python -m evals.cslr_cosign            -> results/cosign_<tag>_<ds>_s<seed>.json
Smoke knobs: FS_CC_EPOCHS, FS_CC_LIMIT (clips per split), FS_CC_MAX_STEPS (per epoch).
FS_CC_GRAD_CKPT=0 disables activation checkpointing in the encoder (exact maths, more memory).
"""
from __future__ import annotations

import json
import math
import os
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from foresign import paths

LR, WD, LLRD, BATCH, EPOCHS, MILESTONES = 4e-4, 1e-4, 0.75, 8, 40, (20, 35)
HIDDEN, BEAM, TOPN, TSCALE = 1024, 10, 40, 0.2
CANON = {"ph14t": "canon_ph14t", "ph14": "canon_ph14"}


def result_name(tag, ds, seed):
    return f"cosign_{tag}_{ds}_s{seed}"


# ---- data ---------------------------------------------------------------------------------

def load_split(ds, split, in_dim, root=None):
    """-> [(utt_id, packed pose (T, F) or None, gloss list)] for the canon PHOENIX release."""
    from evals.cslr_phoenix import ann_path, canon_keys
    from foresign.inputs import pack_c77
    cds = CANON[ds]
    ann = json.load(open(ann_path(split, cds), encoding="utf-8"))
    keys = canon_keys(cds)
    h5 = os.path.join(root or paths.canon_root(), {"ph14t": "phoenix14t", "ph14": "phoenix14"}[ds]
                      + ".h5")
    out = []
    with h5py.File(h5, "r") as f:
        for it in ann:
            k = keys.get(it["id"])
            x = pack_c77(f[k]["x"][:], f[k]["mask"][:], in_dim) if k else None
            out.append((it["id"], x, it["gloss"].split()))
    return out


def temporal_scale(x, rng, r=TSCALE):
    """Resample to round(T * U(1-r, 1+r)) frames (nearest frame)."""
    T = x.shape[0]
    n = max(1, int(round(T * rng.uniform(1 - r, 1 + r))))
    return x[np.round(np.linspace(0, T - 1, n)).astype(np.int64)]


def make_batch(items, win, vocab=None, rng=None):
    xs = [temporal_scale(x, rng) if rng is not None else x for _, x, _ in items]
    lens = [x.shape[0] for x in xs]
    tpad = max(int(math.ceil(l / win)) * win for l in lens)
    P = np.zeros((len(xs), tpad, xs[0].shape[1]), np.float32)
    for b, x in enumerate(xs):
        P[b, :x.shape[0]] = x
    out = [torch.from_numpy(P), torch.tensor(lens)]
    if vocab is not None:
        tg = [[vocab[g] for g in gl if g in vocab] for _, _, gl in items]
        out += [torch.tensor([t for s in tg for t in s], dtype=torch.long),
                torch.tensor([len(s) for s in tg], dtype=torch.long)]
    return out


# ---- model --------------------------------------------------------------------------------

class CoSignCSLR(nn.Module):
    def __init__(self, ckpt, n_cls):
        super().__init__()
        from evals.cslr_phoenix import WindowEncoder
        os.environ.setdefault("FS_TAP_DEPTH", "0")
        self.enc = WindowEncoder(ckpt, which="ema", freeze=False)
        self.enc.body.grad_checkpoint = os.environ.get("FS_CC_GRAD_CKPT", "1") == "1"
        d = self.enc.d_model
        self.conv = nn.Sequential(
            nn.Conv1d(d, HIDDEN, 5, padding=2), nn.BatchNorm1d(HIDDEN), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(HIDDEN, HIDDEN, 5, padding=2), nn.BatchNorm1d(HIDDEN), nn.ReLU(inplace=True))
        self.aux_fc = nn.Linear(HIDDEN, n_cls + 1)
        self.lstm = nn.LSTM(HIDDEN, HIDDEN // 2, num_layers=2, bidirectional=True,
                            batch_first=True, dropout=0.3)
        self.fc = nn.Linear(HIDDEN, n_cls + 1)

    def forward(self, poses, lens):
        z, tl = self.enc(poses, lens)                      # (B, L, d), token lengths
        h = self.conv(z.transpose(1, 2)).transpose(1, 2)   # (B, L//2, H)
        fl = torch.div(tl, 2, rounding_mode="floor").clamp(min=1)
        aux = self.aux_fc(h)
        with torch.autocast(h.device.type, enabled=False):   # LSTM in fp32
            packed = nn.utils.rnn.pack_padded_sequence(h.float(), fl.cpu(), batch_first=True,
                                                       enforce_sorted=False)
            o, _ = self.lstm(packed)
            o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True,
                                                    total_length=h.size(1))
        return self.fc(o), aux, fl


def param_groups(model):
    from evals.finetune_islr_canon import _layer_id
    depth = model.enc.body.depth
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("enc."):
            lid = _layer_id(n[len("enc."):], depth)
            key, lr = ("enc", lid), LR * LLRD ** (depth + 1 - lid)
        else:
            key, lr = ("head", -1), LR
        g = groups.setdefault(key, {"params": [], "lr": lr, "base_lr": lr,
                                    "name": f"{key[0]}{key[1]}"})
        g["params"].append(p)
    return list(groups.values())


# ---- decoding -----------------------------------------------------------------------------

def greedy(logp, T):
    seq = logp[:T].argmax(-1).tolist()
    out, prev = [], 0
    for s in seq:
        if s != 0 and s != prev:
            out.append(s)
        prev = s
    return out


def _lse(a, b):
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def beam_search(logp, T, beam=BEAM, topn=TOPN):
    """CTC prefix beam search (blank = 0), no LM. logp: (T', V+1) log-softmax numpy."""
    beams = {(): (0.0, -np.inf)}                        # prefix -> (log p_blank, log p_nonblank)
    for t in range(T):
        row = logp[t]
        cand = np.argpartition(-row, min(topn, row.size - 1))[:topn]
        nxt = {}
        for prefix, (pb, pnb) in beams.items():
            tot = _lse(pb, pnb)
            for c in cand:
                p = float(row[c])
                if c == 0:
                    b, nb = nxt.get(prefix, (-np.inf, -np.inf))
                    nxt[prefix] = (_lse(b, tot + p), nb)
                    continue
                last = prefix[-1] if prefix else None
                ext = prefix + (int(c),)
                b, nb = nxt.get(ext, (-np.inf, -np.inf))
                nb = _lse(nb, (pb + p) if c == last else (tot + p))
                nxt[ext] = (b, nb)
                if c == last:                                  # repeat without blank: same prefix
                    b2, nb2 = nxt.get(prefix, (-np.inf, -np.inf))
                    nxt[prefix] = (b2, _lse(nb2, pnb + p))
        beams = dict(sorted(nxt.items(), key=lambda kv: -_lse(*kv[1]))[:beam])
    best = max(beams.items(), key=lambda kv: _lse(*kv[1]))[0]
    return list(best)


@torch.no_grad()
def decode(model, items, inv, device, bs=8):
    """-> ({utt: beam glosses}, {utt: greedy glosses}); clips without pose -> empty."""
    model.eval()
    hb, hg = {}, {}
    have = [it for it in items if it[1] is not None]
    for it in items:
        if it[1] is None:
            hb[it[0]] = []; hg[it[0]] = []
    for i in range(0, len(have), bs):
        chunk = have[i:i + bs]
        P, L = make_batch(chunk, model.enc.win)
        with torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
            logit, _, fl = model(P.to(device), L.to(device))
        logp = F.log_softmax(logit.float(), -1).cpu().numpy()
        for b, (uid, _, _) in enumerate(chunk):
            T = int(fl[b])
            hb[uid] = [inv[s] for s in beam_search(logp[b], T)]
            hg[uid] = [inv[s] for s in greedy(logp[b], T)]
    return hb, hg


def run(ckpt, tag, ds, seed=0, device="cuda", epochs=None, limit=0, max_steps=0, root=None):
    from evals.phoenix_wer import wer
    from foresign.results import write_result
    torch.backends.cuda.enable_cudnn_sdp(False)
    name = result_name(tag, ds, seed)
    ck_cfg = torch.load(ckpt, map_location="cpu", weights_only=False)["cfg"]
    in_dim = int(ck_cfg.get("in_dim", 3))
    t0 = time.time()
    tr = load_split(ds, "train", in_dim, root)
    dv = load_split(ds, "dev", in_dim, root)
    te = load_split(ds, "test", in_dim, root)
    cov = {s: {"ann": len(v), "found": sum(x is not None for _, x, _ in v)}
           for s, v in (("train", tr), ("dev", dv), ("test", te))}
    vocab = {g: i + 1 for i, g in enumerate(sorted({g for _, _, gl in tr for g in gl}))}
    inv = {i: g for g, i in vocab.items()}
    if limit:
        tr, dv, te = tr[:limit], dv[:max(8, limit // 4)], te[:max(8, limit // 4)]
    tr_have = [it for it in tr if it[1] is not None]
    print(f"[cosign] {name}: train {len(tr_have)} dev {len(dv)} test {len(te)} vocab "
          f"{len(vocab)} coverage {cov} (load {time.time() - t0:.0f}s)", flush=True)

    torch.manual_seed(seed)
    model = CoSignCSLR(ckpt, len(vocab)).to(device)
    opt = torch.optim.AdamW(param_groups(model), weight_decay=WD)
    n_ep = epochs or EPOCHS
    dtype = device.split(":")[0]
    best = (float("inf"), -1, None)
    hist, n_done, t_train = [], 0, time.time()
    for ep in range(n_ep):
        f = 0.1 ** sum(ep >= m for m in MILESTONES)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * f
        model.train()
        rng = np.random.default_rng([seed, ep])
        perm = rng.permutation(len(tr_have))
        tot = 0.0; k = 0
        for i in range(0, len(perm), BATCH):
            chunk = [tr_have[j] for j in perm[i:i + BATCH]]
            P, L, tg, tgl = make_batch(chunk, model.enc.win, vocab, rng)
            with torch.autocast(dtype, dtype=torch.bfloat16):
                logit, aux, fl = model(P.to(device), L.to(device))
            loss = 0.0
            for lg in (logit, aux):
                lp = F.log_softmax(lg.float(), -1).transpose(0, 1)
                loss = loss + F.ctc_loss(lp, tg.to(device), fl, tgl.to(device), blank=0,
                                         reduction="none", zero_infinity=True).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n_done += 1; k += 1; tot += loss.item()
            if max_steps and k >= max_steps:
                break
        rate = n_done / max(1e-6, time.time() - t_train)
        hb, hg = decode(model, dv, inv, device)
        w_b, w_g = wer(ds, "dev", hb)["wer"], wer(ds, "dev", hg)["wer"]
        hist.append({"epoch": ep + 1, "loss": tot / max(1, k), "dev_wer_beam": w_b,
                     "dev_wer_greedy": w_g})
        if w_b < best[0]:
            best = (w_b, ep + 1, {k_: v.detach().cpu().clone()
                                   for k_, v in model.state_dict().items()})
        print(f"[cosign] {name} ep{ep + 1}/{n_ep} loss {tot / max(1, k):.3f} dev WER beam "
              f"{w_b:.2f} greedy {w_g:.2f} (best {best[0]:.2f} @ep{best[1]}) {rate:.2f} it/s",
              flush=True)
    t_train_end = time.time()
    model.load_state_dict(best[2])
    hb, hg = decode(model, te, inv, device)
    tb, tg_ = wer(ds, "test", hb), wer(ds, "test", hg)
    print(f"[cosign] {name} FINAL dev {best[0]:.2f} (ep{best[1]}) test beam {tb['wer']:.2f} "
          f"greedy {tg_['wer']:.2f}", flush=True)
    from foresign.build import speed_flags
    out = {"task": "cslr_cosign", "recipe": "cslr_cosign", "ckpt": ckpt, "tag": tag,
           "dataset": ds, "seed": seed, "epochs": n_ep, "limit": limit, "max_steps": max_steps,
           "complete": not limit and not max_steps, "dev_wer_best": best[0],
           "best_epoch": best[1], "test_wer_at_best_dev": tb["wer"],
           "test_wer_greedy_at_best_dev": tg_["wer"], "test_counts": tb, "history": hist,
           "coverage": cov, "vocab": len(vocab),
           "train_it_per_s": n_done / max(1e-6, t_train_end - t_train),
           "fast_ops": all(speed_flags({"input_format": "c77"}).values()),
           "grad_ckpt": model.enc.body.grad_checkpoint,
           "hparams": {"lr": LR, "wd": WD, "llrd": LLRD, "batch": BATCH,
                       "milestones": list(MILESTONES), "hidden": HIDDEN, "beam": BEAM,
                       "topn": TOPN, "temporal_scale": TSCALE}}
    write_result(name, out)
    return out


def main():
    run(os.environ["FS_CC_CKPT"], os.environ["FS_TAG"], os.environ.get("FS_CC_DS", "ph14t"),
        int(os.environ.get("FS_CC_SEED", "0")), device=os.environ.get("FS_DEVICE", "cuda"),
        epochs=int(os.environ["FS_CC_EPOCHS"]) if os.environ.get("FS_CC_EPOCHS") else None,
        limit=int(os.environ.get("FS_CC_LIMIT", "0")),
        max_steps=int(os.environ.get("FS_CC_MAX_STEPS", "0")))


if __name__ == "__main__":
    main()
