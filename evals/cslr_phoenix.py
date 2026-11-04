"""Frozen-probe / fine-tuning CSLR on canonicalised (C77-3D) PHOENIX-2014T and PHOENIX-2014.
WER metric (greedy CTC; the official-protocol scorer is used by evals.cslr_cosign).

Pipeline (windowed to respect the encoder's pretraining temporal range):
    canon pose (T, 77*(in_dim+1)) -> 128-frame non-overlapping windows -> ForeSign encoder per
    window -> clip_len/t_patch tokens per window -> concat along time -> (B, T', d)
    -> temporal conv stack + BiLSTM -> Linear(vocab+1) -> CTC

Why windows: the encoder is pretrained at clip_len=128; later temporal positions are never
trained. Concatenating the token streams of non-overlapping windows gives unbounded continuous
length.

Data: FS_CSLR_DATASET=canon_ph14t (default) | canon_ph14 reads $FS_CANON_ROOT/phoenix14t.h5 |
phoenix14.h5, maps annotation ids to canon clip ids (missing clips are scored as full deletions
and counted in `coverage`), packs coords + mask per the checkpoint's in_dim. Needs a c77
checkpoint. Annotations phoenix14t_{split}.json / phoenix14_{split}.json (list of
{id, gloss}; built by tools.build_phoenix_ann) are looked up in $FS_ANN_DIR, then
<project>/data_ann.

  FS_CANON_ROOT=<dir> FS_CSLR_CKPT=<ckpt> FS_CSLR_EPOCHS=40 FS_CSLR_FREEZE=1 \
      python -m evals.cslr_phoenix
"""
from __future__ import annotations

import json
import math
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from foresign import paths

WIN = 128          # frames per window (= pretraining clip_len)

# FS_CSLR_DATASET selects the benchmark (C77-3D canon files under FS_CANON_ROOT); non-default
# datasets append a suffix to the result tag.
DATASETS = {
    "canon_ph14t": {"ann": "phoenix14t_{}.json", "canon": "phoenix14t", "suffix": "",
                    "cache": None, "task": "phoenix14t_cslr_canon"},
    "canon_ph14": {"ann": "phoenix14_{}.json", "canon": "phoenix14", "suffix": "_ph14",
                   "cache": None, "task": "phoenix14_cslr_canon"},
}


def is_canon(ds=None):
    return "canon" in DATASETS[ds or dataset_name()]


def check_compat(cfg, ds):
    """A c77 checkpoint needs canon data and vice versa."""
    c77 = (cfg.get("input_format") or "resona77") == "c77"
    if c77 != is_canon(ds):
        raise ValueError(f"checkpoint input_format={cfg.get('input_format') or 'resona77'} "
                         f"does not match FS_CSLR_DATASET={ds}")


def coverage(split_items):
    """{split: {"ann": n annotations, "found": n with a pose clip}}."""
    return {sp: {"ann": len(it), "found": sum(k is not None for k, _ in it)}
            for sp, it in split_items.items()}


def dataset_name():
    ds = os.environ.get("FS_CSLR_DATASET", "").strip() or "canon_ph14t"
    if ds not in DATASETS:
        raise ValueError(f"unknown FS_CSLR_DATASET={ds}; choose from {sorted(DATASETS)}")
    return ds


def ann_path(split, ds=None):
    return paths.ann_file(DATASETS[ds or dataset_name()]["ann"].format(split))


def h5_path(ds=None):
    d = DATASETS[ds or dataset_name()]
    return os.path.join(paths.canon_root(), d["canon"] + ".h5")


def canon_keys(ds):
    """annotation id -> canon clip id (clip ids are <dataset>__<split folder>__<id>)."""
    out = {}
    with h5py.File(h5_path(ds), "r") as f:
        for k in f.keys():
            out.setdefault(k.split("__", 2)[-1], k)
    return out


def result_tag(tag, ds=None):
    suf = DATASETS[ds or dataset_name()]["suffix"]
    return tag if not suf or tag.endswith(suf) else tag + suf


def load_split(split, vocab=None, ds=None):
    ds = ds or dataset_name()
    ann = json.load(open(ann_path(split, ds), encoding="utf-8"))
    keys = canon_keys(ds)               # key None = no pose clip (scored as full deletion)
    items = [(keys.get(it["id"]), it["gloss"].split()) for it in ann]
    if vocab is None:
        toks = set()
        for _, g in items:
            toks.update(g)
        # index 0 reserved for CTC blank
        vocab = {t: i + 1 for i, t in enumerate(sorted(toks))}
    return items, vocab


def preload(items, vocab, h5path, in_dim):
    """Load every clip ONCE into RAM. Store entries are (x, train_targets, ref):
    train_targets drop OOV glosses (cannot be trained on); ref keeps the FULL
    reference for scoring. Clips without pose keep x=None and are scored as full deletions.
    Canon input: coords + mask packed per in_dim (no further normalisation)."""
    from foresign.inputs import pack_c77
    f = h5py.File(h5path, "r")
    store = []
    for key, gloss in items:
        if key is None or key not in f:
            store.append((None, [], list(gloss))); continue
        g = f[key]
        x = pack_c77(g["x"][:], g["mask"][:], in_dim)
        cap = max_frames()
        store.append((x[:cap] if cap else x, [vocab[t] for t in gloss if t in vocab],
                      list(gloss)))
    f.close()
    return store


def max_frames():
    """Frame cap per clip: 384 for PHOENIX-2014T (longest clip is 329 frames, so it never
    binds); PHOENIX-2014 is not truncated. FS_CSLR_MAXLEN overrides (0 = no cap)."""
    env = os.environ.get("FS_CSLR_MAXLEN")
    if env is not None:
        return int(env) or None
    return 384 if (os.environ.get("FS_CSLR_DATASET", "").strip() or "canon_ph14t") in (
        "ph14t", "canon_ph14t") else None


def edit_distance(a, b):
    """Levenshtein on token lists."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1,
                        prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]


class WindowEncoder(nn.Module):
    """ForeSign online encoder, windowed; -> (B, Tmax_tok, d) + length mask."""

    def __init__(self, ckpt, which="ema", freeze=False):
        super().__init__()
        torch.backends.cuda.enable_cudnn_sdp(False)
        import os as _os
        from foresign.build import build_model
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        m = build_model(cfg)
        # FS_CSLR_RANDOM=1 -> keep fresh random init (ablation: does SSL pretraining
        # capture alignment-relevant structure for continuous recognition?)
        if _os.environ.get("FS_CSLR_RANDOM", "0") != "1":
            m.load_state_dict(ck["model"], strict=False)   # boundary_head absent in pre-BA ckpts
        else:
            print("[cslr] RANDOM-INIT encoder (ablation, no pretrained weights)", flush=True)
        self.tok = m.target_tok if which == "ema" else m.tokenizer
        if _os.environ.get("FS_CONF_GATE") == "0":
            self.tok.conf_gate = False
            print("[cslr] tokenizer conf_gate disabled", flush=True)
        self.body = m.target_ctx if which == "ema" else m.context
        self.freeze = freeze
        for p in self.parameters():
            p.requires_grad_(not freeze)
        self.body.grad_checkpoint = not freeze
        self.t_patch = cfg["t_patch"]
        self.d_model = cfg["d_model"]
        from foresign.confmode import resolve_drop_conf
        self.drop_conf = resolve_drop_conf(cfg)
        from foresign.inputs import layout_for
        self.layout = layout_for(cfg)
        self.in_dim = int(cfg.get("in_dim", 3))
        print(f"[cslr] drop_conf={self.drop_conf}", flush=True)
        self.win = int(cfg.get("clip_len", WIN))         # pretraining clip length
        self.win_tok = self.win // cfg["t_patch"]
        # FS_TAP_DEPTH=N (1..depth) taps the DSTformer at layer N instead of the final
        # layer. Login-node probes showed sign content peaks at L1-L6 and is DILUTED by the
        # deep layers/norm_final (which specialize for the JEPA pretext), so an intermediate
        # tap should give much better CSLR than the default final tap. 0/unset = final.
        td = int(_os.environ.get("FS_TAP_DEPTH", "0"))
        self._tap_depth = td if 0 < td < self.body.depth else None
        if self._tap_depth is not None:
            print(f"[cslr] tapping DSTformer at layer {self._tap_depth}/{self.body.depth} "
                  f"(intermediate; not final+norm)", flush=True)

    def _encode_win(self, pose):                       # pose (N,win,234) -> (N,win_tok,d)
        h = self.tok(pose)                             # (N,32,18,d)
        if self._tap_depth is None:
            return self.body(h).mean(dim=2)            # full body + norm_final
        N, TP, V, d = h.shape                          # intermediate tap
        x = h.reshape(N * TP, V, d)
        body = self.body
        for li, (st, ts, att) in enumerate(zip(body.blocks_st, body.blocks_ts, body.ts_attn), 1):
            x = body._layer_forward(x, TP, st, ts, att)
            if li == self._tap_depth:
                break
        return x.reshape(N, TP, V, d).mean(dim=2)

    def forward(self, poses, lengths):
        """poses: (B, Tpad, 234) padded to a multiple of WIN; lengths: (B,) true frames.
        Fully BATCHED: reshape whole batch into (B*nwin, WIN, 234), encode in one pass,
        reshape back to (B, nwin*32, d). Padding windows yield tokens masked out by
        the returned per-sample token lengths (CTC ignores them)."""
        B, Tpad, _ = poses.shape
        if self.drop_conf:                             # match the encoder's training input
            from foresign.confmode import zero_conf
            poses = zero_conf(poses)
        nwin = Tpad // self.win
        wins = poses.reshape(B * nwin, self.win, poses.shape[-1])
        z = self._encode_win(wins)                          # (B*nwin, 32, d)
        z = z.reshape(B, nwin * self.win_tok, self.d_model)  # (B, nwin*32, d)
        tok_lens = torch.div(lengths + self.t_patch - 1, self.t_patch,
                             rounding_mode="floor").clamp(max=nwin * self.win_tok)
        return z, tok_lens


class CSLR(nn.Module):
    """Standard CSLR recipe: encoder -> temporal context module -> CTC.
    The minimal per-token Linear head plateaus (no gloss-level temporal modeling
    + independent windows sever cross-window context). Here a 1D-temporal-conv
    stack + BiLSTM over the full concatenated token sequence models temporal
    dependencies AND stitches the windows. Encoder frozen by default to curb
    overfit on small PHOENIX (7096 clips)."""

    def __init__(self, enc, vocab_size, hidden=512):
        super().__init__()
        self.enc = enc
        d = enc.d_model
        self.in_norm = nn.LayerNorm(d)
        # temporal conv stack (kernel 5, two layers) — local gloss-scale context
        self.tconv = nn.Sequential(
            nn.Conv1d(d, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.lstm = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(2 * hidden, vocab_size + 1)    # +1 blank@0

    def forward(self, poses, lengths):
        z, tl = self.enc(poses, lengths)                   # (B,L,d)
        z = self.in_norm(z)
        z = self.tconv(z.transpose(1, 2)).transpose(1, 2)  # (B,L,hidden)
        z, _ = self.lstm(z)                                # (B,L,2*hidden)
        return self.fc(z), tl


def make_batch(store, device, idx, win=WIN):
    poses, lens, targets, tgt_lens = [], [], [], []
    for i in idx:
        x, tg = store[i][0], store[i][1]
        if x is None:
            continue
        poses.append(x); lens.append(x.shape[0])
        targets.append(tg); tgt_lens.append(len(tg))
    if not poses:
        return None
    Tpad = max(int(math.ceil(l / win)) * win for l in lens)
    P = np.zeros((len(poses), Tpad, poses[0].shape[1]), np.float32)
    for b, x in enumerate(poses):
        P[b, :x.shape[0]] = x
    flat_t = torch.tensor([t for tg in targets for t in tg], dtype=torch.long)
    return (torch.from_numpy(P).to(device), torch.tensor(lens, device=device),
            flat_t.to(device), torch.tensor(tgt_lens, device=device))


def _reference(rec, inv, legacy_oov_drop=False):
    """Scoring reference: full gloss list (default) or legacy OOV-dropped list."""
    if legacy_oov_drop:
        return [inv[t] for t in rec[1]]
    return rec[2]


@torch.no_grad()
def decode_wer(model, store, inv, device, bs=8, legacy_oov_drop=False):
    model.eval()
    tot_err = tot_len = 0
    for rec in store:                             # unscorable clips = full deletions
        x, ref = rec[0], _reference(rec, inv, legacy_oov_drop)
        if x is None:
            tot_err += len(ref); tot_len += max(1, len(ref))
    for i in range(0, len(store), bs):
        idx = [j for j in range(i, min(i + bs, len(store))) if store[j][0] is not None]
        if not idx:
            continue
        P, L, _, _ = make_batch(store, device, idx, model.enc.win)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit, tl = model(P, L)
        pred = logit.float().argmax(-1).cpu().numpy()         # (B,Ltok)
        for b, j in enumerate(idx):
            seq = pred[b, :int(tl[b].item())]
            out, prev = [], 0
            for s in seq:                                     # CTC collapse
                if s != 0 and s != prev:
                    out.append(inv.get(int(s), "?"))
                prev = s
            ref = _reference(store[j], inv, legacy_oov_drop)
            tot_err += edit_distance(out, ref); tot_len += max(1, len(ref))
    return tot_err / tot_len


def run():
    ckpt = os.environ["FS_CSLR_CKPT"]
    which = os.environ.get("FS_CSLR_WHICH", "ema")
    freeze = os.environ.get("FS_CSLR_FREEZE", "0") == "1"
    epochs = int(os.environ.get("FS_CSLR_EPOCHS", "40"))
    bs = int(os.environ.get("FS_CSLR_BS", "8"))
    legacy_enc_train = os.environ.get("FS_CSLR_LEGACY_ENC_TRAIN", "0") == "1"
    legacy_oov_drop = os.environ.get("FS_CSLR_LEGACY_OOV_DROP", "0") == "1"
    device = "cuda"
    seed = int(os.environ.get("FS_CSLR_SEED", "0"))
    torch.manual_seed(seed); np.random.seed(seed)
    ds = dataset_name()
    h5 = h5_path(ds)
    print(f"[cslr] dataset={ds} h5={h5} ann={ann_path('train', ds)}", flush=True)

    ck_cfg = torch.load(ckpt, map_location="cpu", weights_only=False)["cfg"]
    check_compat(ck_cfg, ds)
    in_dim = int(ck_cfg.get("in_dim", 3))
    tr_items, vocab = load_split("train", ds=ds)
    dev_items, _ = load_split("dev", vocab, ds=ds)
    test_items, _ = load_split("test", vocab, ds=ds)
    cov = coverage({"train": tr_items, "dev": dev_items, "test": test_items})
    print(f"[cslr] annotation coverage {cov}", flush=True)
    if os.environ.get("FS_CSLR_COVERAGE_ONLY") == "1":
        return cov
    lim = int(os.environ.get("FS_CSLR_LIMIT", "0"))
    if lim:
        tr_items = tr_items[:lim]; dev_items = dev_items[:max(16, lim // 4)]
        test_items = test_items[:max(16, lim // 4)]
    inv = {v: k for k, v in vocab.items()}
    print(f"[cslr] loading clips into RAM ...", flush=True)
    tr_store = preload(tr_items, vocab, h5, in_dim)
    dev_store = preload(dev_items, vocab, h5, in_dim)
    test_store = preload(test_items, vocab, h5, in_dim)
    print(f"[cslr] train={len(tr_items)} dev={len(dev_items)} test={len(test_items)} "
          f"vocab={len(vocab)} freeze={freeze} epochs={epochs}", flush=True)

    enc = WindowEncoder(ckpt, which, freeze).to(device)
    model = CSLR(enc, len(vocab)).to(device)
    enc_ids = {id(p) for p in enc.parameters()}
    enc_params = [p for p in enc.parameters() if p.requires_grad]
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    groups = [{"params": head_params, "lr": 1e-3}]
    if enc_params:
        groups.append({"params": enc_params, "lr": 1e-5})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    n = len(tr_items)
    steps = epochs * math.ceil(n / bs); warm = max(1, steps // 20)

    def lr_scale(s):
        return s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))

    base = [g["lr"] for g in groups]
    g = 0; best = 1e9; best_ep = -1; best_state = None
    for ep in range(epochs):
        model.train()
        if freeze and not legacy_enc_train:
            model.enc.eval()                  # no drop_path noise in a frozen encoder
        perm = np.random.permutation(n)
        tot = 0.0
        for i in range(0, n, bs):
            batch = make_batch(tr_store, device, perm[i:i + bs].tolist(), model.enc.win)
            if batch is None:
                continue
            P, L, tgt, tgl = batch
            for gi, gg in enumerate(opt.param_groups):
                gg["lr"] = base[gi] * lr_scale(g)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit, tl = model(P, L)
            logp = F.log_softmax(logit.float(), dim=-1).transpose(0, 1)  # (L,B,V+1)
            loss = F.ctc_loss(logp, tgt, tl, tgl, blank=0, zero_infinity=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); g += 1; tot += loss.item() * P.size(0)
        dev_wer = decode_wer(model, dev_store, inv, device, bs, legacy_oov_drop)
        if dev_wer < best:
            best, best_ep = dev_wer, ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[cslr] ep{ep+1}/{epochs} loss={tot/n:.3f} dev_WER={dev_wer:.4f} best={best:.4f}", flush=True)
    test_wer_last = decode_wer(model, test_store, inv, device, bs, legacy_oov_drop)
    if best_state is not None:
        model.load_state_dict(best_state)
    test_wer = decode_wer(model, test_store, inv, device, bs, legacy_oov_drop)
    n_missing = {s: sum(r[0] is None for r in st) for s, st in
                 (("train", tr_store), ("dev", dev_store), ("test", test_store))}
    print(f"[cslr] FINAL dev_best_WER={best:.4f} (ep{best_ep}) test_WER@best_dev={test_wer:.4f} "
          f"test_WER@last={test_wer_last:.4f} missing={n_missing}", flush=True)
    from foresign.results import write_result
    tag = os.environ.get("FS_TAG") or (os.path.basename(os.path.dirname(ckpt)) + "_" +
                                       os.path.basename(ckpt).replace(".pt", ""))
    tag = result_tag(tag, ds)
    write_result(f"cslr_{'frozen' if freeze else 'ft'}_{tag}_s{seed}", {
        "task": DATASETS[ds]["task"], "dataset": ds, "ckpt": ckpt, "freeze": freeze, "epochs": epochs,
        "head_seed": seed, "drop_conf": enc.drop_conf, "dev_wer_best": best,
        "best_epoch": best_ep, "test_wer_at_best_dev": test_wer,
        "test_wer_last": test_wer_last, "missing_clips": n_missing, "coverage": cov,
        "tap_depth": enc._tap_depth,
        "legacy_enc_train": legacy_enc_train, "legacy_oov_drop": legacy_oov_drop,
    })


if __name__ == "__main__":
    run()
