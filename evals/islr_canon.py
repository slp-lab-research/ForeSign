"""Frozen ISLR probes on C77-3D canon data (Phase 2): AUTSL, WLASL, LSA64, official splits.

Protocol: train on the official train
split, test on the official test split (val unused), classes = labels seen in train, centre crop
to clip_len, pooled features -> linear probe (3 seeds) + attentive probe (3 seeds) + kNN@5.
Splits come from the canon clip's split attribute (foresign.canon_splits); labels from the clip
id (WLASL gloss, LSA64 sign id) or autsl_labels.json (AUTSL; tools.build_autsl_labels, looked up
in $FS_ANN_DIR, then <project>/data_ann).

  FS_CANON_ROOT=<dir> FS_ADAPTER=foresign:<ckpt> FS_TAG=<tag> [FS_TAP_DEPTH=1] \
      python -m evals.islr_canon             -> results/islr_canon_<tag>.json
Env: FS_DATASETS=autsl,wlasl,lsa64  FS_SEEDS=0,1,2  FS_ISLR_LIMIT=<n per split, smoke only>
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from functools import lru_cache

import h5py
import numpy as np
import torch

from foresign import paths
from foresign.canon_splits import canon_split

PLAN = ["autsl", "wlasl", "lsa64"]


@lru_cache(maxsize=1)
def autsl_labels() -> dict:
    with open(paths.ann_file("autsl_labels.json"), encoding="utf-8") as f:
        return json.load(f)


def clip_label(ds: str, clip_id: str, autsl: dict | None = None):
    parts = clip_id.split("__")
    if ds == "autsl":
        name = parts[-1][:-len("_color")] if parts[-1].endswith("_color") else parts[-1]
        return (autsl if autsl is not None else autsl_labels()).get(f"{parts[1]}/{name}")
    if ds == "wlasl":
        return parts[-1].split("-wlasl-", 1)[1] if "-wlasl-" in parts[-1] else None
    if ds == "lsa64":
        return parts[-1].split("_")[0]
    raise ValueError(f"no canon ISLR protocol for {ds}")


def collect_canon(ds: str, root: str, cache_dir=None, limit: int = 0):
    """-> (train_items, test_items, n_classes); items (file_idx, clip_id, class_id)."""
    from foresign.canon_data import load_or_build_canon_index
    idx = load_or_build_canon_index(root, [ds], min_frames=1, cache_dir=cache_dir)
    tr, te = defaultdict(list), defaultdict(list)
    for fi, cid, _T, sp in sorted(idx["clips"][ds], key=lambda r: r[1]):
        s = canon_split(ds, cid, sp)
        if s not in ("train", "test"):
            continue
        lab = clip_label(ds, cid)
        if lab is not None:
            (tr if s == "train" else te)[lab].append((fi, cid))
    classes = sorted(tr)
    cid_of = {c: i for i, c in enumerate(classes)}
    train = [(fi, k, cid_of[c]) for c in classes for fi, k in tr[c]]
    test = [(fi, k, cid_of[c]) for c in sorted(te) if c in cid_of for fi, k in te[c]]
    dropped = sum(len(v) for c, v in te.items() if c not in cid_of)
    if limit:
        rng = np.random.default_rng(0)
        train = [train[i] for i in sorted(rng.permutation(len(train))[:limit])]
        test = [test[i] for i in sorted(rng.permutation(len(test))[:limit])]
    print(f"[collect] {ds}: train={len(train)} test={len(test)} classes={len(classes)} "
          f"test_dropped_unseen_class={dropped}", flush=True)
    return train, test, len(classes), idx["files"][ds]


@torch.no_grad()
def extract_canon(adapter, root, ds, files, items, bs=96):
    """Centre crop / zero-pad to adapter.clip_len -> pooled (N,d), time tokens (N,TP,d),
    time mask (N,TP), labels (N,)."""
    from foresign.inputs import pack_c77
    L = adapter.clip_len
    handles = {}
    pooled, toks, masks, labels = [], [], [], []
    buf, nvs, labs = [], [], []

    def flush():
        if not buf:
            return
        x = torch.from_numpy(np.stack(buf, 0)).to(adapter.device)
        nv = torch.tensor(nvs, device=adapter.device)
        pooled.append(adapter.pool(x, nv).float().cpu())
        tt, tm = adapter.time_tokens(x, nv)
        toks.append(tt.float().cpu()); masks.append(tm.cpu())
        labels.extend(labs)
        buf.clear(); nvs.clear(); labs.clear()

    for fi, cid, y in items:
        if fi not in handles:
            handles[fi] = h5py.File(os.path.join(root, files[fi]), "r")
        g = handles[fi][cid]
        T = g["x"].shape[0]
        s = (T - L) // 2 if T >= L else 0
        p = pack_c77(g["x"][s:s + L], g["mask"][s:s + L], adapter.in_dim)
        nv = p.shape[0]
        if nv < L:
            p = np.concatenate([p, np.zeros((L - nv, p.shape[1]), np.float32)], 0)
        buf.append(p); nvs.append(nv); labs.append(y)
        if len(buf) == bs:
            flush()
    flush()
    for h in handles.values():
        h.close()
    return (torch.cat(pooled), torch.cat(toks), torch.cat(masks),
            torch.tensor(labels, dtype=torch.long))


def main():
    from evals.probes import attentive_probe, knn_probe, linear_probe_full, make_adapter
    from foresign.inputs import C77
    device = os.environ.get("FS_DEVICE", "cuda")
    seeds = [int(s) for s in os.environ.get("FS_SEEDS", "0,1,2").split(",")]
    only = os.environ.get("FS_DATASETS")
    plan = [d for d in PLAN if not only or d in only.split(",")]
    limit = int(os.environ.get("FS_ISLR_LIMIT", "0"))
    root = paths.canon_root()
    adapter, tag = make_adapter(device)
    if getattr(adapter, "layout", None) is None or adapter.layout.fmt != C77:
        raise ValueError("islr_canon needs a c77 (canon) checkpoint")
    print(f"[islr-canon] tag={tag} root={root} seeds={seeds} limit={limit}", flush=True)
    rows = []
    for ds in plan:
        tr, te, n_cls, files = collect_canon(ds, root, limit=limit)
        ftr, ttr, mtr, ytr = extract_canon(adapter, root, ds, files, tr)
        fte, tte, mte, yte = extract_canon(adapter, root, ds, files, te)
        lin = [linear_probe_full(ftr, ytr, fte, yte, n_cls, device, s) for s in seeds]
        att = [attentive_probe(ttr, mtr, ytr, tte, mte, yte, n_cls, device, s) for s in seeds]
        knn = knn_probe(ftr, ytr, fte, yte, n_cls, device)
        l1 = torch.tensor([x[0] for x in lin]); a1 = torch.tensor(att)
        t5 = torch.tensor([x[1] for x in lin])
        pcs = torch.tensor([x[2].nanmean().item() for x in lin])
        sd = (lambda t: t.std().item() if len(t) > 1 else 0.0)
        row = {"dataset": ds, "n_cls": n_cls, "n_train": len(ytr), "n_test": len(yte),
               "linear_mean": l1.mean().item(), "linear_sd": sd(l1),
               "attentive_mean": a1.mean().item(), "attentive_sd": sd(a1), "knn5": knn,
               "linear_pc_mean": pcs.mean().item(), "linear_pc_sd": sd(pcs),
               "linear_top5_mean": t5.mean().item()}
        rows.append(row)
        print(f"[islr-canon] {tag} {ds:6s} cls={n_cls} linear={row['linear_mean']:.3f}"
              f"±{row['linear_sd']:.3f} attentive={row['attentive_mean']:.3f} knn@5={knn:.3f} "
              f"P-C={row['linear_pc_mean']:.3f}", flush=True)
    macro = sum(r["linear_mean"] for r in rows) / max(1, len(rows))
    print(f"macro linear top1 = {macro:.3f}", flush=True)
    from foresign.results import write_result
    write_result(f"islr_canon_{tag}", {
        "task": "islr_probe_canon", "split_mode": "official", "seeds": seeds,
        "canon_root": root, "tap_depth": getattr(adapter, "tap_depth", None),
        "limit": limit, "rows": rows, "macro_linear_top1": macro})
    return rows


if __name__ == "__main__":
    main()
