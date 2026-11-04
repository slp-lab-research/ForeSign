"""Dataset-ID probe on a frozen encoder (canonicalisation spec §9 mitigation).

Question: does the frozen encoder carry *which ASL dataset* a clip comes from? Language is held
fixed (all ASL), so separability = recording/extraction fingerprint. Protocol mirrors the
input-level fingerprint gate (spec §3): asl_citizen, wlasl, how2sign; N train-split clips per
dataset (numpy default_rng(seed=0), clips shorter than the window skipped), one random
WIN-frame window per clip (2 s at 25 fps), so clip length / isolated-vs-continuous cannot be
used. Features: padding-masked mean of the frozen encoder grid (adapter.pool; FS_TAP_DEPTH taps
a layer), plus an input-level baseline (window-mean of the packed coordinates and mask = the
gate's P2 "mean pose"). Probe: 5-fold stratified CV, standardised features, L2 logistic
regression C=1. Chance = 1/3.

  FS_CANON_ROOT=<dir> FS_DSID_CKPT=<ckpt> FS_TAG=<tag> [FS_TAP_DEPTH=1] python -m evals.dsid_probe
Env: FS_DSID_DATASETS (default asl_citizen,wlasl,how2sign), FS_DSID_N (300), FS_DSID_WIN (50)
-> results/dsid_<tag>.json
"""
from __future__ import annotations

import os

import h5py
import numpy as np
import torch

from foresign import paths

DEFAULT_DATASETS = ("asl_citizen", "wlasl", "how2sign")


def sample_windows(root, datasets, n=300, win=50, seed=0, cache_dir=None):
    """-> [(dataset, file_idx, clip_id, start)], deterministic and balanced (n per dataset)."""
    from foresign.canon_data import load_or_build_canon_index
    idx = load_or_build_canon_index(root, list(datasets), min_frames=win, cache_dir=cache_dir)
    rng = np.random.default_rng(seed)
    out = []
    for ds in datasets:
        rows = sorted((r for r in idx["clips"][ds] if r[3] == "train"), key=lambda r: r[1])
        if len(rows) < n:
            raise ValueError(f"{ds}: only {len(rows)} train clips >= {win} frames (need {n})")
        pick = sorted(rng.choice(len(rows), size=n, replace=False))
        for i in pick:
            fi, cid, T, _ = rows[i]
            out.append((ds, fi, cid, int(rng.integers(0, T - win + 1))))
    return out, idx["files"]


def linear_cv(X, y, folds=5, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    k = int(y.max()) + 1
    conf = np.zeros((k, k), np.int64)
    accs = []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        clf.fit(X[tr], y[tr])
        p = clf.predict(X[te])
        accs.append(float((p == y[te]).mean()))
        for a, b in zip(y[te], p):
            conf[a, b] += 1
    return {"acc_mean": float(np.mean(accs)), "acc_sd": float(np.std(accs)),
            "fold_acc": accs, "confusion": conf.tolist()}


@torch.no_grad()
def features(adapter, root, files, samples, win, bs=64):
    from foresign.inputs import pack_c77
    L = adapter.clip_len
    if win > L:
        raise ValueError(f"window {win} > encoder clip_len {L}")
    handles, enc, raw, buf = {}, [], [], []

    def flush():
        if buf:
            x = torch.from_numpy(np.stack(buf)).to(adapter.device)
            nv = torch.full((len(buf),), win, device=adapter.device)
            enc.append(adapter.pool(x, nv).float().cpu().numpy())
            buf.clear()
    for ds, fi, cid, s in samples:
        key = (ds, fi)
        if key not in handles:
            handles[key] = h5py.File(os.path.join(root, files[ds][fi]), "r")
        g = handles[key][cid]
        p = pack_c77(g["x"][s:s + win], g["mask"][s:s + win], adapter.in_dim)
        raw.append(p.mean(0))
        buf.append(np.concatenate([p, np.zeros((L - win, p.shape[1]), np.float32)], 0))
        if len(buf) == bs:
            flush()
    flush()
    for h in handles.values():
        h.close()
    return np.concatenate(enc), np.stack(raw)


def run(ckpt, tag, datasets=DEFAULT_DATASETS, n=300, win=50, seed=0, device="cuda",
        cache_dir=None, folds=5):
    from foresign.adapters import ForeSignAdapter
    from foresign.results import write_result
    root = paths.canon_root()
    adapter = ForeSignAdapter(ckpt, device=device)
    samples, files = sample_windows(root, datasets, n, win, seed, cache_dir)
    X_enc, X_raw = features(adapter, root, files, samples, win)
    y = np.array([list(datasets).index(s[0]) for s in samples])
    res = {"task": "dataset_id_probe", "ckpt": ckpt, "datasets": list(datasets),
           "n_per_dataset": {d: int((y == i).sum()) for i, d in enumerate(datasets)},
           "window": win, "seed": seed, "folds": folds, "chance": 1.0 / len(datasets),
           "tap_depth": adapter.tap_depth, "canon_root": root,
           "encoder": linear_cv(X_enc, y, folds, seed),
           "input_meanpose": linear_cv(X_raw, y, folds, seed)}
    print(f"[dsid] {tag}: encoder acc {res['encoder']['acc_mean']:.3f}±"
          f"{res['encoder']['acc_sd']:.3f} | input mean-pose {res['input_meanpose']['acc_mean']:.3f}"
          f" (chance {res['chance']:.3f})", flush=True)
    write_result(f"dsid_{tag}", res)
    return res


def main():
    ckpt = os.environ["FS_DSID_CKPT"]
    tag = os.environ.get("FS_TAG") or (os.path.basename(os.path.dirname(ckpt)) + "_" +
                                       os.path.basename(ckpt).replace(".pt", ""))
    ds = tuple(os.environ.get("FS_DSID_DATASETS", ",".join(DEFAULT_DATASETS)).split(","))
    run(ckpt, tag, ds, n=int(os.environ.get("FS_DSID_N", "300")),
        win=int(os.environ.get("FS_DSID_WIN", "50")),
        device=os.environ.get("FS_DEVICE", "cuda"))


if __name__ == "__main__":
    main()
