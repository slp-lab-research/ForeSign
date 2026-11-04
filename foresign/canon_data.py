"""Phase 2 pretraining data: C77-3D canon HDF5 (one file per dataset, YouTube-ASL one per batch).

Each clip group has x (T,77,3) float16 at 25 fps and mask (T,77) uint8 (1 = joint present);
attrs split, no_pose, ... (canonicalisation spec §2). A training sample is a random clip_len
crop (partial H5 read), packed by foresign.inputs.pack_c77 into (clip_len, 77*(in_dim+1)) =
coordinates + joint mask; short clips are padded with zeros (mask 0).

Sampling: temperature over datasets as in Phase 1a. sample_unit="windows" (default) measures a
dataset, and weights a clip within it, by ceil(T/clip_len): How2Sign, OpenASL and YouTube-ASL
are whole videos now, so clip counts would under-represent them. sample_unit="clips" is the
Phase 1a rule (dataset weight = clip count, clips uniform).

The index (file, clip id, T, split attr) is cached per file under cache/ and rescanned only when
a file's size or mtime changes. The inductive filter (FS_INDUCTIVE, default on) is applied after
loading: foresign.canon_splits.
"""
from __future__ import annotations

import glob
import hashlib
import math
import os
import random

import h5py
import numpy as np
import torch

from foresign import paths
from foresign.canon_splits import DEFAULT_PRETRAIN, filter_canon_clips, load_exclude
from foresign.inputs import pack_c77

# canon dataset name -> Phase 1a language-map key (foresign/_vendor/resona/data/language_map.py)
LANG_ALIAS = {"chicagofswild": "chicago_fswild", "phoenix14": "phoenix", "phoenix14t": "phoenix",
              "youtube_asl": "yt_asl", "isign": "include"}


def canon_files(root: str, ds: str) -> list[str]:
    """Basenames of the H5 files holding `ds` (sorted)."""
    if ds == "youtube_asl":
        return sorted(os.path.basename(p) for p in glob.glob(os.path.join(root, "youtube_asl_b*.h5")))
    return [f"{ds}.h5"] if os.path.exists(os.path.join(root, f"{ds}.h5")) else []


def scan_file(path: str, min_frames: int):
    """-> ([(clip_id, T, split_attr)], canon_version)."""
    rows = []
    with h5py.File(path, "r") as f:
        ver = f.attrs.get("canon_version", "")
        for cid in f.keys():
            g = f[cid]
            if not isinstance(g, h5py.Group) or "x" not in g:
                continue
            if bool(g.attrs.get("no_pose", False)):
                continue
            T = int(g["x"].shape[0])
            if T >= min_frames:
                sp = g.attrs.get("split", "")
                rows.append((cid, T, sp.decode() if isinstance(sp, bytes) else str(sp)))
    return rows, (ver.decode() if isinstance(ver, bytes) else str(ver))


def _cache_path(root, cache_dir):
    h = hashlib.sha1(os.path.abspath(root).encode()).hexdigest()[:10]
    return os.path.join(cache_dir, f"canon_index_{h}.pt")


def load_or_build_canon_index(root, datasets, min_frames=24, cache_dir=None):
    """{"root", "files": {ds: [basename]}, "clips": {ds: [(file_idx, clip_id, T, split)]},
    "versions": {basename: canon_version}}. Missing dataset files raise."""
    cache_dir = cache_dir or paths.CACHE
    cp = _cache_path(root, cache_dir)
    cache = torch.load(cp, weights_only=False) if os.path.exists(cp) else {}
    changed = False
    out = {"root": root, "files": {}, "clips": {}, "versions": {}}
    missing = [ds for ds in datasets if not canon_files(root, ds)]
    if missing:
        raise FileNotFoundError(f"no canon file for {missing} under {root}")
    for ds in datasets:
        files = canon_files(root, ds)
        out["files"][ds] = files
        rows = []
        for fi, base in enumerate(files):
            p = os.path.join(root, base)
            st = os.stat(p)
            sig = (st.st_size, int(st.st_mtime))
            ent = cache.get(base)
            if ent is None or ent["sig"] != sig:
                r, ver = scan_file(p, 1)          # cache every clip; min_frames applied below
                ent = cache[base] = {"sig": sig, "rows": r, "version": ver}
                changed = True
                print(f"[canon-index] {base}: {len(r)} clips ({ver})", flush=True)
            out["versions"][base] = ent["version"]
            rows.extend((fi, cid, T, sp) for cid, T, sp in ent["rows"] if T >= min_frames)
        out["clips"][ds] = rows
    if changed:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = cp + ".tmp"
        torch.save(cache, tmp)
        os.replace(tmp, cp)
    return out


def lang_id(ds: str) -> int:
    from foresign._vendor.resona.data.language_map import (DATASET_LANGUAGE, LANGUAGE_TO_ID,
                                                           UNKNOWN_LANG_ID)
    lang = DATASET_LANGUAGE.get(LANG_ALIAS.get(ds, ds))
    return LANGUAGE_TO_ID.get(lang, UNKNOWN_LANG_ID) if lang else UNKNOWN_LANG_ID


def read_window(f: h5py.File, cid: str, start: int, length: int, in_dim: int,
                R: np.ndarray | None = None) -> np.ndarray:
    """Packed window; R (3,3) rotates the full xyz before packing (pretraining view aug)."""
    g = f[cid]
    x = g["x"][start:start + length]
    if R is not None:
        from foresign.augment import rotate
        x = rotate(x, R)
    return pack_c77(x, g["mask"][start:start + length], in_dim)


class CanonClips(torch.utils.data.IterableDataset):
    """Infinite stream: temperature-sample dataset -> clip -> random crop. Yields
    (pose (clip_len, 77*(in_dim+1)) float32, n_valid, lang_id, dataset_id)."""

    def __init__(self, datasets=None, clip_len=128, temperature=0.5, seed=0, root=None,
                 in_dim=3, sample_unit="windows", min_frames=24, exclude_path=None,
                 cache_dir=None, aug_rot_yaw_deg=0.0, aug_rot_pitch_deg=0.0):
        super().__init__()
        if sample_unit not in ("windows", "clips"):
            raise ValueError(f"sample_unit must be windows|clips, got {sample_unit!r}")
        self.root = root or paths.canon_root()
        self.datasets = list(datasets or DEFAULT_PRETRAIN)
        idx = load_or_build_canon_index(self.root, self.datasets, min_frames, cache_dir)
        self.files = idx["files"]
        self.versions = idx["versions"]
        clips = idx["clips"]
        if os.environ.get("FS_INDUCTIVE", "1") == "1":
            ex = load_exclude(exclude_path or paths.CANON_EXCLUDE)
            before = {d: len(v) for d, v in clips.items()}
            clips = filter_canon_clips(clips, ex)
            for d in before:
                print(f"[canon] inductive {d}: {before[d]} -> {len(clips.get(d, []))}", flush=True)
        self.index = clips
        self.names = [d for d in self.datasets if d in clips]
        self.clip_len, self.in_dim, self.seed = clip_len, in_dim, seed
        self.sample_unit = sample_unit
        # pretraining-only view augmentation (plan addendum 2026-11-02); 0 = off
        self.aug_yaw, self.aug_pitch = float(aug_rot_yaw_deg or 0), float(aug_rot_pitch_deg or 0)
        self._cum = {}
        sizes = []
        for d in self.names:
            if sample_unit == "windows":
                w = np.array([max(1, math.ceil(r[2] / clip_len)) for r in clips[d]], np.float64)
                self._cum[d] = np.cumsum(w).tolist()
                sizes.append(w.sum())
            else:
                self._cum[d] = None
                sizes.append(float(len(clips[d])))
        w = np.asarray(sizes) ** temperature
        self.weights = (w / w.sum()).tolist()
        self.name2id = {n: i for i, n in enumerate(self.names)}
        self.lang = {n: lang_id(n) for n in self.names}
        self._h5 = {}

    def _file(self, ds, fi):
        key = (ds, fi)
        if key not in self._h5:
            self._h5[key] = h5py.File(os.path.join(self.root, self.files[ds][fi]), "r")
        return self._h5[key]

    def _pick(self, ds, rng):
        rows = self.index[ds]
        if self._cum[ds] is None:
            return rng.choice(rows)
        return rng.choices(rows, cum_weights=self._cum[ds], k=1)[0]

    def sample(self, ds, row, rng):
        fi, cid, T, _ = row
        L = self.clip_len
        s = rng.randint(0, T - L) if T > L else 0
        from foresign.augment import random_rotation
        R, _, _ = random_rotation(rng, self.aug_yaw, self.aug_pitch)   # no draw when off
        arr = read_window(self._file(ds, fi), cid, s, L, self.in_dim, R)
        nv = arr.shape[0]
        if nv < L:
            arr = np.concatenate([arr, np.zeros((L - nv, arr.shape[1]), np.float32)], 0)
        return arr, nv

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (wi.id if wi else 0) * 9973)
        while True:
            ds = rng.choices(self.names, weights=self.weights, k=1)[0]
            row = self._pick(ds, rng)
            try:
                pose, nv = self.sample(ds, row, rng)
            except Exception as e:                       # unreadable clip: skip, keep going
                print(f"[canon] skip {row[1]}: {e}", flush=True)
                continue
            if not pose[:nv, self.in_dim::self.in_dim + 1].any():
                continue                                 # crop without any detected joint
            yield torch.from_numpy(pose), nv, self.lang[ds], self.name2id[ds]
