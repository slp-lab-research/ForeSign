"""INDEX.tsv + SHA256SUMS for a canon output directory.

  python -m tools.canon.index ~/canon/out/canon_v4 --workers 8

INDEX.tsv columns: file, dataset, batch, canon_version, code_git_sha, clips, frames (25 fps),
no_pose_clips, splits (split:count;...), bytes, sha256. SHA256SUMS is in `sha256sum -c` format
(relative names). Both are written atomically.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

COLS = ["file", "dataset", "batch", "canon_version", "code_git_sha", "clips", "frames",
        "no_pose_clips", "splits", "bytes", "sha256"]


def sha256_file(path: str, block: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _s(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def describe(path: str) -> dict:
    import h5py
    splits, frames, nop = Counter(), 0, 0
    with h5py.File(path, "r") as f:
        a = f.attrs
        row = dict(dataset=_s(a.get("dataset", "")), batch=_s(a.get("batch", "")),
                   canon_version=_s(a.get("canon_version", "")),
                   code_git_sha=_s(a.get("code_git_sha", "")))
        for cid in f.keys():
            g = f[cid]
            frames += g["x"].shape[0]
            splits[_s(g.attrs.get("split", "")) or "none"] += 1
            nop += bool(g.attrs.get("no_pose", False))
        row.update(clips=len(f.keys()))
    row.update(file=os.path.basename(path), frames=frames, no_pose_clips=nop,
               splits=";".join(f"{k}:{v}" for k, v in sorted(splits.items())),
               bytes=os.path.getsize(path), sha256=sha256_file(path))
    return row


def build(out_dir: str, workers: int = 4) -> list[dict]:
    out_dir = os.path.expanduser(out_dir)
    files = sorted(glob.glob(os.path.join(out_dir, "*.h5")))
    with ProcessPoolExecutor(max(1, workers)) as ex:
        rows = list(ex.map(describe, files))
    tmp = os.path.join(out_dir, "INDEX.tsv.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(COLS) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in COLS) + "\n")
    os.replace(tmp, os.path.join(out_dir, "INDEX.tsv"))
    tmp = os.path.join(out_dir, "SHA256SUMS.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(f"{r['sha256']}  {r['file']}\n")
    os.replace(tmp, os.path.join(out_dir, "SHA256SUMS"))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    rows = build(a.out_dir, a.workers)
    print(f"{len(rows)} files, {sum(r['clips'] for r in rows)} clips, "
          f"{sum(r['frames'] for r in rows)} frames")


if __name__ == "__main__":
    main()
