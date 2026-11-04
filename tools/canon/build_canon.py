"""Canonicalise one extraction unit (dataset or YouTube-ASL batch) to C77-3D HDF5.

  python -m tools.canon.build_canon --version canon-v4 \
      --src-root ~/reextract --ds asl_citizen --out <canon dir>/asl_citizen.h5 --workers 16
  ... --ds youtube_asl --batch b00001 --out <canon dir>/youtube_asl_b00001.h5
(canon-v4 is the version used for all Phase-2 data; the CLI default stays canon-v1.)

Reads every shard_*.h5 of <src-root>/out/<ds>[/<batch>]/ (format: extract_dataset.py) and
<src-root>/manifest/<ds>[_<batch>].csv for the split. Writes one group per clip:
  <clip_id>/x    (T',77,3) float16  C77-3D at 25 fps (tools/canon/canon.py)
  <clip_id>/mask (T',77)   uint8
  attrs: fps (25), fps_src, s, scale, yaw, n_in, no_pose, shoulder_floor, torso_degenerate,
         split, source, dataset, width, height, canon_version, code_git_sha
--version canon-v1 (default) | canon-v2a | canon-v2b (spec sections 2 and 6).
Resumable: each shard is first converted to <out>.parts/<shard>.h5 (+ .json stats),
written atomically; finished parts are skipped on restart. The parts are then merged into
<out>.tmp, renamed to <out>, and a summary <out minus .h5>_summary.json is written.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from tools.canon.canon import CANON_VERSION, FPS_OUT, PARTS, VERSIONS, canon_clip

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def code_git_sha() -> str:
    p = os.path.join(CODE_ROOT, "GIT_SHA")
    if os.path.exists(p):
        return open(p).read().strip()
    try:
        return subprocess.check_output(["git", "-C", CODE_ROOT, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def unit_dirs(src_root: str, ds: str, batch: str | None) -> tuple[str, str]:
    stem = f"{ds}_{batch}" if batch else ds
    out = os.path.join(src_root, "out", ds, batch) if batch else os.path.join(src_root, "out", ds)
    return out, os.path.join(src_root, "manifest", f"{stem}.csv")


def read_splits(manifest: str) -> dict:
    import csv
    if not os.path.exists(manifest):
        return {}
    with open(manifest, encoding="utf-8", newline="") as f:
        return {r["clip_id"]: r.get("split", "") for r in csv.DictReader(f)}


def _s(v):
    return v.decode() if isinstance(v, bytes) else v


def convert_shard(shard: str, part: str, splits: dict, sha: str,
                  version: str = CANON_VERSION) -> dict:
    """shard -> part h5 (atomic) + part.json stats. Returns the stats."""
    import h5py
    stats = []
    tmp = part + ".tmp"
    with h5py.File(shard, "r") as fi, h5py.File(tmp, "w") as fo:
        for cid in fi.keys():
            g = fi[cid]
            a = g.attrs
            x, m, info = canon_clip(g["kp"], g["world"][()], g["flags"][()],
                                    float(a["width"]), float(a["height"]), float(a["fps"]),
                                    version=version)
            comp = dict(compression="gzip", compression_opts=4, shuffle=True) if len(x) else {}
            go = fo.create_group(cid)
            go.create_dataset("x", data=x.astype(np.float16), **comp)
            go.create_dataset("mask", data=m, **comp)
            split = splits.get(cid, _s(a.get("split", "")))
            go.attrs.update(dict(
                fps=FPS_OUT, fps_src=info["fps_src"], s=info["s"], scale=info["scale"],
                yaw=info["yaw"], n_in=info["n_in"], no_pose=info["no_pose"],
                shoulder_floor=info["shoulder_floor"], split=split or "",
                source=_s(a.get("source", "")), dataset=_s(a.get("dataset", "")),
                width=int(a["width"]), height=int(a["height"]),
                canon_version=version, code_git_sha=sha, torso_degenerate=info["torso_degenerate"]))
            stats.append(dict(clip=cid, n_in=info["n_in"], n_out=info["n_out"],
                              no_pose=info["no_pose"], floor=info["shoulder_floor"],
                              degenerate=info["torso_degenerate"],
                              rate=info["mask_rate"]))
    os.replace(tmp, part)
    with open(part + ".json.tmp", "w") as f:
        json.dump(stats, f)
    os.replace(part + ".json.tmp", part + ".json")
    return stats


def _job(args):
    return convert_shard(*args)


def summarise(stats: list[dict]) -> dict:
    n_out = sum(s["n_out"] for s in stats)
    rates = {k: (sum(s["rate"][k] * s["n_out"] for s in stats) / n_out if n_out else 0.0)
             for k in PARTS}
    return dict(clips=len(stats), frames_in=sum(s["n_in"] for s in stats), frames_out=n_out,
                no_pose_clips=sum(s["no_pose"] for s in stats),
                no_pose_pct=100.0 * sum(s["no_pose"] for s in stats) / max(len(stats), 1),
                shoulder_floor_clips=sum(s["floor"] for s in stats),
                torso_degenerate_clips=sum(s.get("degenerate", False) for s in stats),
                mask_rate_frames=rates)


def merge(parts: list[str], out: str, attrs: dict) -> None:
    import h5py
    tmp = out + ".tmp"
    with h5py.File(tmp, "w") as fo:
        fo.attrs.update(attrs)
        for p in parts:
            with h5py.File(p, "r") as fi:
                for cid in fi.keys():
                    fi.copy(fi[cid], fo, name=cid)
    os.replace(tmp, out)


def run(src_root: str, ds: str, out: str, batch: str | None = None, workers: int = 1,
        log=print, version: str = CANON_VERSION) -> dict:
    src_root = os.path.expanduser(src_root)
    out = os.path.expanduser(out)
    shard_dir, manifest = unit_dirs(src_root, ds, batch)
    shards = sorted(glob.glob(os.path.join(shard_dir, "shard_*.h5")))
    if not shards:
        raise FileNotFoundError(f"no shards in {shard_dir}")
    splits = read_splits(manifest)
    sha = code_git_sha()
    pdir = out + ".parts"
    os.makedirs(pdir, exist_ok=True)
    for p in glob.glob(os.path.join(pdir, "*.tmp")):
        os.remove(p)
    parts = [os.path.join(pdir, os.path.basename(s)) for s in shards]
    todo = [(s, p, splits, sha, version) for s, p in zip(shards, parts)
            if not (os.path.exists(p) and os.path.exists(p + ".json"))]
    log(f"{ds}{'/' + batch if batch else ''}: {len(shards)} shards, {len(todo)} to convert")
    t0 = time.time()
    if workers > 1 and len(todo) > 1:
        with ProcessPoolExecutor(min(workers, len(todo))) as ex:
            for i, st in enumerate(ex.map(_job, todo), 1):
                log(f"  {i}/{len(todo)} shards ({len(st)} clips) {time.time() - t0:.0f}s")
    else:
        for i, j in enumerate(todo, 1):
            st = _job(j)
            log(f"  {i}/{len(todo)} shards ({len(st)} clips) {time.time() - t0:.0f}s")
    stats = []
    for p in parts:
        with open(p + ".json") as f:
            stats += json.load(f)
    summ = summarise(stats)
    summ.update(dataset=ds, batch=batch or "", src=shard_dir, out=out,
                canon_version=version, code_git_sha=sha, shards=len(shards),
                convert_s=round(time.time() - t0, 1))
    merge(parts, out, dict(canon_version=version, code_git_sha=sha, dataset=ds,
                           batch=batch or "", fps=FPS_OUT))
    sp = os.path.splitext(out)[0] + "_summary.json"
    with open(sp + ".tmp", "w") as f:
        json.dump(summ, f, indent=1)
    os.replace(sp + ".tmp", sp)
    shutil.rmtree(pdir)
    log(json.dumps(summ))
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src-root", required=True, help="REEXTRACT_ROOT (has out/ and manifest/)")
    ap.add_argument("--ds", required=True)
    ap.add_argument("--batch", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--version", default=CANON_VERSION, choices=VERSIONS)
    a = ap.parse_args(argv)
    if os.path.exists(os.path.expanduser(a.out)):
        print(f"{a.out} exists; nothing to do")
        return
    run(a.src_root, a.ds, a.out, a.batch, a.workers, log=lambda s: print(s, flush=True),
        version=a.version)


if __name__ == "__main__":
    main()
