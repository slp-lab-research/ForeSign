"""Post-verification cleanup (spec section 4, step 4).

  python -m tools.extract.cleanup include --scratch          # remove our unpacked copies
  python -m tools.extract.cleanup <ds> --delete-videos       # only downloaded_by_us rows
  python -m tools.extract.cleanup youtube_asl --batch b00001 --delete-videos

Both actions require reports/<ds>_verify.json with status "pass".

--scratch        removes only files (frame folders: the folder) under scratch/<ds>/ that
                 the manifest marks unpacked_to_scratch=1 (our own extracted copies of
                 archive members).
                 The original archives are never touched (and must still exist).
--delete-videos  off by default. Deletes a source video only if the manifest row has
                 downloaded_by_us=1 (fetched by our own download step), the file lies
                 under the staging dir of that step (registry download_root, default
                 <root>/downloads/<ds>), and it is not an archive member.
                 Anything else (every dataset that was already on the machines) is
                 refused, as are clips listed in manifest/<stem>_exclude.csv (not
                 extracted). A registry origin_archive (our unpacked copy of an archive
                 held elsewhere) must still exist. Use --dry-run to list what would be
                 deleted.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

from tools.extract.build_manifest import read_manifest
from tools.extract.datasets import (DATASETS, load_excludes, scratch_root as _scratch_root,
                                    unit_paths, wpath)


def _verified(ds: str, report: str | None = None) -> bool:
    report = report or wpath("reports", f"{ds}_verify.json")
    if not os.path.exists(report):
        return False
    with open(report) as f:
        return json.load(f).get("status") == "pass"


def _under(path: str, root: str) -> bool:
    path, root = os.path.realpath(path), os.path.realpath(root)
    return os.path.commonpath([path, root]) == root and path != root


def clean_scratch(ds: str, manifest: str | None = None, scratch_root: str | None = None,
                  report: str | None = None, dry_run: bool = False, log=print) -> int:
    if not _verified(ds, report):
        raise RuntimeError(f"{ds}: verification has not passed; nothing removed")
    rows = read_manifest(manifest or wpath("manifest", f"{ds}.csv"))
    root = os.path.join(scratch_root or _scratch_root(), ds)
    n = 0
    for r in rows:
        if not r["unpacked_to_scratch"]:
            continue
        p = r["path"]
        if not _under(p, root):
            raise RuntimeError(f"refusing: {p} is not under {root}")
        arc = r["source"].split("::")[0]
        if "::" not in r["source"] or not os.path.exists(arc):
            raise RuntimeError(f"refusing: original archive for {r['clip_id']} not found ({arc})")
        if os.path.isdir(p):  # unpacked frame folder (tar_frames)
            if not dry_run:
                shutil.rmtree(p)
            n += 1
        elif os.path.exists(p):
            if not dry_run:
                os.remove(p)
            n += 1
    if not dry_run and os.path.isdir(root):
        for dp, dns, fns in os.walk(root):  # indexes written by unpack_tar / unpack_tar_frames
            for idx in ("_members.tsv", "_frames.tsv"):
                if idx in fns and os.path.basename(os.path.dirname(dp)) == "_tar":
                    os.remove(os.path.join(dp, idx))
        for dp, dns, fns in os.walk(root, topdown=False):
            if not os.listdir(dp):
                os.rmdir(dp)
    log(f"[{ds}] scratch: {'would remove' if dry_run else 'removed'} {n} extracted copies")
    return n


def delete_videos(ds: str, manifest: str | None = None, report: str | None = None,
                  download_root: str | None = None, dry_run: bool = False, log=print) -> int:
    if not _verified(ds, report):
        raise RuntimeError(f"{ds}: verification has not passed; nothing deleted")
    manifest = manifest or wpath("manifest", f"{ds}.csv")
    rows = read_manifest(manifest)
    excl = load_excludes(manifest)
    droot = (download_root or DATASETS.get(ds, {}).get("download_root")
             or os.path.join(wpath("downloads"), ds))
    origin = DATASETS.get(ds, {}).get("origin_archive")
    if origin and not os.path.exists(origin):
        raise RuntimeError(f"{ds}: origin archive {origin} not found; nothing deleted")
    refused = [r["clip_id"] for r in rows if not r["downloaded_by_us"] or r["clip_id"] in excl]
    n = 0
    for r in rows:
        if not r["downloaded_by_us"] or r["clip_id"] in excl:
            continue
        p = r["source"]
        if "::" in p or not _under(p, droot):
            refused.append(r["clip_id"])
            continue
        if os.path.exists(p):
            if not dry_run:
                os.remove(p)
            n += 1
    log(f"[{ds}] delete-videos: {'would delete' if dry_run else 'deleted'} {n}; "
        f"refused {len(refused)} (not downloaded by us)")
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(DATASETS))
    ap.add_argument("--scratch", action="store_true")
    ap.add_argument("--delete-videos", action="store_true", default=False)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch", help="queue datasets: the batch (manifest/<ds>_<batch>.csv)")
    a = ap.parse_args(argv)
    if not (a.scratch or a.delete_videos):
        ap.error("nothing to do (use --scratch and/or --delete-videos)")
    try:
        up = unit_paths(a.dataset, a.batch)
        if a.scratch:
            clean_scratch(a.dataset, manifest=up["manifest"], report=up["report"],
                          dry_run=a.dry_run)
        if a.delete_videos:
            delete_videos(a.dataset, manifest=up["manifest"], report=up["report"],
                          dry_run=a.dry_run)
    except RuntimeError as e:
        print("cleanup:", e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
