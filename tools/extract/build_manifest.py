"""Enumerate a dataset's sources and write manifest/<ds>.csv (spec section 4).

  nice -n 19 python -m tools.extract.build_manifest autsl --workers 48

One row per clip, sorted by clip_id (stable across runs; see datasets.make_clip_id).
Columns:
  clip_id, dataset, source        source = absolute path or "<archive>::<member>"
  path                            what the extractor reads (= source, or the unpacked
                                  copy under scratch/<ds>/ for archive members)
  kind                            video | frames
  split                           official split if known from the dataset's files, else ''
  frames                          *decoded* frame count (videos: full decode pass;
                                  frame folders: number of image files)
  fps, width, height
  size_bytes, sha256, sha256_mode
     full          sha256 of the whole file (files <= 8 MiB)
     fast4mb       sha256(size || first 4 MiB || last 4 MiB)  (files > 8 MiB; fast mode)
     frames_md5    frame folders: sha256 over the per-image md5 digests in sorted order
     unreadable    the source could not be opened/decoded (frames/fps/size 0, no hash);
                   the clip is listed in manifest/<stem>_exclude.csv with the error
  downloaded_by_us                1 only for files fetched by our own download step
                                  (registry flag downloaded_by_us, e.g. youtube_asl;
                                  always 0 for the tier-1 datasets)
  unpacked_to_scratch             1 if `path` is our extracted copy of an archive member
  dup_of                          clip_id (possibly of another dataset) with identical
                                  content (same sha256 + frames); its output is copied
                                  instead of re-extracting (MediaPipe is deterministic)

Archive members are unpacked to scratch/<ds>/<clip_id><ext> (zip CRC checked on read;
originals are never modified). Frame-folder datasets also write
manifest/<ds>_frame_md5.tsv.gz (per-image md5) used for the PHOENIX overlap report.

Unreadable sources (cannot be opened or decoded, missing archive member, ...) do not
fail the build: they get a manifest row with sha256_mode=unreadable and are written to
manifest/<stem>_exclude.csv (clip_id, reason, date; the mechanism used for the corrupt
PHOENIX-2014T clip), so extract_dataset skips them and verify_dataset lists them.
Hand-written exclusions already in that file are kept. Only if more than
MAX_UNREADABLE_FRAC of the sources are unreadable is the build a hard failure: no
manifest is written and the exit code is 1. manifest/<stem>_manifest_errors.csv lists
the errors in every case.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

from tools.extract.datasets import (DATASETS, Source, enumerate_sources, exclude_path,
                                    listdir_unique, make_clip_id, roots_of, scratch_root,
                                    split_of, wpath)

FIELDS = ["clip_id", "dataset", "source", "path", "kind", "split", "frames", "fps",
          "width", "height", "size_bytes", "sha256", "sha256_mode", "downloaded_by_us",
          "unpacked_to_scratch", "dup_of"]
FAST_BLOCK = 4 << 20
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")
MAX_UNREADABLE_FRAC = 0.02   # more unreadable sources than this: hard failure
AUTO_REASON = "unreadable at manifest build: "  # prefix of the exclusions written here


def file_sha256(path: str) -> tuple[str, str]:
    size = os.path.getsize(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if size <= 2 * FAST_BLOCK:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
            return h.hexdigest(), "full"
        h.update(str(size).encode())
        h.update(f.read(FAST_BLOCK))
        f.seek(size - FAST_BLOCK)
        h.update(f.read(FAST_BLOCK))
    return h.hexdigest(), "fast4mb"


def count_video(path: str, default_fps: float = 25.0) -> tuple[int, float, int, int]:
    """Decoded frame count (grab() decodes every packet), fps, width, height."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or default_fps
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = 0
        while cap.grab():
            n += 1
    finally:
        cap.release()
    return n, fps, w, h


def frame_files(path: str) -> list[str]:
    return [f for f in listdir_unique(path) if f.lower().endswith(IMG_EXT)]


def unpack_member(archive: str, member: str, dst: str) -> None:
    import zipfile
    with zipfile.ZipFile(archive) as z:
        info = z.getinfo(member)
        if os.path.exists(dst) and os.path.getsize(dst) == info.file_size:
            return
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".tmp"
        with z.open(member) as fi, open(tmp, "wb") as fo:  # ZipExtFile checks the CRC
            while True:
                b = fi.read(1 << 20)
                if not b:
                    break
                fo.write(b)
        os.replace(tmp, dst)


def _row(args) -> tuple[dict, list[str] | None, str]:
    ds, cid, src, split, scratch, frame_fps, *rest = args  # no registry lookups (spawned workers)
    by_us = int(rest[0]) if rest else 0
    row = dict(clip_id=cid, dataset=ds, source=src.source, kind=src.kind, split=split,
               downloaded_by_us=by_us, unpacked_to_scratch=0, dup_of="")
    try:
        md5s = None
        if src.kind == "frames":
            import cv2
            fdir = src.unpacked or src.source  # tar_frames: our unpacked copy
            if src.unpacked:
                row["unpacked_to_scratch"] = 1
            files = frame_files(fdir)
            md5s = []
            for f in files:
                with open(os.path.join(fdir, f), "rb") as fh:
                    md5s.append(hashlib.md5(fh.read()).hexdigest())
            img = cv2.imread(os.path.join(fdir, files[0])) if files else None
            h, w = (img.shape[:2] if img is not None else (0, 0))
            row.update(path=fdir, frames=len(files), fps=frame_fps,
                       width=w, height=h,
                       size_bytes=sum(os.path.getsize(os.path.join(fdir, f)) for f in files),
                       sha256=hashlib.sha256("".join(md5s).encode()).hexdigest(),
                       sha256_mode="frames_md5")
        else:
            path = src.source
            if src.unpacked:  # tar member, already unpacked by datasets.unpack_tar
                path = src.unpacked
                if not os.path.exists(path):
                    raise FileNotFoundError(path)
                row["unpacked_to_scratch"] = 1
            elif src.archive:
                ext = os.path.splitext(src.member)[1].lower()
                path = os.path.join(scratch, cid + ext)
                unpack_member(src.archive, src.member, path)
                row["unpacked_to_scratch"] = 1
            n, fps, w, h = count_video(path)
            sha, mode = file_sha256(path)
            row.update(path=path, frames=n, fps=round(fps, 4), width=w, height=h,
                       size_bytes=os.path.getsize(path), sha256=sha, sha256_mode=mode)
        return row, md5s, ""
    except Exception as e:  # noqa: BLE001 - reported per row
        return unreadable_row(row, src), None, f"{type(e).__name__}: {e}"


def unreadable_row(row: dict, src: Source) -> dict:
    """Manifest row of a source that could not be read (excluded from extraction)."""
    path = row.get("path") or src.unpacked or (src.source if not src.archive else "")
    try:
        size = os.path.getsize(path) if path else 0
    except OSError:
        size = 0
    return dict(row, path=path, frames=0, fps=0.0, width=0, height=0, size_bytes=size,
                sha256="", sha256_mode="unreadable", dup_of="")


def write_excludes(manifest: str, errors: list[tuple[str, str]]) -> int:
    """Rewrite manifest/<stem>_exclude.csv: hand-written rows are kept, rows written by an
    earlier build (reason starting with AUTO_REASON) are replaced by ``errors``. Returns
    the number of rows written by this build. The file is removed if it ends up empty."""
    p = exclude_path(manifest)
    keep: list[dict] = []
    if os.path.exists(p):
        with open(p, encoding="utf-8", newline="") as f:
            keep = [r for r in csv.DictReader(f)
                    if (r.get("clip_id") or "").strip()
                    and not (r.get("reason") or "").startswith(AUTO_REASON)]
    have = {r["clip_id"].strip() for r in keep}
    today = time.strftime("%Y-%m-%d")
    new = [dict(clip_id=c, reason=AUTO_REASON + " ".join(e.split()), date=today)
           for c, e in errors if c not in have]
    rows = keep + new
    if not rows:
        if os.path.exists(p):
            os.remove(p)
        return 0
    fields = list(dict.fromkeys(["clip_id", "reason", "date"]
                                + [k for r in keep for k in r if k is not None]))
    with open(p + ".tmp", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) or "" for k in fields})
    os.replace(p + ".tmp", p)
    return len(new)


def read_manifest(path: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["fps"] = float(r["fps"])
        for k in ("frames", "width", "height", "size_bytes", "downloaded_by_us",
                  "unpacked_to_scratch"):
            r[k] = int(r[k])
    return rows


def write_manifest(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    os.replace(tmp, path)


def assign_dups(rows: list[dict], prior: list[dict]) -> int:
    """Set dup_of for rows whose (sha256, sha256_mode, frames) equals an earlier row of
    this dataset (in clip_id order) or a row of a previously built manifest."""
    seen: dict[tuple, str] = {}
    for r in prior:
        if not r.get("dup_of"):
            seen.setdefault((r["sha256"], r["sha256_mode"], int(r["frames"])), r["clip_id"])
    n = 0
    for r in rows:
        if int(r["frames"]) == 0:
            continue
        k = (r["sha256"], r["sha256_mode"], int(r["frames"]))
        if k in seen:
            r["dup_of"] = seen[k]
            n += 1
        else:
            seen[k] = r["clip_id"]
    return n


_scratch_root = scratch_root


def build(ds: str, workers: int, roots=None, limit: int | None = None,
          manifest_dir: str | None = None, scratch_root: str | None = None,
          prior_datasets: list[str] | None = None, log=print, stem: str | None = None) -> dict:
    """Write <manifest_dir>/<stem>.csv (stem defaults to the dataset name; a batch of a
    queue dataset uses '<ds>_<batch>')."""
    manifest_dir = manifest_dir or wpath("manifest")
    stem = stem or ds
    scratch = os.path.join(scratch_root or _scratch_root(), ds)
    t0 = time.time()
    roots = roots if roots is not None else roots_of(ds)
    srcs: list[Source] = enumerate_sources(ds, roots)
    multi = len(roots) > 1  # prefix the root label so ids stay unique across roots
    jobs, ids, nrep = [], {}, 0
    for s in srcs:
        cid = make_clip_id(ds, s, multi)
        if cid in ids:
            if ids[cid] == s.source:  # same file listed twice (NFS readdir repeats)
                nrep += 1
                continue
            raise RuntimeError(f"clip_id collision: {cid} ({ids[cid]} vs {s.source})")
        ids[cid] = s.source
        jobs.append((ds, cid, s, split_of(ds, s), scratch, DATASETS[ds].get("fps", 25.0),
                     int(DATASETS[ds].get("downloaded_by_us", 0)) and not s.archive))
    jobs.sort(key=lambda j: j[1])
    if limit:
        jobs = jobs[:limit]
    log(f"[{stem}] {len(jobs)} sources enumerated in {time.time() - t0:.0f}s"
        + (f" ({nrep} repeated listings dropped)" if nrep else ""))
    rows, md5s, errors = [], {}, []
    if workers <= 1:
        it = map(_row, jobs)
        pool = None
    else:
        pool = mp.get_context("spawn").Pool(workers)
        it = pool.imap_unordered(_row, jobs, chunksize=4)
    try:
        for i, (row, m, err) in enumerate(it, 1):
            rows.append(row)
            if err:
                errors.append((row["clip_id"], err))
            else:
                if m is not None:
                    md5s[row["clip_id"]] = m
            if i % 2000 == 0:
                log(f"[{ds}] manifest {i}/{len(jobs)} ({time.time() - t0:.0f}s)")
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    rows.sort(key=lambda r: r["clip_id"])
    errors.sort()
    os.makedirs(manifest_dir, exist_ok=True)
    manifest =os.path.join(manifest_dir, f"{stem}.csv")
    err_csv = os.path.join(manifest_dir, f"{stem}_manifest_errors.csv")
    if errors:
        with open(err_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows([("clip_id", "error")] + errors)
    elif os.path.exists(err_csv):
        os.remove(err_csv)
    frac = len(errors) / max(len(jobs), 1)
    if frac > MAX_UNREADABLE_FRAC:
        summary = dict(dataset=ds, unit=stem, status="fail", sources=len(jobs),
                       errors=len(errors), unreadable_frac=round(frac, 4),
                       max_unreadable_frac=MAX_UNREADABLE_FRAC,
                       elapsed_s=round(time.time() - t0, 1))
        with open(os.path.join(manifest_dir, f"{stem}_summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
        log(f"[{stem}] manifest NOT written: {len(errors)}/{len(jobs)} sources unreadable "
            f"({frac:.1%} > {MAX_UNREADABLE_FRAC:.0%}); see {os.path.basename(err_csv)}")
        return summary
    prior = []
    for p in prior_datasets or []:
        pp = os.path.join(manifest_dir, f"{p}.csv")
        if os.path.exists(pp):
            prior += read_manifest(pp)
    ndup = assign_dups(rows, prior)
    nexcl = write_excludes(manifest, errors)
    write_manifest(rows, manifest)
    if md5s:
        with gzip.open(os.path.join(manifest_dir, f"{stem}_frame_md5.tsv.gz"), "wt") as f:
            for cid in sorted(md5s):
                f.write(cid + "\t" + ",".join(x[:16] for x in md5s[cid]) + "\n")
    summary = dict(dataset=ds, unit=stem, status="ok", clips=len(rows),
                   frames=sum(r["frames"] for r in rows), dup_rows=ndup, errors=len(errors),
                   excluded_unreadable=nexcl, repeated_listings=nrep,
                   elapsed_s=round(time.time() - t0, 1),
                   splits={s: sum(1 for r in rows if r["split"] == s)
                           for s in sorted({r["split"] for r in rows})},
                   sha256_modes={m: sum(1 for r in rows if r["sha256_mode"] == m)
                                 for m in sorted({r["sha256_mode"] for r in rows})},
                   zero_frame_clips=[r["clip_id"] for r in rows if r["frames"] == 0][:50])
    with open(os.path.join(manifest_dir, f"{stem}_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log(f"[{stem}] manifest: {json.dumps({k: v for k, v in summary.items() if k != 'zero_frame_clips'})}")
    return summary


def overlap_report(ds: str, other: str, manifest_dir: str | None = None) -> dict:
    """Share of `ds` frames (by image md5) that also occur in `other` (frame folders)."""
    manifest_dir = manifest_dir or wpath("manifest")

    def load(d):
        with gzip.open(os.path.join(manifest_dir, f"{d}_frame_md5.tsv.gz"), "rt") as f:
            return {l.split("\t")[0]: l.rstrip("\n").split("\t")[1].split(",") for l in f}
    a, b = load(ds), load(other)
    bset = {h for v in b.values() for h in v}
    n = sum(len(v) for v in a.values())
    shared = sum(1 for v in a.values() for h in v if h in bset)
    clips_any = sum(1 for v in a.values() if any(h in bset for h in v))
    rows = read_manifest(os.path.join(manifest_dir, f"{ds}.csv"))
    rep = dict(dataset=ds, other=other, frames=n, frames_shared=shared,
               frames_shared_frac=round(shared / max(n, 1), 4), clips=len(a),
               clips_with_any_shared_frame=clips_any,
               clips_identical=sum(1 for r in rows if r["dup_of"].startswith(other + "__")))
    with open(os.path.join(manifest_dir, f"{ds}_overlap_{other}.json"), "w") as f:
        json.dump(rep, f, indent=1)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(DATASETS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="first N clips by clip_id (smoke tests, "
                    "queue batches)")
    ap.add_argument("--batch", help="queue datasets: write manifest/<ds>_<batch>.csv")
    ap.add_argument("--prior", nargs="*", default=None,
                    help="datasets whose manifests are checked for identical content "
                         "(default: phoenix14t for phoenix14)")
    a = ap.parse_args(argv)
    prior = a.prior if a.prior is not None else (["phoenix14t"] if a.dataset == "phoenix14" else [])
    from tools.extract.datasets import unit_stem
    s = build(a.dataset, a.workers, limit=a.limit, prior_datasets=prior,
              log=lambda m: print(time.strftime("%F %T"), m, flush=True),
              stem=unit_stem(a.dataset, a.batch))
    if a.dataset == "phoenix14":
        try:
            print(json.dumps(overlap_report("phoenix14", "phoenix14t")), flush=True)
        except FileNotFoundError as e:
            print("overlap report skipped:", e, flush=True)
    sys.exit(1 if s["status"] != "ok" else 0)


if __name__ == "__main__":
    main()
