"""Dataset registry for the pose extraction (tier 1, tier 2, YouTube-ASL queue, others).

Each entry says where the sources live, how to enumerate them, and how the official split is
read. All sources are found under one root, ``REEXTRACT_SRC_ROOT`` (default ``./videos``), with
one folder per dataset laid out as listed in docs/extraction.md (the layout of each dataset's
official release; obtain every dataset from its authors under its licence). Enumeration yields
``Source`` records; ``clip_id`` is derived only from the dataset name and the source's path
*relative to its declared root* (or the archive member name), so it is stable across runs and
machines and does not depend on where the root is.

Working directory layout (``REEXTRACT_ROOT``, default ``~/reextract``):
  manifest/<ds>.csv            one row per clip (build_manifest)
  out/<ds>/shard_NNNNN.h5      packed outputs (extract_dataset)
  out/<ds>/_staging/*.h5       per-clip outputs not yet packed
  out/<ds>/failures.csv        extraction failures
  reports/<ds>_verify.json     verify_dataset
  scratch/<ds>/                unpacked archive members (removed after verification)
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache

SRC = os.environ.get("REEXTRACT_SRC_ROOT", "videos")
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")

_PH14T = f"{SRC}/phoenix14t/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px"
_PH14 = f"{SRC}/phoenix14/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px"
_LSE = f"{SRC}/lse/extracted"

# kind: files (flat dir), files_rec (recursive), zip (archive members), framedirs
# roots: list of (path, label). label is the split when split == "root".
DATASETS: dict[str, dict] = {
    "autsl": dict(kind="files", split="root", suffix="_color.mp4",
                  roots=[(f"{SRC}/autsl/{s}", s) for s in ("train", "val", "test")]),
    "asl_citizen": dict(kind="files", split="asl_citizen",
                        roots=[(f"{SRC}/asl_citizen/ASL_Citizen/videos", "")]),
    "wlasl": dict(kind="zip", split="wlasl", strip_prefix="video_format/",
                  roots=[(f"{SRC}/wlasl/wlasl_video_format.zip", "")]),
    "include": dict(kind="zip", split=None, roots="include_zips"),
    "lse": dict(kind="files_rec", split="lse", roots=[(_LSE, "")]),
    "phoenix14t": dict(kind="framedirs", split="root", fps=25.0,
                       roots=[(f"{_PH14T}/{d}", s) for d, s in
                              (("train", "train"), ("dev", "val"), ("test", "test"))]),
    # multisigner only; the SI5 split re-uses the same recordings (same folder names).
    "phoenix14": dict(kind="framedirs", split="root", fps=25.0, subdir="1",
                      roots=[(f"{_PH14}/{d}", s) for d, s in
                             (("train", "train"), ("dev", "val"), ("test", "test"))]),
}
TIER1_ORDER = ["autsl", "asl_citizen", "wlasl", "include", "lse", "phoenix14t", "phoenix14"]

# YouTube-ASL: full-length YouTube videos (obtained by the user, e.g. with yt-dlp, from the
# official YouTube-ASL id list) in <dest>/videos/<yid>.<ext>. Only videos listed in
# <dest>/completed.tsv (yid, file, bytes, utc; one line appended per finished download) are
# sources, so half-written files are never picked up. Extraction runs in batches as the queue
# grows; set REEXTRACT_QUEUE_SKIP to the clip_ids (one per line, file path) already handled to
# build a manifest of only the new ones. clip_id = youtube_asl__<yid>. No split (pretraining
# only; drop the ids that overlap the OpenASL test set when OpenASL test is used for
# evaluation). downloaded_by_us=1: cleanup --delete-videos may remove the videos under
# download_root after verification.
YTASL_DEST = os.environ.get("YTASL_DEST", f"{SRC}/youtube_asl")
DATASETS["youtube_asl"] = dict(kind="queue", split=None, downloaded_by_us=1,
                               queue=f"{YTASL_DEST}/completed.tsv",
                               download_root=f"{YTASL_DEST}/videos",
                               roots=[(f"{YTASL_DEST}/videos", "")])

# Tier 2: public datasets downloaded from their official pages into <SRC>/<ds>/.
# downloaded_by_us=1: after verification, cleanup --delete-videos may remove the
# unpacked/downloaded videos under download_root (never the original archives).
# lsa_t: the cuts are unpacked from the release's clips.7z (origin_archive must still exist
# before the unpacked copy is deleted).
NEW = SRC
DATASETS.update({
    # Leiden HANDS! GSL lexicon: one sign per video, <SIGN>/<SIGN>.mp4; no split.
    "ghanaian_sl": dict(kind="files_rec", split=None, downloaded_by_us=1,
                        download_root=f"{NEW}/ghanaian_sl/GSL_openpose_data",
                        roots=[(f"{NEW}/ghanaian_sl/GSL_openpose_data", "")]),
    # LSA64 raw (uncut): <sign>_<signer>_<rep>.mp4; no official split (protocols are
    # signer-based; derive them downstream from the file name).
    "lsa64": dict(kind="files", split=None, downloaded_by_us=1,
                  download_root=f"{NEW}/lsa64/raw",
                  roots=[(f"{NEW}/lsa64/raw/all", "")]),
    # Sign2MINT DGS STEM lexicon (metadata in entries.json); no split.
    "sign2mint": dict(kind="files", split=None, downloaded_by_us=1,
                      download_root=f"{NEW}/sign2mint/videos",
                      roots=[(f"{NEW}/sign2mint/videos", "")]),
    # LSA-T release cuts (8459 clips, meta.csv); the release has no official split.
    "lsa_t": dict(kind="files", split=None, downloaded_by_us=1,
                  download_root=f"{NEW}/lsa_t/cuts",
                  origin_archive=f"{NEW}/lsa_t/clips.7z",
                  roots=[(f"{NEW}/lsa_t/cuts", "")]),
    # Slovo (trimmed): official train/test folders (= annotations.csv 'train' column).
    "slovo": dict(kind="files", split="root", downloaded_by_us=1,
                  download_root=f"{NEW}/slovo",
                  roots=[(f"{NEW}/slovo/{s}", s) for s in ("train", "test")]),
    # KArSL-502 RGB: <signer>/<train|test>/<range>/<sign>/<file>_c.mp4; official
    # per-signer train/test folders.
    "karsl": dict(kind="files_rec", split="karsl", downloaded_by_us=1,
                  download_root=f"{NEW}/karsl",
                  roots=[(f"{NEW}/karsl/{s}", s) for s in ("01", "02", "03")]),
    # ChicagoFSWild / FSWild+: JPEG sequences inside the *-Frames.tgz of each release
    # (unpacked to scratch, removed after verification; the archives are kept). The
    # authors cut frames at the source videos' native rate (no -r in their ffmpeg
    # call), so 30 fps is nominal. Split = 'partition' column of the csv (dev -> val).
    "chicagofswild": dict(kind="tar_frames", split="fswild", fps=30.0,
                          strip_prefix="ChicagoFSWildPlus/",
                          roots=[(f"{NEW}/chicagofswild/ChicagoFSWild/ChicagoFSWild/"
                                  "ChicagoFSWild-Frames.tgz", "fswild"),
                                 (f"{NEW}/chicagofswild/ChicagoFSWildPlus/"
                                  "ChicagoFSWildPlus-Frames.tgz", "fswildplus")]),
})
TIER2_ORDER = ["ghanaian_sl", "lsa64", "sign2mint", "lsa_t", "slovo", "karsl", "chicagofswild"]

# Other corpora of the pretraining pool. Clip ids depend only on paths relative to each root.
LAP = SRC
_ISIGN = f"{SRC}/isign"
DATASETS.update({
    # Full-frame How2Sign. train: the pre-cut clips; val / test: the full-length raw videos,
    # extracted whole (the manifest builder cannot cut by the official sentence timestamps;
    # slice the outputs with them downstream). The archives are ZIP files.
    "how2sign": dict(kind="zip", split="root", strip_prefix="raw_videos/",
                     roots=[(f"{SRC}/how2sign/h2s_{a}.tar.gz", s)
                            for a, s in (("train_clips", "train"), ("val", "val"),
                                         ("test", "test"))]),
    # OpenASL: full-length YouTube source videos (video-only streams, one per id:
    # <yid>.f605/.f134 H.264, .f396 = AV1 transcoded to H.264). Extracted whole; clips are cut
    # downstream with the official timestamps. The split is per clip, so it is left empty here.
    # clip_id = openasl__<yid> (format code dropped). .f251.webm files are audio-only streams.
    "openasl": dict(kind="files", split=None, id_sub=(r"\.f\d+$", ""),
                    exclude=r"\.f251\.webm$",
                    roots=[(f"{SRC}/openasl/videos", "")]),
    "isign": dict(kind="files", split="isign",
                  roots=[(f"{_ISIGN}/isign_videos/iSign-videos_v1.1", "v1.1"),
                         (f"{_ISIGN}/isign_test_videos/test_mp4", "test_mp4")]),
})
OTHER_ORDER = ["how2sign", "openasl", "isign"]

# Bake-off sample name used as the detection-rate reference for each dataset.
BAKEOFF_REF = {"autsl": "autsl", "asl_citizen": "asl_citizen", "wlasl": "wlasl",
               "include": "include", "lse": "lse", "phoenix14t": "phoenix14t",
               "phoenix14": "phoenix14t", "how2sign": "how2sign", "isign": "isign"}


def unit_stem(ds: str, batch: str | None = None) -> str:
    """Name of one processing unit: the dataset, or '<ds>_<batch>' for a batch of a
    queue dataset (YouTube-ASL is extracted in batches of finished downloads)."""
    return f"{ds}_{batch}" if batch else ds


def unit_paths(ds: str, batch: str | None = None) -> dict:
    """manifest / out dir / verify report of a unit:
    manifest/<stem>.csv, out/<ds>[/<batch>]/, reports/<stem>_verify.json."""
    stem = unit_stem(ds, batch)
    return dict(stem=stem, manifest=wpath("manifest", f"{stem}.csv"),
                out_dir=wpath("out", ds, batch) if batch else wpath("out", ds),
                report=wpath("reports", f"{stem}_verify.json"))


def exclude_path(manifest: str) -> str:
    """manifest/<stem>.csv -> manifest/<stem>_exclude.csv"""
    return os.path.splitext(manifest)[0] + "_exclude.csv"


def load_excludes(manifest: str) -> dict[str, str]:
    """clip_id -> reason from manifest/<stem>_exclude.csv (columns clip_id, reason, ...).

    Excluded clips stay in the manifest (it records what exists) but are not extracted,
    verification does not expect an output for them (the report lists them with their
    reason) and cleanup never deletes their sources. Used for sources that cannot be
    decoded, e.g. a corrupt PNG inside a PHOENIX frame folder."""
    p = exclude_path(manifest)
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8", newline="") as f:
        return {r["clip_id"].strip(): (r.get("reason") or "").strip()
                for r in csv.DictReader(f) if (r.get("clip_id") or "").strip()}


def work_root() -> str:
    return os.path.expanduser(os.environ.get("REEXTRACT_ROOT", "~/reextract"))


def wpath(*parts: str) -> str:
    return os.path.join(work_root(), *parts)


def scratch_root() -> str:
    """Where archive members are unpacked (REEXTRACT_SCRATCH, default <root>/scratch)."""
    return os.path.expanduser(os.environ.get("REEXTRACT_SCRATCH") or wpath("scratch"))


def tar_unpack_dir(ds: str, label: str) -> str:
    return os.path.join(scratch_root(), ds, "_tar", sanitize(label or "root"))


@dataclass
class Source:
    rel: str            # path relative to its root, '/'-separated (archive: member name)
    source: str         # absolute path, or "<archive>::<member>"
    label: str          # root label (split for split == "root")
    kind: str           # "video" | "frames"
    archive: str = ""   # archive path if the source is an archive member
    member: str = ""
    unpacked: str = ""  # tar members: our copy already unpacked by unpack_tar()


def _is_video(name: str) -> bool:
    return name.lower().endswith(VIDEO_EXT)


@lru_cache(maxsize=4)
def _load_file_list(path: str) -> tuple[str, ...]:
    """Relative paths from an uploader list ("<size>\t<relpath>" per line)."""
    with open(path, encoding="utf-8") as f:
        return tuple(sorted({l.rstrip("\n").split("\t")[-1] for l in f if l.strip()}))


def listed_files(root: str, recursive: bool, base: str | None = None) -> list[str] | None:
    """Files under ``root`` (relative to it) taken from the upload file list instead of
    a directory listing, or None if there is no list (or it has nothing under root).

    REEXTRACT_FILE_LIST names a list of "<size>\t<path relative to REEXTRACT_SRC_ROOT>"
    lines (e.g. written when the videos were copied). Paths are relative to ``base`` (default
    SRC). Useful when readdir on a network mount of large directories is slow or repeats
    entries; a listed file that is missing just becomes an unreadable source."""
    lst = os.environ.get("REEXTRACT_FILE_LIST")
    if not lst or not os.path.exists(lst):
        return None
    base = base if base is not None else LAP
    r = os.path.relpath(root, base).replace(os.sep, "/")
    if r == ".." or r.startswith("../"):
        return None
    pre = "" if r == "." else r + "/"
    out = [x[len(pre):] for x in _load_file_list(lst) if x.startswith(pre)]
    if not recursive:
        out = [x for x in out if "/" not in x]
    return out or None


def listdir_unique(path: str) -> list[str]:
    """os.listdir without repeated names, sorted. Some NFSv3 mounts return the same entry
    many times for large directories (observed: ~10x repeats), which would make enumeration
    yield duplicate sources (clip_id collision) and multiply the work."""
    return sorted(set(os.listdir(path)))


def walk_unique(root: str):
    """os.walk(root) (top-down, symlinked dirs not followed) with each directory's
    entries de-duplicated (see listdir_unique), so a repeated subdirectory entry is not
    walked twice. Uses scandir's d_type, no stat per file."""
    dirs: dict[str, bool] = {}
    files: set[str] = set()
    try:
        with os.scandir(root) as it:
            for e in it:
                if e.is_dir(follow_symlinks=True):
                    dirs[e.name] = e.is_symlink()
                else:
                    files.add(e.name)
    except OSError:
        return
    yield root, sorted(dirs), sorted(files)
    for d in sorted(dirs):
        if not dirs[d]:
            yield from walk_unique(os.path.join(root, d))


def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


def make_clip_id(ds: str, src: Source, multi_root: bool) -> str:
    rel = src.rel
    if src.kind == "video":
        rel = os.path.splitext(rel)[0]
        sub = DATASETS.get(ds, {}).get("id_sub")
        if sub:
            rel = re.sub(sub[0], sub[1], rel)
    parts = [p for p in rel.split("/") if p]
    if multi_root and src.label:
        parts = [src.label] + parts
    return f"{ds}__" + "__".join(sanitize(p) for p in parts)


def roots_of(ds: str) -> list[tuple[str, str]]:
    r = DATASETS[ds]["roots"]
    if r == "include_zips":
        return [(p, os.path.splitext(os.path.basename(p))[0])
                for p in sorted(glob.glob(f"{SRC}/include/*.zip"))]
    return list(r)


def _tar_rel(member: str, label: str) -> str:
    """'./train/x.webm' -> 'x.webm' (drop './' and a leading dir equal to the label)."""
    parts = [p for p in member.split("/") if p not in ("", ".")]
    if len(parts) > 1 and parts[0] == label:
        parts = parts[1:]
    return "/".join(parts)


def unpack_tar_frames(archive: str, udir: str, strip: str = "", log=print) -> list[tuple[str, int]]:
    """Unpack the image members of a (gzip) tar of frame folders to udir in one pass and
    return [(dir_rel, n_images)] (dir_rel = the images' folder, ``strip`` removed).
    Cached like unpack_tar: udir/_frames.tsv (dir_rel, n, bytes) is written last; a later
    call re-reads the archive only if a folder is missing or incomplete. The archive is
    opened read-only."""
    import tarfile
    idx = os.path.join(udir, "_frames.tsv")

    def complete(items):
        for rel, n, size in items:
            d = os.path.join(udir, *rel.split("/"))
            if not os.path.isdir(d):
                return False
            fs = [f for f in listdir_unique(d) if f.lower().endswith(IMG_EXT)]
            if len(fs) != n or sum(os.path.getsize(os.path.join(d, f)) for f in fs) != size:
                return False
        return True
    if os.path.exists(idx):
        with open(idx, encoding="utf-8") as f:
            items = [(r, int(n), int(z)) for r, n, z in
                     (l.rstrip("\n").split("\t") for l in f if l.strip())]
        if complete(items):
            return [(r, n) for r, n, _ in items]
    os.makedirs(udir, exist_ok=True)
    acc: dict[str, list[int]] = {}
    n_all = 0
    with tarfile.open(archive, "r|*") as t:
        for m in t:
            if not m.isfile() or not m.name.lower().endswith(IMG_EXT):
                continue
            name = m.name[2:] if m.name.startswith("./") else m.name
            if strip and name.startswith(strip):
                name = name[len(strip):]
            parts = [p for p in name.split("/") if p not in ("", ".")]
            if len(parts) < 2 or ".." in parts:
                continue
            rel = "/".join(parts[:-1])
            dst = os.path.join(udir, *parts)
            a = acc.setdefault(rel, [0, 0])
            a[0] += 1
            a[1] += m.size
            n_all += 1
            if os.path.exists(dst) and os.path.getsize(dst) == m.size:
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst + ".tmp", "wb") as fo:
                fo.write(t.extractfile(m).read())
            os.replace(dst + ".tmp", dst)
            if n_all % 200000 == 0:
                log(f"unpack {os.path.basename(archive)}: {n_all} images, {len(acc)} folders")
    with open(idx + ".tmp", "w", encoding="utf-8") as f:
        for rel in sorted(acc):
            f.write(f"{rel}\t{acc[rel][0]}\t{acc[rel][1]}\n")
    os.replace(idx + ".tmp", idx)
    return [(rel, acc[rel][0]) for rel in sorted(acc)]


def unpack_tar(archive: str, udir: str, label: str, log=print) -> list[tuple[str, str, int]]:
    """Unpack every video member of a (gzip) tar to udir in one sequential pass and return
    [(member, rel, size)]. A gzip tar cannot be read member by member, so this replaces
    the per-member unpack used for zips. The member list is cached in udir/_members.tsv
    (written last); a later call only re-reads the archive if a copy is missing or has
    the wrong size. The archive is opened read-only and never modified."""
    import tarfile
    idx = os.path.join(udir, "_members.tsv")
    if os.path.exists(idx):
        with open(idx, encoding="utf-8") as f:
            items = [(m, r, int(s)) for m, r, s in (l.rstrip("\n").split("\t") for l in f if l.strip())]
        if all(os.path.exists(os.path.join(udir, *r.split("/"))) and
               os.path.getsize(os.path.join(udir, *r.split("/"))) == s for _, r, s in items):
            return items
    os.makedirs(udir, exist_ok=True)
    items = []
    with tarfile.open(archive, "r|*") as t:
        for m in t:
            if not m.isfile() or not _is_video(m.name):
                continue
            rel = _tar_rel(m.name, label)
            dst = os.path.join(udir, *rel.split("/"))
            items.append((m.name, rel, m.size))
            if os.path.exists(dst) and os.path.getsize(dst) == m.size:
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            fi = t.extractfile(m)
            with open(dst + ".tmp", "wb") as fo:
                while True:
                    b = fi.read(1 << 20)
                    if not b:
                        break
                    fo.write(b)
            os.replace(dst + ".tmp", dst)
            if len(items) % 2000 == 0:
                log(f"unpack {os.path.basename(archive)}: {len(items)} members")
    with open(idx + ".tmp", "w", encoding="utf-8") as f:
        for m, r, s in items:
            f.write(f"{m}\t{r}\t{s}\n")
    os.replace(idx + ".tmp", idx)
    return items


def _queue_sources(ds: str, queue: str, root: str, label: str, multi: bool = False) -> list[Source]:
    """Sources listed in a downloader queue file (yid, file, bytes, ...), last entry per
    file wins. Entries whose file is gone (already cleaned up) or has a different size
    are left out; so are the clip_ids listed in REEXTRACT_QUEUE_SKIP (a file)."""
    skip: set[str] = set()
    sk = os.environ.get("REEXTRACT_QUEUE_SKIP")
    if sk and os.path.exists(sk):
        with open(sk, encoding="utf-8") as f:
            skip = {l.strip() for l in f if l.strip()}
    entries: dict[str, int] = {}
    if os.path.exists(queue):
        with open(queue, encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 3 and _is_video(p[1]) and "/" not in p[1]:
                    entries[p[1]] = int(p[2])
    out = []
    for name, size in entries.items():
        path = os.path.join(root, name)
        if not os.path.exists(path) or os.path.getsize(path) != size:
            continue
        src = Source(name, path, label, "video")
        if skip and make_clip_id(ds, src, multi) in skip:
            continue
        out.append(src)
    return out


def enumerate_sources(ds: str, roots: list[tuple[str, str]] | None = None) -> list[Source]:
    """All sources of a dataset, sorted by (label, rel). ``roots`` overrides the
    registry (tests)."""
    spec = DATASETS[ds]
    kind = spec["kind"]
    roots = roots if roots is not None else roots_of(ds)
    out: list[Source] = []
    for root, label in roots:
        if kind == "files":
            suf, excl = spec.get("suffix"), spec.get("exclude")
            listed = listed_files(root, recursive=False)
            for n in (listed if listed is not None else listdir_unique(root)):
                if excl and re.search(excl, n):
                    continue
                if _is_video(n) and (suf is None or n.endswith(suf)):
                    out.append(Source(n, os.path.join(root, n), label, "video"))
        elif kind == "queue":
            out += _queue_sources(ds, spec["queue"], root, label, multi=len(roots) > 1)
        elif kind == "files_rec":
            listed = listed_files(root, recursive=True)
            if listed is not None:
                out += [Source(rel, os.path.join(root, *rel.split("/")), label, "video")
                        for rel in listed if _is_video(rel.rsplit("/", 1)[-1])]
                continue
            for dp, _, fs in walk_unique(root):
                for f in fs:
                    if _is_video(f):
                        p = os.path.join(dp, f)
                        out.append(Source(os.path.relpath(p, root).replace(os.sep, "/"), p,
                                          label, "video"))
        elif kind == "zip":
            import zipfile
            strip = spec.get("strip_prefix", "")
            with zipfile.ZipFile(root) as z:
                for n in z.namelist():
                    if n.endswith("/") or not _is_video(n):
                        continue
                    rel = n[len(strip):] if strip and n.startswith(strip) else n
                    out.append(Source(rel, f"{root}::{n}", label, "video", root, n))
        elif kind == "tar":
            udir = tar_unpack_dir(ds, label)
            for member, rel, size in unpack_tar(root, udir, label):
                out.append(Source(rel, f"{root}::{member}", label, "video", root, member,
                                  unpacked=os.path.join(udir, *rel.split("/"))))
        elif kind == "tar_frames":
            udir = tar_unpack_dir(ds, label)
            for rel, _n in unpack_tar_frames(root, udir, spec.get("strip_prefix", "")):
                out.append(Source(rel, f"{root}::{rel}", label, "frames", root, rel,
                                  unpacked=os.path.join(udir, *rel.split("/"))))
        elif kind == "framedirs":
            sub = spec.get("subdir")
            for d in listdir_unique(root):
                p = os.path.join(root, d)
                if not os.path.isdir(p):
                    continue
                if sub:
                    p = os.path.join(p, sub)
                    if not os.path.isdir(p):
                        continue
                out.append(Source(d, p, label, "frames"))
        else:
            raise ValueError(kind)
    out.sort(key=lambda s: (s.label, s.rel))
    return out


# ---------------------------------------------------------------- official splits
@lru_cache(maxsize=None)
def _asl_citizen_splits() -> dict:
    d = {}
    base = f"{SRC}/asl_citizen/ASL_Citizen/splits"
    for s in ("train", "val", "test"):
        p = os.path.join(base, f"{s}.csv")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    d[r["Video file"]] = s
    return d


@lru_cache(maxsize=None)
def _wlasl_splits() -> dict:
    p = f"{SRC}/wlasl/WLASL_v0.3.json"
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return {i["video_id"]: i["split"] for e in data for i in e["instances"]}


_LSE_DIRS = {"MSSL_Train_Set": "train", "MSSL_Val_Set": "val", "MSSL_Test_set": "test",
             "OSLWL_Query_Set-and-OSLWL_Val_Set": "oslwl_val",
             "OSLWL_Query_Set-and-OSLWL_Test_Set": "oslwl_test"}
_DIR_SPLIT = {"train": "train", "dev": "val", "val": "val", "test": "test"}


def split_of(ds: str, src: Source) -> str:
    """Official split if known from the dataset's own files, else ''."""
    rule = DATASETS[ds].get("split")
    if rule == "root":
        return src.label
    if rule == "asl_citizen":
        return _asl_citizen_splits().get(os.path.basename(src.rel), "")
    if rule == "wlasl":
        first = src.rel.split("/")[0]
        vid = os.path.basename(src.rel).split("-")[0]
        return _wlasl_splits().get(vid, "") or _DIR_SPLIT.get(first, "")
    if rule == "lse":
        return _LSE_DIRS.get(src.rel.split("/")[0], "")
    uid = os.path.splitext(os.path.basename(src.rel))[0]
    if rule == "isign":  # optional <SRC>/isign/{val,test}.csv (column uid); else no split
        return _uid_splits(_ISIGN, (("val.csv", "val"), ("test.csv", "test"))).get(uid, "")
    if rule == "karsl":  # <train|test>/<range>/<sign>/<file>
        return _DIR_SPLIT.get(src.rel.split("/")[0], "")
    if rule == "fswild":
        p = _fswild_splits().get((src.label, src.rel), "")
        return _DIR_SPLIT.get(p, p)
    return ""


@lru_cache(maxsize=None)
def _fswild_splits() -> dict:
    """(root label, filename) -> partition from ChicagoFSWild.csv / ChicagoFSWildPlus.csv."""
    d = {}
    base = f"{NEW}/chicagofswild"
    for label, p in (("fswild", f"{base}/ChicagoFSWild/ChicagoFSWild/ChicagoFSWild.csv"),
                     ("fswildplus", f"{base}/ChicagoFSWildPlus/ChicagoFSWildPlus.csv")):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    d[(label, r["filename"].strip())] = r["partition"].strip()
    return d


@lru_cache(maxsize=None)
def _uid_splits(base: str, files: tuple) -> dict:
    d = {}
    for name, split in files:
        p = os.path.join(base, name)
        if os.path.exists(p):
            with open(p, encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    d.setdefault(r["uid"].strip(), split)
    return d
