"""Build data_ann/canon_exclude.json: clips that must stay out of Phase 2 pretraining although
their own split is "train".

  python -m tools.build_canon_exclude --manifest-dir <dir> [--old-dup results/dup_exclude.json]

Sources (all fail-closed towards exclusion):
  phoenix_overlap  PHOENIX-2014 and -2014T share most frames (byte-identical images). Every clip
                   of either release sharing >= 1 frame md5 (manifest *_frame_md5.tsv.gz) with a
                   val/test clip of either release is excluded.
  manifest_dup     exact duplicates (manifest `dup_of`, sha256 of the source video): every
                   connected group containing a held-out clip has all its other clips excluded.
  old_near_dup     Phase 1a near-duplicate audit (results/dup_exclude.json, old keys
                   wlasl2000/all/<video_id>) mapped to canon WLASL ids by video id.
The manifest directory is the extraction work root's manifest/ (tools.extract.build_manifest).
--old-dup is optional (only used if the file exists).
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import os
import sys
from collections import defaultdict

from foresign.canon_splits import CANON_POOL, EVAL_DATASETS, canon_split

PHOENIX = ("phoenix14t", "phoenix14")


def _read_manifest(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _manifests(mdir):
    out = {}
    for p in sorted(glob.glob(os.path.join(mdir, "*.csv"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem.endswith("_exclude"):
            continue
        ds = stem.split("_b0")[0] if stem.startswith("youtube_asl_b") else stem
        if ds not in CANON_POOL:
            continue
        out.setdefault(ds, []).extend(_read_manifest(p))
    return out


def _split(ds, row):
    return canon_split(ds, row["clip_id"], row.get("split", ""))


def phoenix_overlap(mdir, manifests):
    md5 = {}
    for ds in PHOENIX:
        p = os.path.join(mdir, f"{ds}_frame_md5.tsv.gz")
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} missing (needed for the PHOENIX cross-release exclusion)")
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                cid, _, rest = line.rstrip("\n").partition("\t")
                md5[cid] = set(rest.split(",")) if rest else set()
    split = {r["clip_id"]: _split(ds, r) for ds in PHOENIX for r in manifests.get(ds, [])}
    held = set()
    for cid, s in split.items():
        if s != "train":
            held |= md5.get(cid, set())
    out = defaultdict(list)
    for cid, frames in md5.items():
        ds = cid.split("__", 1)[0]
        if split.get(cid, "excluded") == "train" and frames & held:
            out[ds].append(cid)
    no_md5 = sorted(c for c in split if c not in md5)
    for cid in no_md5:                      # overlap unknown -> fail closed
        if split[cid] == "train":
            out[cid.split("__", 1)[0]].append(cid)
    return {ds: sorted(v) for ds, v in out.items()}, {"clips_without_md5": no_md5,
                                                      "heldout_frames": len(held)}


def manifest_dups(manifests):
    out = defaultdict(set)
    for ds, rows in manifests.items():
        if ds not in EVAL_DATASETS:
            continue
        parent = {}

        def find(a):
            while parent.get(a, a) != a:
                a = parent[a]
            return a
        for r in rows:
            if r.get("dup_of"):
                ra, rb = find(r["clip_id"]), find(r["dup_of"])
                if ra != rb:
                    parent[ra] = rb
        groups = defaultdict(list)
        for r in rows:
            groups[find(r["clip_id"])].append(r)
        for members in groups.values():
            if len(members) < 2:
                continue
            if any(_split(ds, r) != "train" for r in members):
                out[ds] |= {r["clip_id"] for r in members if _split(ds, r) == "train"}
    return {ds: sorted(v) for ds, v in out.items() if v}


def old_near_dups(old_path, manifests):
    if not old_path or not os.path.exists(old_path):
        return {}, {"unmapped": [], "note": f"{old_path} not found"}
    with open(old_path, encoding="utf-8") as f:
        old = json.load(f)
    by_vid = defaultdict(list)
    for r in manifests.get("wlasl", []):
        name = r["clip_id"].split("__")[-1]
        by_vid[name.split("-wlasl-")[0]].append(r)
    out, unmapped = set(), []
    for key in old.get("wlasl", []):
        rows = by_vid.get(key.split("/")[-1])
        if not rows:
            unmapped.append(key)
            continue
        out |= {r["clip_id"] for r in rows if _split("wlasl", r) == "train"}
    others = {ds: len(v) for ds, v in old.items() if ds != "wlasl" and v}
    if others:
        raise ValueError(f"old near-dup list has non-WLASL entries {others}: add a mapping")
    return ({"wlasl": sorted(out)} if out else {}), {"unmapped": unmapped}


def build(mdir, old_dup=None):
    manifests = _manifests(mdir)
    ph, ph_info = phoenix_overlap(mdir, manifests)
    md = manifest_dups(manifests)
    nd, nd_info = old_near_dups(old_dup, manifests)
    exclude = defaultdict(set)
    for src in (ph, md, nd):
        for ds, ids in src.items():
            exclude[ds] |= set(ids)
    missing = sorted(EVAL_DATASETS - set(manifests))
    return {
        "exclude": {ds: sorted(v) for ds, v in sorted(exclude.items())},
        "counts": {ds: len(v) for ds, v in sorted(exclude.items())},
        "sources": {
            "phoenix_overlap": {**{ds: len(v) for ds, v in ph.items()}, **ph_info},
            "manifest_dup": {ds: len(v) for ds, v in md.items()},
            "old_near_dup": {**{ds: len(v) for ds, v in nd.items()}, **nd_info},
        },
        "eval_datasets_without_manifest": missing,
    }


def main(argv=None):
    from foresign import paths
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--old-dup", default=paths.DUP_EXCLUDE)
    ap.add_argument("--out", default=paths.CANON_EXCLUDE)
    a = ap.parse_args(argv)
    rep = build(a.manifest_dir, a.old_dup)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "exclude"}, indent=1)[:4000])
    print(f"[canon_exclude] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
