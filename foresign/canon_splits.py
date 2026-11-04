"""Evaluation splits and the inductive pretraining filter for C77-3D canon data (Phase 2).

canon_split(ds, clip_id, split_attr) -> "train" | "val" | "test" | "excluded".
- Evaluation datasets with official splits (EVAL_ATTR) use the clip's `split` attribute, which
  the canon builder copied from the extraction manifest; a missing or unknown value is
  "excluded" (fail-closed: such a clip never enters pretraining).
- LSA64 has no split attribute: signers 1-8 train, 9-10 test (id = sign_signer_repetition), the
  same rule as Phase 1a.
- Other known corpora are pretraining-only: "train".
- Unknown datasets raise, so a new corpus can never silently enter pretraining.

The pretraining pool keeps only "train" clips that are not in the exclusion list
(data_ann/canon_exclude.json, built by tools/build_canon_exclude.py: PHOENIX cross-release frame
overlap, manifest exact duplicates of held-out clips, Phase 1a WLASL near-duplicates).
"""
from __future__ import annotations

import json
import os

EVAL_ATTR = {"autsl", "wlasl", "phoenix14t", "phoenix14", "how2sign"}
EVAL_DATASETS = EVAL_ATTR | {"lsa64"}
CANON_POOL = {
    "asl_citizen", "autsl", "chicagofswild", "ghanaian_sl", "how2sign", "include", "isign",
    "karsl", "lsa64", "lsa_t", "lse", "openasl", "phoenix14", "phoenix14t", "sign2mint",
    "slovo", "wlasl", "youtube_asl",
}
# phoenix14 is (almost) the same recordings as phoenix14t; as in Phase 1a only the 2014T
# release enters pretraining (its train clips minus every clip overlapping a held-out clip).
DEFAULT_PRETRAIN = sorted(CANON_POOL - {"phoenix14"})
_ATTR = {"train": "train", "val": "val", "dev": "val", "test": "test"}


def _name(ds: str, clip_id: str) -> str | None:
    pre = f"{ds}__"
    return clip_id[len(pre):] if clip_id.startswith(pre) else None


def canon_split(ds: str, clip_id: str, split_attr: str) -> str:
    if ds not in CANON_POOL:
        raise ValueError(f"unknown canon dataset {ds!r}: add a split rule to foresign/canon_splits.py")
    name = _name(ds, clip_id)
    if name is None:
        return "excluded"
    if ds in EVAL_ATTR:
        return _ATTR.get((split_attr or "").strip().lower(), "excluded")
    if ds == "lsa64":
        bits = name.split("_")
        if len(bits) != 3 or not all(b.isdigit() for b in bits):
            return "excluded"
        return "train" if int(bits[1]) <= 8 else "test"
    return "train"


def load_exclude(path: str) -> dict:
    """{dataset: set(clip ids)}; missing file fails closed unless FS_ALLOW_NO_DUP=1."""
    if not os.path.exists(path):
        if os.environ.get("FS_ALLOW_NO_DUP") == "1":
            return {}
        raise FileNotFoundError(
            f"{path} missing: run `python -m tools.build_canon_exclude` (or set FS_ALLOW_NO_DUP=1 "
            "to pretrain without the overlap/duplicate exclusion)")
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {ds: set(ids) for ds, ids in d.get("exclude", {}).items()}


def filter_canon_clips(clips: dict, exclude: dict) -> dict:
    """clips: {ds: [(file_idx, clip_id, T, split_attr), ...]} -> pretraining pool."""
    out = {}
    for ds, rows in clips.items():
        drop = exclude.get(ds, set())
        kept = [r for r in rows if r[1] not in drop and canon_split(ds, r[1], r[3]) == "train"]
        if kept:
            out[ds] = kept
    return out
