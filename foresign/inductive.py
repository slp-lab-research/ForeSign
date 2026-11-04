"""Data-exposure provenance stored in every pretraining checkpoint (inductive setting: the
pretraining pool excludes every evaluation val/test clip and the clips listed in the canon
exclusion list; see foresign.canon_splits)."""
from __future__ import annotations

import os

from foresign import paths


def provenance(cfg: dict, dup_path: str | None = None) -> dict:
    """Data-exposure facts stored in every checkpoint. c77 (canon) runs record the canon
    exclusion list's hash and the canon root instead of the Phase 1a dup list."""
    import hashlib
    from foresign.inputs import C77, layout_for
    canon = layout_for(cfg).fmt == C77
    if dup_path is None:
        dup_path = paths.CANON_EXCLUDE if canon else paths.DUP_EXCLUDE
    sha = None
    if os.path.exists(dup_path):
        with open(dup_path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
    out = {"inductive": os.environ.get("FS_INDUCTIVE", "1") == "1",
           "dup_sha256": sha, "drop_conf": bool(cfg.get("drop_conf", False))}
    if canon:
        out.update({"input_format": C77, "in_dim": int(cfg.get("in_dim", 3)),
                    "canon_root": os.environ.get("FS_CANON_ROOT") or cfg.get("canon_root"),
                    "exclude_file": os.path.basename(dup_path)})
    return out
