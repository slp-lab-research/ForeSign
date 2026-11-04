"""Every filesystem location used by the code. Nothing here points at a specific machine; all
locations are relative to the repository or set through environment variables:

  FS_PROJECT_ROOT   working root holding data_ann/, cache/, results/, runs/ (default: the
                    repository root)
  FS_CANON_ROOT     directory with the C77-3D canon HDF5 files (<ds>.h5, youtube_asl_b*.h5);
                    a config's `canon_root` key is used when the variable is unset
  FS_ANN_DIR        directory searched first for annotation files (CSLR gloss JSONs,
                    autsl_labels.json, PHOENIX .stm scoring files); then <project>/data_ann
  FS_CANON_EXCLUDE  pretraining exclusion list (default <project>/data_ann/canon_exclude.json)
  FS_RESULTS_DIR    result JSON directory used by tools.run_queue (default <project>/results)
"""
from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)                      # repository root (code)
CODE = REPO
PROJECT = os.environ.get("FS_PROJECT_ROOT", REPO)  # working root (annotations, cache, results)

DATA_ANN = os.path.join(PROJECT, "data_ann")
# Annotation search order: $FS_ANN_DIR (if set), then <project>/data_ann.
ANN_DIRS = (DATA_ANN,)


def ann_file(name: str) -> str:
    """First existing `name` in $FS_ANN_DIR / ANN_DIRS; else the first candidate."""
    env = os.environ.get("FS_ANN_DIR")
    cands = ((env,) if env else ()) + tuple(ANN_DIRS)
    for d in cands:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return os.path.join(cands[0], name)


CACHE = os.path.join(PROJECT, "cache")
CANON_EXCLUDE = os.environ.get("FS_CANON_EXCLUDE") or os.path.join(DATA_ANN, "canon_exclude.json")


def canon_root(cfg: dict | None = None) -> str:
    root = os.environ.get("FS_CANON_ROOT") or (cfg or {}).get("canon_root")
    if not root:
        raise RuntimeError("canon data root not set: export FS_CANON_ROOT=<dir with canon *.h5> "
                           "or set canon_root in the config")
    if not os.path.isdir(root):
        raise FileNotFoundError(f"canon root {root!r} is not a directory")
    return root


RESULTS = os.path.join(PROJECT, "results")
# Optional Phase-1 near-duplicate list consumed by tools.build_canon_exclude --old-dup.
DUP_EXCLUDE = os.path.join(RESULTS, "dup_exclude.json")
RUNS = os.path.join(PROJECT, "runs")


def resolve_out(p: str) -> str:
    """Config output_dir: absolute as given, relative = under the project root (portable)."""
    return p if os.path.isabs(p) or p.startswith("/") else os.path.join(PROJECT, *p.split("/"))
