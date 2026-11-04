"""One JSON file per evaluation run; tables and figures are generated from these."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

from foresign import paths


def _clean(o):
    """Strict JSON: map float NaN/inf to None, recursively."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def write_result(name: str, payload: dict) -> Path:
    out_dir = Path(os.environ.get("FS_RESULTS_DIR", paths.RESULTS))
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        "name": name,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "env": {k: v for k, v in os.environ.items() if k.startswith("FS_")},
        **payload,
    }
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(_clean(rec), indent=2, allow_nan=False), encoding="utf-8")
    print(f"[result] wrote {path}", flush=True)
    return path
