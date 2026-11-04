"""Write the Phase 2 pretraining configs (pre-registered constants; C77-3D canon data).

  python -m tools.make_configs p2         -> configs/p2/p2_{random_init,pred,nce,ms}_s0.yaml

The other configs in configs/p2/ (long / rot / large variants) were derived from these by hand.
canon_root is left null: set FS_CANON_ROOT (or edit the key) before training.
"""
from __future__ import annotations

import os
import sys

import yaml

from foresign import paths

BASE = dict(
    d_model=192, depth=8, n_heads=6, pred_dim=96, pred_depth=3, pred_heads=6,
    ema_base=0.999, masking="rand78", drop_conf=True, group_layout="anatomical",
    batch_size=64, lr=1.5e-4, weight_decay=0.05, grad_clip=1.0, warmup_steps=500,
    total_steps=16000, const_until=12000, temperature=0.5, num_workers=6,
    resume=True, collapse_std=0.02, collapse_patience=300, log_every=50,
    ckpt_every=2000, datasets=None, tau=0.1,
)
OBJECTIVES = {
    "pred": {"alpha_pred": 1.0, "beta": 0.0, "lambda_bound": 0.0},
    "nce": {"alpha_pred": 0.0, "beta": 1.0, "lambda_bound": 0.0},
    "ms": {"alpha_pred": 1.0, "beta": 0.3, "lambda_bound": 0.2},
}
RUN_DIR = os.path.join(paths.RUNS, "p2")


def spec(name, objective, seed, t_patch=2, clip_len=128, **over):
    return {"name": name, **BASE, **OBJECTIVES[objective], "seed": seed,
            "t_patch": t_patch, "clip_len": clip_len, **over}


# Phase 2: same model / optimisation constants as the earlier 2D pilots; only the data and the
# input layout change (C77-3D coords + joint mask, in_dim 3, no conf channel). canon_root is
# null (FS_CANON_ROOT supplies it). output_dir is relative to the project root.
P2_CANON_ROOT = None
P2_DATA = dict(input_format="c77", in_dim=3, drop_conf=False, sample_unit="windows",
               canon_root=P2_CANON_ROOT, datasets=None)
P2_CFG_DIR = os.path.join(paths.CODE, "configs", "p2")


def p2_specs(seed=0):
    out = [spec(f"p2_{o}_s{seed}", o, seed, **P2_DATA) for o in ("pred", "nce", "ms")]
    # random-init baseline: same architecture, no training (the queue writes a step-0 ckpt)
    out.insert(0, spec(f"p2_random_init_s{seed}", "pred", seed, random_init=True, **P2_DATA))
    return out


def write_configs(specs, out_dir=P2_CFG_DIR, run_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for sp in specs:
        cfg = {k: v for k, v in sp.items() if k != "name"}
        cfg["output_dir"] = os.path.join(run_dir or RUN_DIR, sp["name"]).replace("\\", "/")
        p = os.path.join(out_dir, f"{sp['name']}.yaml")
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=True)
        written.append(p)
    return written


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "p2"
    if which != "p2":
        sys.exit(f"unknown config set {which!r} (only 'p2')")
    written = write_configs(p2_specs(), P2_CFG_DIR, run_dir="runs/p2")
    for p in written:
        print(p)
