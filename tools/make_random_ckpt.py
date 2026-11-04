"""Save a random-init ForeSign checkpoint (baseline for every evaluator).

  python -m tools.make_random_ckpt --config configs/p2/p2_random_init_s0.yaml \
      --out runs/random_init/last.pt --seed 0"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    from foresign.build import build_model
    torch.manual_seed(a.seed)
    model = build_model(cfg)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({"model": model.state_dict(), "step": 0, "cfg": cfg}, a.out)
    print(f"[random_ckpt] wrote {a.out} seed={a.seed}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
