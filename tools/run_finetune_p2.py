"""Low-label fine-tuning ISLR launcher (Phase 2): checkpoints x datasets x budgets x seeds.

  python -m tools.run_finetune_p2                     # default checkpoints, all datasets/budgets
  python -m tools.run_finetune_p2 --dry-run           # list jobs (done / todo / missing ckpt)
  python -m tools.run_finetune_p2 --summary           # mean ± sd table from result JSONs
  python -m tools.run_finetune_p2 --ckpts runs/p2/p2_ms_s0/step_2000.pt,... --datasets autsl

Each job runs `evals.finetune_islr_canon` in a subprocess (failures isolated, logs in
runs/p2/ft_logs/). Finished jobs (result JSON with complete=true) are skipped; an interrupted job
resumes from its per-epoch state. A last.pt counts only once its step reaches the config's
total_steps (a run still training is skipped). Order: dataset -> budget -> seed -> checkpoint, so
every checkpoint gets the same cell before the next cell starts. GPU = CUDA_VISIBLE_DEVICES.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

import torch

from evals.finetune_islr_canon import SEEDS, result_name
from foresign import paths

DEFAULT_CKPTS = ("runs/p2/p2_random_init_s0/last.pt", "runs/p2/p2_ms_s0/step_2000.pt",
                 "runs/p2/p2_ms_s0/last.pt", "runs/p2/p2_pred_s0/step_2000.pt",
                 "runs/p2/p2_pred_s0/last.pt")
DATASETS = ("autsl", "wlasl", "lsa64")
BUDGETS = (10, 100)


def ckpt_tag(path):
    """-> (tag, problem). tag = <run>_k<step>_L0; problem set if missing / unfinished."""
    p = paths.resolve_out(path)
    if not os.path.exists(p):
        return None, f"missing {path}"
    ck = torch.load(p, map_location="cpu", weights_only=False)
    step, cfg = int(ck.get("step", 0)), ck.get("cfg", {})
    if os.path.basename(p) == "last.pt" and not cfg.get("random_init") and step > 0 \
            and step < int(cfg.get("total_steps", 0)):
        return None, f"unfinished {path} (step {step} < {cfg.get('total_steps')})"
    return f"{os.path.basename(os.path.dirname(p))}_k{step}_L0", None


def results_dir():
    return os.environ.get("FS_RESULTS_DIR", paths.RESULTS)


def is_done(name):
    p = os.path.join(results_dir(), name + ".json")
    if not os.path.exists(p):
        return False
    try:
        with open(p, encoding="utf-8") as f:
            return bool(json.load(f).get("complete"))
    except Exception:
        return False


def jobs(ckpts, datasets, budgets):
    tags, problems = [], []
    for c in ckpts:
        t, prob = ckpt_tag(c)
        (problems.append(prob) if prob else tags.append((paths.resolve_out(c), t)))
    out = []
    for ds in datasets:
        for b in budgets:
            for s in SEEDS[b]:
                for ck, tag in tags:
                    out.append({"ckpt": ck, "tag": tag, "ds": ds, "budget": b, "seed": s,
                                "name": result_name(tag, ds, b, s)})
    return out, problems


def summary(ckpts, datasets, budgets):
    rows = []
    for ds in datasets:
        for b in budgets:
            for c in ckpts:
                t, prob = ckpt_tag(c)
                if prob:
                    continue
                vals = []
                for s in SEEDS[b]:
                    p = os.path.join(results_dir(), result_name(t, ds, b, s) + ".json")
                    if os.path.exists(p):
                        d = json.load(open(p, encoding="utf-8"))
                        if d.get("complete"):
                            vals.append((d["top1"], d["top5"]))
                if vals:
                    t1 = [v[0] for v in vals]; t5 = [v[1] for v in vals]
                    sd = st.stdev(t1) if len(t1) > 1 else 0.0
                    rows.append(f"{ds:6s} {b:3d}% {t:28s} n={len(vals)} top1 {st.mean(t1):.4f}"
                                f"±{sd:.4f} top5 {st.mean(t5):.4f}")
    print("\n".join(rows) if rows else "(no finished fine-tuning results)")
    return rows


def main(argv=None):
    from tools.run_queue import _run
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", default=",".join(DEFAULT_CKPTS))
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--budgets", default=",".join(map(str, BUDGETS)))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args(argv)
    ckpts = [c for c in a.ckpts.split(",") if c]
    datasets = [d for d in a.datasets.split(",") if d]
    budgets = [int(b) for b in a.budgets.split(",") if b]
    if a.summary:
        summary(ckpts, datasets, budgets)
        return 0
    todo, problems = jobs(ckpts, datasets, budgets)
    for p in problems:
        print(f"[ft-queue] SKIP {p}", flush=True)
    failures = []
    for j in todo:
        if is_done(j["name"]):
            continue
        print(f"[ft-queue] {'todo' if a.dry_run else 'run'} {j['name']}", flush=True)
        if a.dry_run:
            continue
        log = os.path.join(paths.RUNS, "p2", "ft_logs", j["name"] + ".log")
        rc = _run("evals.finetune_islr_canon", {
            "FS_FT_CKPT": j["ckpt"], "FS_TAG": j["tag"], "FS_FT_DS": j["ds"],
            "FS_FT_BUDGET": str(j["budget"]), "FS_FT_SEED": str(j["seed"]),
            "FS_TAP_DEPTH": "0"}, log)
        if not is_done(j["name"]):
            failures.append(f"{j['name']} rc={rc} see {log}")
            print(f"[ft-queue] FAILED {failures[-1]}", flush=True)
        elif rc != 0:
            print(f"[ft-queue] WARN {j['name']} rc={rc} but result complete", flush=True)
    print(f"[ft-queue] done, {len(failures)} failure(s)", flush=True)
    for f in failures:
        print(f"  FAILED {f}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
