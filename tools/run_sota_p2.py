"""Launcher for the SOTA-comparable protocols (docs/design/2026-10-30-phase2-c77-3d-adaptation.md, addendum 2026-11-03): `islr_std` and
`cslr_cosign`, for every checkpoint with identical recipes.

  python -m tools.run_sota_p2 [--protocols islr,cslr] [--datasets ...] [--ckpts a.pt,b.pt]
                              [--compile] [--dry-run] [--summary]

Order: protocol -> dataset -> budget -> seed -> checkpoint (every checkpoint gets a cell before the
next cell starts). Finished jobs (result JSON with complete=true) are skipped; islr_std resumes
from its per-epoch state; a last.pt still training is skipped (as in run_finetune_p2). Each job is
a subprocess (logs in runs/p2/sota_logs/). GPU = CUDA_VISIBLE_DEVICES; compile for islr_std with
--compile (static shapes); CSLR runs eager (variable lengths).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

from foresign import paths

DEFAULT_CKPTS = ("runs/p2/p2_random_init_s0/last.pt", "runs/p2/p2_pred_long_s0/last.pt",
                 "runs/p2/p2_ms_long_s0/last.pt", "runs/p2/p2_pred_rot_s0/last.pt")
ISLR_CELLS = (("wlasl", 100, (0,)), ("wlasl", 20, (0, 1, 2)), ("autsl", 100, (0,)),
              ("lsa64_signer", 100, (0,)), ("lsa64_rand", 100, (0, 1, 2)),
              ("asl_citizen", 100, (0,)))
CSLR_CELLS = (("ph14t", None, (0,)), ("ph14", None, (0,)))


def _name(proto, tag, ds, budget, seed):
    if proto == "islr":
        from evals.islr_std import result_name
        return result_name(tag, ds, budget, seed)
    from evals.cslr_cosign import result_name
    return result_name(tag, ds, seed)


def is_done(name):
    p = os.path.join(os.environ.get("FS_RESULTS_DIR", paths.RESULTS), name + ".json")
    try:
        return bool(json.load(open(p, encoding="utf-8")).get("complete"))
    except Exception:
        return False


def jobs(ckpts, protocols, datasets=None):
    from tools.run_finetune_p2 import ckpt_tag
    tags, problems = [], []
    for c in ckpts:
        t, prob = ckpt_tag(c)
        (problems.append(prob) if prob else tags.append((paths.resolve_out(c), t)))
    out = []
    for proto in protocols:
        cells = ISLR_CELLS if proto == "islr" else CSLR_CELLS
        for ds, budget, seeds in cells:
            if datasets and ds not in datasets:
                continue
            for s in seeds:
                for ck, tag in tags:
                    out.append({"proto": proto, "ckpt": ck, "tag": tag, "ds": ds,
                                "budget": budget, "seed": s,
                                "name": _name(proto, tag, ds, budget, s)})
    return out, problems


def job_env(j, compile_on):
    if j["proto"] == "islr":
        return "evals.islr_std", {"FS_STD_CKPT": j["ckpt"], "FS_TAG": j["tag"],
                                  "FS_STD_DS": j["ds"], "FS_STD_BUDGET": str(j["budget"]),
                                  "FS_STD_SEED": str(j["seed"]),
                                  "FS_STD_COMPILE": "1" if compile_on else "0"}
    return "evals.cslr_cosign", {"FS_CC_CKPT": j["ckpt"], "FS_TAG": j["tag"], "FS_CC_DS": j["ds"],
                                 "FS_CC_SEED": str(j["seed"]), "FS_TAP_DEPTH": "0"}


def summary(ckpts, protocols):
    todo, _ = jobs(ckpts, protocols)
    res = os.environ.get("FS_RESULTS_DIR", paths.RESULTS)
    cells = {}
    for j in todo:
        p = os.path.join(res, j["name"] + ".json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        if not d.get("complete"):
            continue
        key = (j["proto"], j["ds"], j["budget"], j["tag"])
        if j["proto"] == "islr":
            cells.setdefault(key, []).append((d["pi_top1"], d["pc_top1"], d["pi_top5"]))
        else:
            cells.setdefault(key, []).append((d["dev_wer_best"], d["test_wer_at_best_dev"]))
    for (proto, ds, b, tag), v in sorted(cells.items()):
        cols = list(zip(*v))
        fmt = " ".join(f"{st.mean(c):.4f}±{(st.stdev(c) if len(c) > 1 else 0):.4f}" for c in cols)
        what = "P-I P-C P-I@5" if proto == "islr" else "devWER testWER"
        print(f"{proto:4s} {ds:13s} {str(b or ''):>3s} {tag:28s} n={len(v)} {what}: {fmt}")


def main(argv=None):
    from tools.run_queue import _run
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", default=",".join(DEFAULT_CKPTS))
    ap.add_argument("--protocols", default="islr,cslr")
    ap.add_argument("--datasets", default="")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args(argv)
    ckpts = [c for c in a.ckpts.split(",") if c]
    protocols = [p for p in a.protocols.split(",") if p]
    if a.summary:
        summary(ckpts, protocols)
        return 0
    todo, problems = jobs(ckpts, protocols, [d for d in a.datasets.split(",") if d])
    for p in problems:
        print(f"[sota] SKIP {p}", flush=True)
    failures = []
    for j in todo:
        if is_done(j["name"]):
            continue
        print(f"[sota] {'todo' if a.dry_run else 'run'} {j['name']}", flush=True)
        if a.dry_run:
            continue
        module, env = job_env(j, a.compile)
        log = os.path.join(paths.RUNS, "p2", "sota_logs", j["name"] + ".log")
        rc = _run(module, env, log)
        if not is_done(j["name"]):
            failures.append(f"{j['name']} rc={rc} see {log}")
            print(f"[sota] FAILED {failures[-1]}", flush=True)
    print(f"[sota] done, {len(failures)} failure(s)", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
