"""torch.compile equivalence + speed check (compile off vs on, same seed, same data).

Pretraining (default):
  FS_CANON_ROOT=<dir> CUDA_VISIBLE_DEVICES=<gpu> python -m tools.equiv_compile \
      --config configs/p2/p2_pred_s0.yaml [--steps 200] [--tol 0.005]
  Runs foresign.train twice in subprocesses (FS_COMPILE=0, then 1; FS_STEPS=steps, log every
  10 steps, fresh output dirs under runs/equiv_compile/). Prints per-log-step |Δloss| and
  |Δeff_rank| and the it/s of each run excluding the first 20 steps (compile warm-up). Exit 1 if
  max |Δloss| > tol.

Fine-tuning:
  ... python -m tools.equiv_compile --finetune --ckpt runs/p2/p2_pred_s0/step_2000.pt \
      [--dataset lsa64] [--epochs 1] [--ft-tol 0.01]
  Runs evals.finetune_islr_canon twice (FS_FT_COMPILE=0/1, 100% budget, seed 0); compares the
  epoch loss and top-1 and the it/s after the first 10 steps. Exit 1 if |Δloss| > ft-tol.

Both runs see identical inputs; differences come from compiled-kernel rounding (and, for
fine-tuning only, drop-path draws in the zero-padded last batch).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

from foresign import paths

CODE = paths.CODE
LINE = re.compile(r"step\s+(\d+) \| loss ([\d.]+) \|.*?eff_rank ([\d.na]+).*?\| ([\d.]+) it/s")


def _env(extra):
    env = dict(os.environ)
    env.update(extra)
    env["PYTHONPATH"] = CODE
    return env


def parse_train_log(path):
    """-> {step: (loss, eff_rank, cumulative it/s)}."""
    out = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LINE.search(line)
            if m:
                er = float(m[3]) if m[3] != "nan" else float("nan")
                out[int(m[1])] = (float(m[2]), er, float(m[4]))
    return out


def rate_after(rows, skip=20):
    """it/s between the first logged step >= skip and the last one. The logged rate is
    cumulative ((step+1)/elapsed), so elapsed(step) = (step+1)/rate."""
    steps = sorted(s for s in rows if s >= skip and rows[s][2] > 0)
    if len(steps) < 2:
        return None
    a, b = steps[0], steps[-1]
    ta, tb = (a + 1) / rows[a][2], (b + 1) / rows[b][2]
    return (b - a) / (tb - ta) if tb > ta else None


def compare_train(off, on):
    common = sorted(set(off) & set(on))
    dl = [abs(off[s][0] - on[s][0]) for s in common]
    de = [abs(off[s][1] - on[s][1]) for s in common]
    return common, dl, de


def run_train(cfg, steps, tag, compile_on, log_dir):
    out = os.path.join(paths.RUNS, "equiv_compile", f"{tag}_{'on' if compile_on else 'off'}")
    os.makedirs(log_dir, exist_ok=True)
    log = os.path.join(log_dir, f"{tag}_{'on' if compile_on else 'off'}.log")
    env = _env({"FS_COMPILE": "1" if compile_on else "0", "FS_STEPS": str(steps),
                "FS_OUTDIR": out, "FS_LOG_EVERY": "10", "FS_CKPT_EVERY": str(steps + 10**6)})
    t = time.time()
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.call([sys.executable, "-u", "-m", "foresign.train", "--config", cfg],
                             cwd=CODE, env=env, stdout=f, stderr=subprocess.STDOUT)
    return rc, log, time.time() - t


def main_train(a):
    tag = os.path.splitext(os.path.basename(a.config))[0] + time.strftime("_%Y%m%d_%H%M%S")
    log_dir = os.path.join(paths.RUNS, "equiv_compile", "logs")
    rows = {}
    for on in (False, True):
        rc, log, wall = run_train(a.config, a.steps, tag, on, log_dir)
        print(f"[equiv] compile={'on' if on else 'off'} rc={rc} wall {wall:.0f}s log {log}",
              flush=True)
        if rc != 0:
            print(f"[equiv] FAILED: training with compile={'on' if on else 'off'} exited {rc}")
            return 2
        rows[on] = parse_train_log(log)
    common, dl, de = compare_train(rows[False], rows[True])
    if not common:
        print("[equiv] no common logged steps"); return 2
    print(f"{'step':>6} {'loss_off':>9} {'loss_on':>9} {'|dloss|':>8} {'|deff_rank|':>11}")
    for s, x, y in zip(common, dl, de):
        print(f"{s:6d} {rows[False][s][0]:9.4f} {rows[True][s][0]:9.4f} {x:8.4f} {y:11.2f}")
    r_off, r_on = rate_after(rows[False]), rate_after(rows[True])
    print(f"[equiv] max |dloss| {max(dl):.4f} mean {sum(dl) / len(dl):.4f} | max |deff_rank| "
          f"{max(de):.2f} | it/s (after step 20) off {r_off and round(r_off, 3)} "
          f"on {r_on and round(r_on, 3)} speedup "
          f"{(r_on / r_off) if r_off and r_on else float('nan'):.2f}x")
    ok = max(dl) <= a.tol
    print(f"[equiv] {'PASS' if ok else 'FAIL'} (tol {a.tol})")
    return 0 if ok else 1


def main_finetune(a):
    res = os.path.join(paths.RUNS, "equiv_compile", "ft_" + time.strftime("%Y%m%d_%H%M%S"))
    log_dir = os.path.join(paths.RUNS, "equiv_compile", "logs")
    os.makedirs(log_dir, exist_ok=True)
    out = {}
    for on in (False, True):
        tag = f"equiv_compile_{'on' if on else 'off'}"
        log = os.path.join(log_dir, os.path.basename(res) + f"_{tag}.log")
        env = _env({"FS_FT_COMPILE": "1" if on else "0", "FS_RESULTS_DIR": res,
                    "FS_FT_CKPT": paths.resolve_out(a.ckpt), "FS_TAG": tag,
                    "FS_FT_DS": a.dataset, "FS_FT_BUDGET": "100", "FS_FT_SEED": "0",
                    "FS_FT_EPOCHS": str(a.epochs), "FS_TAP_DEPTH": "0"})
        with open(log, "w", encoding="utf-8") as f:
            rc = subprocess.call([sys.executable, "-u", "-m", "evals.finetune_islr_canon"],
                                 cwd=CODE, env=env, stdout=f, stderr=subprocess.STDOUT)
        p = os.path.join(res, f"ft_{tag}_{a.dataset}_100pct_s0.json")
        if not os.path.exists(p):
            print(f"[equiv-ft] FAILED compile={'on' if on else 'off'} rc={rc}; see {log}")
            return 2
        out[on] = json.load(open(p, encoding="utf-8"))
        d = out[on]
        print(f"[equiv-ft] compile={'on' if on else 'off'} loss {d['epoch_loss_last']:.4f} "
              f"top1 {d['top1']:.4f} it/s(after 10) {d['train_it_per_s_after10']} log {log}")
    dloss = abs(out[False]["epoch_loss_last"] - out[True]["epoch_loss_last"])
    r0, r1 = out[False]["train_it_per_s_after10"], out[True]["train_it_per_s_after10"]
    print(f"[equiv-ft] |dloss| {dloss:.4f} |dtop1| {abs(out[False]['top1'] - out[True]['top1']):.4f}"
          f" speedup {(r1 / r0) if r0 and r1 else float('nan'):.2f}x")
    ok = dloss <= a.ft_tol
    print(f"[equiv-ft] {'PASS' if ok else 'FAIL'} (tol {a.ft_tol})")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--tol", type=float, default=0.005)
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--dataset", default="lsa64")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--ft-tol", type=float, default=0.01)
    a = ap.parse_args(argv)
    if a.finetune:
        if not a.ckpt:
            ap.error("--finetune needs --ckpt")
        return main_finetune(a)
    if not a.config:
        ap.error("--config required")
    return main_train(a)


if __name__ == "__main__":
    sys.exit(main())
