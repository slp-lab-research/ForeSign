"""Resumable queue: train each config, then evaluate checkpoints x readouts.

  FS_CANON_ROOT=<dir> python -m tools.run_queue configs/p2/p2_*_s0.yaml
Re-running skips finished training and any evaluation whose result JSON exists.
Works on Windows and Linux; set FS_PROJECT_ROOT to keep runs/ results/ cache/ outside the repo.

Configs must be input_format c77 (configs/p2/). Per checkpoint x tap L0/L1 the jobs are the canon
ISLR probes (AUTSL, WLASL, LSA64), frozen-CTC CSLR on canon PHOENIX-2014T and the dataset-ID probe;
FS_QUEUE_EXTRA=cslr_ph14 runs canon PHOENIX-2014 CSLR only. `random_init: true` configs get a
step-0 checkpoint (tools.make_random_ckpt) instead of training. FS_CANON_ROOT and
CUDA_VISIBLE_DEVICES are passed through to every job; the GPU is chosen only by the launcher."""
from __future__ import annotations

import glob
import os
import subprocess
import sys

import torch
import yaml

from foresign import paths

DEFAULT_CKPTS = "step_2000.pt,step_6000.pt,last.pt"
TAPS = (0, 1)


def taps():
    """Readout taps for evaluation jobs: FS_QUEUE_TAPS (default "0,1"; e.g. "0" = L0 only)."""
    v = os.environ.get("FS_QUEUE_TAPS", "").strip()
    return tuple(int(t) for t in v.split(",") if t.strip()) if v else TAPS
CODE = paths.CODE
ENV_ALLOW = {"FS_PROJECT_ROOT", "FS_RESULTS_DIR", "FS_ALLOW_NO_DUP",
             "FS_INDUCTIVE", "FS_ANN_DIR", "FS_CANON_ROOT", "FS_DSID_DATASETS",
             "FS_FT_CKPT_LAYERS", "FS_FAST_OPS", "FS_COMPILE", "FS_FT_COMPILE",
             "FS_CANON_EXCLUDE", "FS_PHOENIX_EVAL_DIR"}
MODULES = {"islr_canon": "evals.islr_canon",
           "cslr": "evals.cslr_phoenix", "dsid": "evals.dsid_probe"}
RESULT_NAMES = {"islr_canon": "islr_canon_{}.json",
                "cslr": "cslr_frozen_{}_s0.json", "dsid": "dsid_{}.json"}


def ckpt_names():
    return tuple(x.strip() for x in os.environ.get("FS_QUEUE_CKPTS", DEFAULT_CKPTS).split(",")
                 if x.strip())


def result_path(kind, tag):
    return os.path.join(os.environ.get("FS_RESULTS_DIR", paths.RESULTS),
                        RESULT_NAMES[kind].format(tag))


def extra_mode():
    return os.environ.get("FS_QUEUE_EXTRA", "").strip() or None


def canon_eval_jobs(name, ckpt_steps, run_dir):
    """Phase 2 (c77) job set; see module docstring."""
    extra = extra_mode()
    if extra not in (None, "cslr_ph14"):
        raise ValueError(f"FS_QUEUE_EXTRA={extra} not available for c77 configs")
    smoke = os.environ.get("FS_QUEUE_SMOKE") == "1"
    jobs = []
    for fn, step in ckpt_steps.items():
        ck = os.path.join(run_dir, fn)
        for tap in taps():
            tag = f"{name}_k{step}_L{tap}"
            cslr = {"FS_CSLR_CKPT": ck, "FS_CSLR_FREEZE": "1",
                    "FS_CSLR_EPOCHS": "1" if smoke else "40", "FS_CSLR_SEED": "0",
                    "FS_TAP_DEPTH": str(tap)}
            if smoke:
                cslr["FS_CSLR_LIMIT"] = "64"
            if extra == "cslr_ph14":
                jobs.append({"kind": "cslr", "tag": tag + "_ph14", "env": {
                    **cslr, "FS_TAG": tag + "_ph14", "FS_CSLR_DATASET": "canon_ph14"}})
                continue
            islr = {"FS_ADAPTER": f"foresign:{ck}", "FS_DATASETS": "autsl,wlasl,lsa64",
                    "FS_SEEDS": "0" if smoke else "0,1,2", "FS_TAP_DEPTH": str(tap),
                    "FS_TAG": tag}
            if smoke:
                islr["FS_ISLR_LIMIT"] = "300"
            dsid = {"FS_DSID_CKPT": ck, "FS_TAP_DEPTH": str(tap), "FS_TAG": tag}
            if smoke:
                dsid["FS_DSID_N"] = "30"
            jobs.append({"kind": "islr_canon", "tag": tag, "env": islr})
            jobs.append({"kind": "cslr", "tag": tag, "env": {
                **cslr, "FS_TAG": tag, "FS_CSLR_DATASET": "canon_ph14t"}})
            jobs.append({"kind": "dsid", "tag": tag, "env": dsid})
    return jobs


def eval_jobs(name, ckpt_steps, run_dir, canon=True):
    """Evaluation jobs for a c77 config (the only input format in this release)."""
    if not canon:
        raise ValueError("only input_format c77 configs are supported")
    return canon_eval_jobs(name, ckpt_steps, run_dir)


def child_env(env_extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("FS_") or k in ENV_ALLOW}
    env.update(env_extra)
    env["PYTHONPATH"] = CODE
    return env


def _run(module, env_extra, log_path, args=()):
    """Run one job; returns the process return code."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        return subprocess.call([sys.executable, "-u", "-m", module, *args], cwd=CODE,
                               env=child_env(env_extra), stdout=log, stderr=subprocess.STDOUT)


def _ckpt_step(path):
    return int(torch.load(path, map_location="cpu", weights_only=False)["step"])


def _last_step(last):
    try:
        return _ckpt_step(last) if os.path.exists(last) else None
    except Exception as e:  # corrupt/partial checkpoint
        print(f"[queue] cannot read {last}: {e}", flush=True)
        return None


def process(cfg_path):
    """Train (if needed) then evaluate; returns a list of failure strings."""
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    name = os.path.splitext(os.path.basename(cfg_path))[0]
    run_dir = paths.resolve_out(cfg["output_dir"])
    if (cfg.get("input_format") or "resona77") != "c77":
        return [f"{name}: input_format must be c77"]
    logs = os.path.join(run_dir, "logs")
    last = os.path.join(run_dir, "last.pt")
    need = cfg["total_steps"] - 1
    failures = []
    st = _last_step(last)
    no_train = os.environ.get("FS_QUEUE_NOTRAIN") == "1"   # evaluate existing ckpts only
    if cfg.get("random_init"):
        if st is None:
            print(f"[queue] random-init checkpoint {name}", flush=True)
            log = os.path.join(logs, "random_init.log")
            rc = _run("tools.make_random_ckpt", {}, log,
                      ("--config", cfg_path, "--out", last, "--seed", str(cfg.get("seed", 0))))
            st = _last_step(last)
            if rc != 0 or st is None:
                msg = f"random-init {name} rc={rc}; see {log}"
                print(f"[queue] FAILED {msg}", flush=True)
                return [msg]
    elif not extra_mode() and not no_train and (st is None or st < need):
        print(f"[queue] train {name}", flush=True)
        log = os.path.join(logs, "train.log")
        rc = _run("foresign.train", {}, log, ("--config", cfg_path))
        st = _last_step(last)
        if rc != 0 or st is None or st < need:
            msg = f"train {name} rc={rc} last step {st} < {need}; see {log}"
            print(f"[queue] FAILED {msg}", flush=True)
            return [msg]
    steps = {}
    for fn in ckpt_names():
        fp = os.path.join(run_dir, fn)
        s = _last_step(fp)
        if s is not None:
            steps[fn] = s
    jobs = eval_jobs(name, steps, run_dir)
    for job in jobs:
        if os.path.exists(result_path(job["kind"], job["tag"])):
            continue
        module = MODULES[job["kind"]]
        log = os.path.join(logs, f"{job['kind']}_{job['tag']}.log")
        print(f"[queue] {job['kind']} {job['tag']}", flush=True)
        rc = _run(module, job["env"], log)
        if rc != 0 and os.path.exists(result_path(job["kind"], job["tag"])):
            print(f"[queue] WARN {job['kind']} {job['tag']} rc={rc} but result written "
                  f"(interpreter-exit crash); treating as success", flush=True)
            continue
        if rc != 0:
            msg = f"{job['kind']} {job['tag']} rc={rc} see {log}"
            print(f"[queue] FAILED {msg}", flush=True)
            failures.append(msg)
    return failures


def main(patterns):
    cfgs = sorted(p for pat in patterns for p in glob.glob(pat))
    failures = []
    for c in cfgs:
        failures += process(c)
    if failures:
        print(f"[queue] done with {len(failures)} failure(s):", flush=True)
        for f in failures:
            print(f"  FAILED {f}", flush=True)
        return 1
    print("[queue] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
