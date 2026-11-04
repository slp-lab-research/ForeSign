"""Profile pretraining steps: data wait vs target / context / predictor forward vs backward vs
optimiser vs EMA (CUDA-event timing, synchronised per step; profiling only, not used in training).

  FS_CANON_ROOT=<dir> python -m tools.profile_step --config <cfg.yaml> [--steps 30] [--set k=v ...]
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

import torch
import yaml


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    from foresign import paths
    from foresign.build import build_model
    from foresign.canon_data import CanonClips
    from foresign.data import collate
    from foresign.inputs import layout_for
    from foresign.jepa import ema_momentum
    from foresign.masking import build_mask, temporal_boundary_target
    from foresign.train import time_valid_mask, make_optimizer, ema_update
    torch.backends.cuda.enable_cudnn_sdp(False)
    dev = "cuda"
    torch.manual_seed(0)
    layout = layout_for(cfg)
    model = build_model(cfg).to(dev)
    ds = CanonClips(datasets=cfg.get("datasets"), clip_len=cfg["clip_len"],
                    temperature=cfg.get("temperature", 0.5), seed=0,
                    root=paths.canon_root(cfg), in_dim=int(cfg.get("in_dim", 3)),
                    sample_unit=cfg.get("sample_unit", "windows"),
                    aug_rot_yaw_deg=cfg.get("aug_rot_yaw_deg", 0.0),
                    aug_rot_pitch_deg=cfg.get("aug_rot_pitch_deg", 0.0))
    nw = cfg.get("num_workers", 6)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"], num_workers=nw,
                                     collate_fn=collate, persistent_workers=nw > 0,
                                     pin_memory=True, prefetch_factor=4 if nw > 0 else None)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = make_optimizer(params, cfg)
    tp, L, V = cfg["t_patch"], cfg["clip_len"], 18

    timers = defaultdict(list)
    ev = {}

    def pre(name):
        def f(mod, inp):
            e = torch.cuda.Event(enable_timing=True); e.record()
            ev.setdefault(name, []).append([e, None])
        return f

    def post(name):
        def f(mod, inp, out):
            e = torch.cuda.Event(enable_timing=True); e.record()
            ev[name][-1][1] = e
        return f
    for name, mod in (("target_enc", model.target_ctx), ("context_enc", model.context),
                      ("predictor", model.predictor), ("tokenizer", model.tokenizer)):
        mod.register_forward_pre_hook(pre(name)); mod.register_forward_hook(post(name))

    it = iter(dl)
    total = cfg["total_steps"]
    for step in range(a.warmup + a.steps):
        ev.clear()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        pose, nv, _, _ = next(it)
        t_data = time.perf_counter() - t0
        pose = pose.to(dev, non_blocking=True); nv = nv.to(dev)
        e = {k: torch.cuda.Event(enable_timing=True) for k in ("s", "f", "b", "o", "m")}
        e["s"].record()
        valid = time_valid_mask(nv, L, tp, V, dev)
        mask = build_mask(cfg.get("masking", "m3"), pose.size(0), L // tp, V, dev,
                          valid=valid, cfg=cfg)
        bt = None
        if float(cfg.get("lambda_bound", 0.0)) > 0:
            bt, _ = temporal_boundary_target(pose, tp, layout=layout)
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, stats = model(pose, mask, valid, vreg=cfg.get("vreg", 0.0), boundary_target=bt,
                                lambda_bound=float(cfg.get("lambda_bound", 0.0)),
                                alpha_pred=float(cfg.get("alpha_pred", 1.0)),
                                beta=float(cfg.get("beta", 0.0)), tau=float(cfg.get("tau", 0.1)),
                                fast_stats=bool(cfg.get("fast_stats", False)),
                                want_stats=False)
        e["f"].record()
        loss.backward()
        e["b"].record()
        torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        opt.step()
        e["o"].record()
        ema_update(model, ema_momentum(step, total, cfg["ema_base"]), cfg)
        e["m"].record()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        if step < a.warmup:
            continue
        timers["data_wait"].append(t_data * 1e3)
        timers["forward_total"].append(e["s"].elapsed_time(e["f"]))
        timers["backward"].append(e["f"].elapsed_time(e["b"]))
        timers["clip+opt"].append(e["b"].elapsed_time(e["o"]))
        timers["ema"].append(e["o"].elapsed_time(e["m"]))
        timers["step_wall"].append(wall * 1e3)
        for name, pairs in ev.items():
            for i, (s, f) in enumerate(pairs):
                if f is not None:
                    timers[f"  fwd {name}#{i}"].append(s.elapsed_time(f))
    print(f"[profile] {a.config} steps={a.steps} batch={cfg['batch_size']} "
          f"tokens/window={(L // tp) * V} peak_mem={torch.cuda.max_memory_allocated() / 2**30:.2f}"
          f" GiB set={a.set}")
    for k, v in timers.items():
        v = sorted(v)
        print(f"  {k:28s} median {v[len(v) // 2]:8.1f} ms   mean {sum(v) / len(v):8.1f} ms")
    med = sorted(timers["step_wall"])[len(timers["step_wall"]) // 2]
    print(f"  => {1000 / med:.3f} it/s (median wall)")


if __name__ == "__main__":
    main()
