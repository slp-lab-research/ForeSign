"""ForeSign JEPA pretraining loop. Single GPU.

Collapse diagnostics are logged; the run aborts early (exit code 3) with a clear message if the
target features collapse.

  FS_CANON_ROOT=<canon dir> python -u -m foresign.train --config configs/p2/p2_pred_s0.yaml
Env overrides: FS_STEPS, FS_OUTDIR, FS_DATASETS, FS_BATCH, FS_CKPT_EVERY, FS_LOG_EVERY,
FS_MASKING, FS_VREG, FS_SEED, FS_CONST_UNTIL, FS_INIT_CKPT (weight-only warm start).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch
import torch.backends.cuda
import yaml

from foresign.build import build_model
from foresign.inputs import C77, layout_for
from foresign.jepa import ema_momentum
from foresign.masking import build_mask


def time_valid_mask(n_valid, clip_len, t_patch, V, device):
    """(B,) valid frames → (B,TP,V) bool valid (non-pad) token mask."""
    tp = clip_len // t_patch
    vtp = torch.div(n_valid + t_patch - 1, t_patch, rounding_mode="floor").clamp(1, tp)
    ar = torch.arange(tp, device=device)
    tvalid = ar[None, :] < vtp[:, None].to(device)       # (B,TP)
    return tvalid[:, :, None].expand(-1, -1, V).clone()


COLLAPSE_EXIT_CODE = 3


def make_optimizer(params, cfg):
    """AdamW as in Phase 1a; cfg fused_adamw=True uses the fused CUDA kernel (same update rule)."""
    from foresign.build import speed_flag
    kw = {"fused": True} if speed_flag(cfg, "fused_adamw") else {}
    return torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                             betas=(0.9, 0.95), **kw)


@torch.no_grad()
def ema_update(model, m, cfg):
    """EMA target update; cfg foreach_ema=True does the same k = m*k + (1-m)*q with
    multi-tensor (foreach) kernels instead of a Python loop over parameters."""
    from foresign.build import speed_flag
    if not speed_flag(cfg, "foreach_ema"):
        model.update_ema(m)
        return
    for src, dst in ((model.tokenizer, model.target_tok), (model.context, model.target_ctx)):
        q = [p.detach() for p in src.parameters()]
        k = list(dst.parameters())
        torch._foreach_mul_(k, m)
        torch._foreach_add_(k, q, alpha=1 - m)


def collapse_message(step, std_thresh, n):
    return (f"[foresign] COLLAPSE at step {step} (feat_std<{std_thresh} for {n} steps) - "
            f"abort. R-2 triggered: try CE+centering loss or stronger EMA.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config, encoding="utf-8"))
    for k, env in [("total_steps", "FS_STEPS"), ("output_dir", "FS_OUTDIR"),
                   ("batch_size", "FS_BATCH"), ("ckpt_every", "FS_CKPT_EVERY"),
                   ("log_every", "FS_LOG_EVERY")]:
        if os.environ.get(env):
            cfg[k] = int(os.environ[env]) if k != "output_dir" else os.environ[env]
    if os.environ.get("FS_MASKING"):              # gate: m3|m3prime|m1|m2|m1m2|part78|rand78
        cfg["masking"] = os.environ["FS_MASKING"]
    if os.environ.get("FS_VREG"):
        cfg["vreg"] = float(os.environ["FS_VREG"])
    if os.environ.get("FS_SEED"):                 # 3-seed confirmation
        cfg["seed"] = int(os.environ["FS_SEED"])
    if os.environ.get("FS_CONST_UNTIL"):
        cfg["const_until"] = int(os.environ["FS_CONST_UNTIL"])
    # Optional SSL-time language SupCon (off by default and in every Phase-2 config).
    lang_weight = float(os.environ.get("FS_LANG_WEIGHT", "0.0"))
    lang_temp = float(os.environ.get("FS_LANG_TEMP", "0.125"))
    lang_proj = os.environ.get("FS_LANG_PROJ", "0") == "1"
    lang_center = os.environ.get("FS_LANG_CENTER", "0") == "1"
    torch.backends.cuda.enable_cudnn_sdp(False)          # cuDNN SDPA off (as in all reported runs)
    dev = "cuda"                     # GPU choice: CUDA_VISIBLE_DEVICES from the launcher
    torch.manual_seed(cfg.get("seed", 0))
    layout = layout_for(cfg)                             # validates input_format / in_dim
    canon = layout.fmt == C77
    if not canon:
        raise ValueError("this release trains on C77-3D canon data only (input_format: c77)")
    from foresign.canon_data import CanonClips
    from foresign.canon_splits import DEFAULT_PRETRAIN as DEFAULT_DATASETS
    from foresign.data import collate
    datasets = (os.environ["FS_DATASETS"].split(",")
                if os.environ.get("FS_DATASETS") else (cfg.get("datasets") or DEFAULT_DATASETS))

    model = build_model(cfg).to(dev)
    from foresign.build import compile_flag, compile_jepa
    use_compile = compile_flag(cfg)
    if use_compile:            # in place: state_dict keys / checkpoint format unchanged
        print(f"[foresign] torch.compile: {compile_jepa(model)}", flush=True)
    V = 18
    clip_len, tp = cfg["clip_len"], cfg["t_patch"]

    from foresign import paths as _paths
    ds = CanonClips(datasets=datasets, clip_len=clip_len,
                    temperature=cfg.get("temperature", 0.5), seed=cfg.get("seed", 0),
                    root=_paths.canon_root(cfg), in_dim=int(cfg.get("in_dim", 3)),
                    sample_unit=cfg.get("sample_unit", "windows"),
                    aug_rot_yaw_deg=cfg.get("aug_rot_yaw_deg", 0.0),
                    aug_rot_pitch_deg=cfg.get("aug_rot_pitch_deg", 0.0))
    print(f"[foresign] view aug: rot yaw ±{ds.aug_yaw}° pitch ±{ds.aug_pitch}° "
          f"(0 = off; pretraining stream only)", flush=True)
    print(f"[foresign] canon root={ds.root} versions={sorted(set(ds.versions.values()))} "
          f"weights={dict(zip(ds.names, [round(w, 4) for w in ds.weights]))}", flush=True)
    nw = cfg.get("num_workers", 6)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"], num_workers=nw,
                                     collate_fn=collate, persistent_workers=nw > 0,
                                     pin_memory=True, prefetch_factor=4 if nw > 0 else None)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = make_optimizer(params, cfg)
    total = cfg["total_steps"]
    warm = cfg.get("warmup_steps", 500)

    def lr_at(s):
        if s < warm:
            return cfg["lr"] * s / warm
        if s < cfg.get("const_until", int(total * 0.8)):
            return cfg["lr"]
        prog = (s - cfg.get("const_until", int(total * 0.8))) / max(1, total - cfg.get("const_until", int(total * 0.8)))
        return cfg["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

    from foresign.paths import resolve_out
    outdir = resolve_out(cfg["output_dir"]); os.makedirs(outdir, exist_ok=True)
    start = 0
    last = os.path.join(outdir, "last.pt")
    if cfg.get("resume", True) and os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        # strict=False: the checkpoint may predate a module added later (e.g.
        # wpp_match_scale) -- those get fresh init, everything the checkpoint DOES
        # contain still loads. Report any mismatch so an unexpected miss is visible.
        miss = model.load_state_dict(ck["model"], strict=False)
        try:
            opt.load_state_dict(ck["opt"])
            opt_resumed = True
        except (ValueError, RuntimeError) as e:
            # param count changed since the checkpoint was saved (new module ->
            # new trainable params) -- Adam's per-param state can't be remapped,
            # so start the optimizer fresh; step count (and thus the LR schedule)
            # still resumes correctly below.
            print(f"[foresign] WARNING: optimizer state incompatible ({e}); "
                  f"starting optimizer fresh, keeping step count", flush=True)
            opt_resumed = False
        start = ck["step"]
        print(f"[foresign] resumed at {start} (optimizer resumed={opt_resumed}) "
              f"missing keys: {miss.missing_keys}", flush=True)
    elif os.environ.get("FS_INIT_CKPT"):
        # WEIGHT-ONLY warm start from a checkpoint outside this run's output_dir --
        # fresh optimizer, step counter restarts at 0 for THIS run's own short
        # schedule (used for a quick add-a-loss-term smoke test on an already-
        # pretrained encoder, not a literal resume of the original run).
        init_path = os.environ["FS_INIT_CKPT"]
        ck = torch.load(init_path, map_location="cpu", weights_only=False)
        # strict=False: newer modules added after this checkpoint was saved (e.g. the
        # lang_proj head) are simply left at their fresh init; everything the checkpoint
        # DOES contain is loaded. Report any mismatch so an unexpected miss is visible.
        miss = model.load_state_dict(ck["model"], strict=False)
        newkeys = [k for k in miss.missing_keys]
        print(f"[foresign] weight-only warm start from {init_path} "
              f"(was step {ck.get('step','?')}), fresh optimizer, this run starts at 0; "
              f"fresh-init modules not in ckpt: {newkeys}", flush=True)
        if miss.unexpected_keys:
            print(f"[foresign] WARNING unexpected keys in ckpt: {miss.unexpected_keys}",
                  flush=True)

    print(f"[foresign] S-JEPA d={cfg['d_model']} depth={cfg['depth']} "
          f"masking={cfg.get('masking','m3')} "
          f"params={sum(p.numel() for p in params)/1e6:.1f}M datasets={len(datasets)} "
          f"batch={cfg['batch_size']} steps {start}..{total} out={cfg['output_dir']}", flush=True)

    it = iter(dl)
    t0 = time.time()
    collapsed = 0
    collapse_hit = False
    lang_hist = None            # running tally of per-language batch composition (diag)
    model.train()
    from foresign.inductive import provenance
    prov = provenance(cfg)
    from foresign.build import speed_flag, speed_flags
    fast = speed_flag(cfg, "fast_stats")
    prov["speed_flags"] = speed_flags(cfg)
    prov["compile"] = use_compile
    print(f"[foresign] speed flags {prov['speed_flags']}", flush=True)
    if fast:
        col_run = torch.zeros((), device=dev)      # GPU-side consecutive-collapse counter
        col_max = torch.zeros((), device=dev)
    if canon:
        prov["aug_rot_deg"] = {"yaw": ds.aug_yaw, "pitch": ds.aug_pitch}
        prov["canon_versions"] = sorted(set(ds.versions.values()))
        prov["canon_datasets"] = list(ds.names)
    for step in range(start, total):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        pose, nv, lang, dsid = next(it)
        pose = pose.to(dev, non_blocking=True); nv = nv.to(dev)
        lang = lang.to(dev, non_blocking=True); dsid = dsid.to(dev, non_blocking=True)
        if lang_weight > 0:                     # tally batch language diversity (diagnostic)
            u = torch.bincount(lang[lang >= 0], minlength=32).float()
            lang_hist = u if lang_hist is None else lang_hist + u
        valid = time_valid_mask(nv, clip_len, tp, V, dev)
        cond = cfg.get("masking", "m3")
        motion = None
        bt = None
        lam_b = float(cfg.get("lambda_bound", 0.0))
        seq_jepa = False
        if cond == "dseqt":                          # DSeq-T-JEPA sequential-discriminative
            from foresign.masking import temporal_boundary_target
            bt, _ = temporal_boundary_target(pose, tp, layout=layout)   # (B,TP) warmup prior
            # seq path builds its own per-step masks internally; pass placeholder
            mask = torch.zeros(pose.size(0), clip_len // tp, V, dtype=torch.bool, device=dev)
            seq_jepa = True
        elif cond in ("boundary", "bajepa"):         # BA-JEPA boundary-aware masking
            from foresign.masking import boundary_mask
            mask, bt = boundary_mask(pose, tp, V, ratio=cfg.get("mask_ratio", 0.7),
                                     valid=valid, thr_q=cfg.get("boundary_q", 0.6),
                                     layout=layout)
        else:
            if "saliency" in cond:                   # S-JEPA motion-aware arm
                from foresign.masking import group_motion_energy
                motion = group_motion_energy(pose, tp, model.tokenizer.group_idx, layout=layout)
            mask = build_mask(cond, pose.size(0), clip_len // tp, V, dev,
                              valid=valid, cfg=cfg, motion=motion)
            if lam_b > 0 and bt is None:            # MS: boundary target from conf-intact pose
                from foresign.masking import temporal_boundary_target
                bt, _ = temporal_boundary_target(pose, tp, layout=layout)
        # conf-as-mask: zero the confidence channel fed to the ENCODER (it is the #1
        # dataset-fingerprint driver, measured safe to drop for downstream). bt/motion
        # above are computed from the conf-INTACT pose, so the boundary/saliency signal
        # is unaffected — only the encoder input loses the fingerprint-leaking conf value.
        if cfg.get("drop_conf"):
            B_, T_, D_ = pose.shape
            pv = pose.view(B_, T_, D_ // 3, 3).clone()
            pv[..., 2] = 0.0
            pose = pv.view(B_, T_, D_)
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if seq_jepa:
                warm_frac = float(cfg.get("sal_warmup_frac", 0.3))
                sal_lambda = min(1.0, step / max(1, int(warm_frac * total))) \
                    * float(cfg.get("sal_lambda", 1.0))
                loss, stats = model(pose, mask, valid, vreg=cfg.get("vreg", 0.0),
                                    boundary_target=bt, lambda_bound=lam_b,
                                    seq_jepa=True, n_seg=int(cfg.get("n_seg", 5)),
                                    beta=float(cfg.get("beta", 0.3)),
                                    tau=float(cfg.get("tau", 0.1)),
                                    sal_lambda=sal_lambda,
                                    lang_id=lang, lang_weight=lang_weight,
                                    lang_temp=lang_temp, lang_proj=lang_proj,
                                    ds_id=dsid, lang_center=lang_center)
            else:
                loss, stats = model(pose, mask, valid, vreg=cfg.get("vreg", 0.0),
                                    boundary_target=bt, lambda_bound=lam_b,
                                    alpha_pred=float(cfg.get("alpha_pred", 1.0)),
                                    beta=float(cfg.get("beta", 0.0)),
                                    tau=float(cfg.get("tau", 0.1)),
                                    fast_stats=fast,
                                    want_stats=(not fast) or step % cfg["log_every"] == 0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        opt.step()
        ema_update(model, ema_momentum(step, total, cfg["ema_base"]), cfg)

        # collapse guard
        if fast and "feat_std_t" in stats:
            # same counter on the GPU; checked (host sync) on log steps only, so an abort can
            # come up to log_every-1 steps after the patience is first reached
            low = (stats["feat_std_t"] < cfg.get("collapse_std", 0.02)).float()
            col_run = (col_run + 1) * low
            col_max = torch.maximum(col_max, col_run)
            if step % cfg["log_every"] == 0 or step == total - 1:
                collapsed = int(col_max.item())
                col_max = col_run.clone()
        elif stats.get("feat_std", 1.0) < cfg.get("collapse_std", 0.02):
            collapsed += 1
        else:
            collapsed = 0
        if step % cfg["log_every"] == 0:
            seqlog = ""
            if seq_jepa:
                lhist = ""
                if lang_hist is not None:
                    nz = (lang_hist > 0).sum().item()
                    top = lang_hist.max().item() / (lang_hist.sum().item() + 1e-9)
                    lhist = f"langs_seen={nz} top_frac={top:.2f} "
                seqlog = (f"l_pred {stats.get('l_pred', 0):.4f} l_disc "
                          f"{stats.get('l_disc', 0):.4f} l_lang "
                          f"{stats.get('l_lang', 0):.4f} (n_known="
                          f"{stats.get('n_lang_known', 0)}) l_content "
                          f"{stats.get('l_content', 0):.4f} (w="
                          f"{stats.get('content_weight', 0):.3f}) {lhist}selfsim_off "
                          f"{stats.get('selfsim_offdiag', 0):.3f} | ")
            else:
                seqlog = (f"l_pred {stats.get('l_pred', 0):.4f} l_nce "
                          f"{stats.get('l_nce', 0):.4f} l_bound "
                          f"{stats.get('l_bound', 0):.4f} | ")
            wpplog = ""
            print(f"step {step:6d} | loss {loss.item():.4f} | {seqlog}{wpplog}eff_rank "
                  f"{stats.get('eff_rank', float('nan')):.1f} feat_std "
                  f"{stats.get('feat_std', 0):.3f} offdiag {stats.get('offdiag_corr', 0):.3f} "
                  f"pred_var {stats.get('pred_var', 0):.3f} | lr {lr_at(step):.2e} | "
                  f"{(step+1-start)/(time.time()-t0):.2f} it/s", flush=True)
        if collapsed >= cfg.get("collapse_patience", 200):
            collapse_hit = True
            break
        if step > 0 and step % cfg["ckpt_every"] == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "cfg": cfg, "provenance": prov}, last)
            torch.save({"model": model.state_dict(), "step": step, "cfg": cfg, "provenance": prov},
                       os.path.join(outdir, f"step_{step}.pt"))
            print(f"  [ckpt] step {step}", flush=True)

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": step + 1, "cfg": cfg, "provenance": prov}, last)
    print(f"[foresign] done at step {step+1}", flush=True)
    if collapse_hit:
        print(collapse_message(step, cfg.get("collapse_std", 0.02), collapsed), flush=True)
        sys.exit(COLLAPSE_EXIT_CODE)


if __name__ == "__main__":
    main()
