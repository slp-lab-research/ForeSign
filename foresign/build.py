"""Single constructor for ForeSign models so every config key reaches train and eval."""
from __future__ import annotations

import os


def build_model(cfg: dict):
    from foresign.inputs import layout_for
    from foresign.jepa import ForeSignJEPA
    m = ForeSignJEPA(
        d_model=cfg["d_model"], depth=cfg["depth"], n_heads=cfg["n_heads"],
        pred_dim=cfg["pred_dim"], pred_depth=cfg["pred_depth"],
        pred_heads=cfg["pred_heads"], t_patch=cfg["t_patch"],
        ema_base=cfg["ema_base"], n_seg=cfg.get("n_seg", 5),
        group_layout=cfg.get("group_layout", "anatomical"), layout=layout_for(cfg))
    # speed knobs (identical maths): number of checkpointed layers (None = all, original)
    m.context.ckpt_layers = cfg.get("ckpt_layers")
    m.predictor.body.ckpt_layers = cfg.get("pred_ckpt_layers")
    if speed_flag(cfg, "fast_attn"):
        from foresign.fastops import enable           # efficient SDPA for temporal attention
        enable(m)
    if speed_flag(cfg, "fast_norm"):
        from foresign.fastops import enable_norm      # fused RMSNorm
        enable_norm(m)
    return m


SPEED_KEYS = ("fast_attn", "fast_norm", "fast_stats", "foreach_ema", "fused_adamw")


def speed_flag(cfg: dict, key: str) -> bool:
    """Speed knobs (same maths up to bf16 kernel rounding; equivalence check:
    tools/equiv_compile.py). Precedence: env FS_FAST_OPS=0/1 (all knobs) > explicit cfg key >
    default: on for c77 (Phase 2), off for resona77 (Phase 0/1a stays bit-reproducible)."""
    env = os.environ.get("FS_FAST_OPS")
    if env in ("0", "1"):
        return env == "1"
    if cfg.get(key) is not None:
        return bool(cfg[key])
    from foresign.inputs import C77, layout_for
    return layout_for(cfg).fmt == C77


def speed_flags(cfg: dict) -> dict:
    return {k: speed_flag(cfg, k) for k in SPEED_KEYS}


def compile_flag(cfg: dict, env: str = "FS_COMPILE") -> bool:
    """torch.compile opt-in: env (FS_COMPILE / FS_FT_COMPILE = 0/1) > cfg `compile` > off."""
    v = os.environ.get(env)
    if v in ("0", "1"):
        return v == "1"
    return bool(cfg.get("compile", False))


def compile_modules(modules) -> list:
    """In-place nn.Module.compile (mode "default", dynamic=False): forward is compiled but the
    module object, parameters and state_dict keys stay the same, so checkpoints keep their
    format. Returns the names compiled."""
    import torch
    done = []
    for name, m in modules:
        m.compile(mode="default", dynamic=False)
        done.append(name)
    return done


def compile_jepa(model) -> list:
    """Pretraining: tokenizer, context encoder, EMA target (tokenizer + encoder), predictor."""
    return compile_modules([("tokenizer", model.tokenizer), ("context", model.context),
                            ("target_tok", model.target_tok), ("target_ctx", model.target_ctx),
                            ("predictor", model.predictor)])
