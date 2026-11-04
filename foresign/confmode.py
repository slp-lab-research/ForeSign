"""Confidence-channel handling shared by all evaluators: follow the checkpoint's
training config (cfg['drop_conf']) unless FS_DROP_CONF=0/1 explicitly overrides."""
from __future__ import annotations

import os

import torch


def resolve_drop_conf(cfg: dict) -> bool:
    env = os.environ.get("FS_DROP_CONF")
    if (cfg.get("input_format") or "resona77") == "c77":     # no conf channel (coords + mask)
        if env == "1":
            raise ValueError("FS_DROP_CONF=1 is invalid for a c77 checkpoint (channel 2 is z)")
        return False
    if env in ("0", "1"):
        return env == "1"
    return bool(cfg.get("drop_conf", False))


def zero_conf(pose: torch.Tensor) -> torch.Tensor:
    out = pose.clone()
    out.view(*pose.shape[:-1], pose.shape[-1] // 3, 3)[..., 2] = 0.0
    return out
