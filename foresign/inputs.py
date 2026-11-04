"""Model input layouts.

resona77 (Phase 0/1a): flat (T, 234) = 78 joints x (x, y, conf); all 3 channels are projected,
    conf (channel 2) is the token-validity gate, motion uses (x, y) weighted by conf.
c77 (Phase 2, C77-3D canon data): flat (T, 77*(in_dim+1)) = 77 joints x (coords..., mask), the
    first `in_dim` of (x, y, z) zeroed where the joint is absent, then the 0/1 joint mask. Only
    the coordinates are projected; the mask is the gate and the motion weight.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RESONA77 = "resona77"
C77 = "c77"
N_C77 = 77


@dataclass(frozen=True)
class Layout:
    n_joints: int
    chans: int          # channels per joint in the flat input
    proj: int           # leading channels fed to the token projection
    gate: int           # channel index used as validity gate / motion weight
    motion: int         # leading coordinate channels used for motion energy
    fmt: str = RESONA77

    @property
    def flat(self) -> int:
        return self.n_joints * self.chans


OLD = Layout(78, 3, 3, 2, 2, RESONA77)


def layout_for(cfg: dict) -> Layout:
    fmt = cfg.get("input_format") or RESONA77
    if fmt == RESONA77:
        return OLD
    if fmt != C77:
        raise ValueError(f"unknown input_format {fmt!r}")
    if cfg.get("drop_conf"):
        raise ValueError("drop_conf has no meaning for c77 input (no conf channel); set it false")
    d = int(cfg.get("in_dim", 3))
    if d not in (2, 3):
        raise ValueError(f"in_dim must be 2 (x,y) or 3 (x,y,z), got {d}")
    return Layout(N_C77, d + 1, d, d, d, C77)


def pack_c77(x: np.ndarray, mask: np.ndarray, in_dim: int = 3) -> np.ndarray:
    """x (T,77,3) + mask (T,77) -> (T, 77*(in_dim+1)) float32; absent joints are zero."""
    m = mask.astype(np.float32)
    out = np.empty((x.shape[0], x.shape[1], in_dim + 1), np.float32)
    out[..., :in_dim] = x[..., :in_dim].astype(np.float32) * m[..., None]
    out[..., in_dim] = m
    return out.reshape(x.shape[0], -1)
