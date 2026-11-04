"""ForeSign tokenizer — linear part-aware patchify (design §5.1 default).

(B, T=128, 234) pose  →  (B, TP=32, 18, d) tokens

Each token = Linear projection of one spatial group's (n_joints × l × 3) values
over l=4 consecutive frames. Per-group linear (groups differ in size). Adds a
learnable spatial (group) embedding and a learnable temporal embedding, plus a
part-validity vector gated by per-token mean confidence (design §4.2: padded /
low-conf tokens get a learned "absent" signal rather than silent zeros).

GCN stem is an ABLATION (design §5.1) — default is this linear patchify, which
also avoids conv-receptive-field leakage into JEPA masking.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from foresign.groups import GROUPS, GROUP_NAMES, N_GROUPS, shuffled_groups
from foresign.inputs import OLD, Layout

FLAT = 234
N_JOINTS = 78          # 234 / 3 (77 real + 1 tail joint; groups use 0..76)


class PartLinearTokenizer(nn.Module):
    def __init__(self, d_model: int = 384, t_patch: int = 4, max_tp: int = 64,
                 conf_gate: bool = True, group_layout: str = "anatomical",
                 layout: Layout | None = None):
        super().__init__()
        # input layout (foresign/inputs.py): resona77 (x,y,conf; default) or c77 (coords+mask)
        self.layout = layout or OLD
        self.t_patch = t_patch
        self.d_model = d_model
        self.conf_gate = conf_gate
        # per-group linear: (n_joints_g * t_patch * proj_channels) → d_model
        if group_layout == "anatomical":
            self.group_idx = [GROUPS[g] for g in GROUP_NAMES]
        elif group_layout == "shuffled":
            self.group_idx = shuffled_groups(0)
        else:
            raise ValueError(f"unknown group_layout {group_layout!r}")
        self.proj = nn.ModuleList([
            nn.Linear(len(js) * t_patch * self.layout.proj, d_model) for js in self.group_idx
        ])
        self.spatial_embed = nn.Parameter(torch.zeros(1, 1, N_GROUPS, d_model))
        self.temp_embed = nn.Parameter(torch.zeros(1, max_tp, 1, d_model))
        # learned "low-confidence / absent" vector added per token, scaled by how
        # invalid the token is (1 - mean_conf). Lets the model represent absence.
        self.absent = nn.Parameter(torch.zeros(1, 1, N_GROUPS, d_model))
        nn.init.trunc_normal_(self.spatial_embed, std=0.02)
        nn.init.trunc_normal_(self.temp_embed, std=0.02)
        nn.init.trunc_normal_(self.absent, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, layout.flat) (234 for resona77) → tokens (B, TP, 18, d)."""
        B, T, C = x.shape
        L = self.layout
        assert C == L.flat and T % self.t_patch == 0, (T, C, L.flat)
        TP = T // self.t_patch
        # patchify time: (B, TP, t_patch, J, chans)
        vp = x.reshape(B, TP, self.t_patch, L.n_joints, L.chans)
        toks, gates = [], []
        for g, (js, lin) in enumerate(zip(self.group_idx, self.proj)):
            sel = vp[:, :, :, js, :]                            # (B,TP,t_patch,n_g,chans)
            flat = sel[..., :L.proj].reshape(B, TP, -1)         # (B,TP,n_g*t_patch*proj)
            toks.append(lin(flat))                             # (B,TP,d)
            conf = sel[..., L.gate].reshape(B, TP, -1).mean(dim=-1)  # (B,TP) mean conf/mask
            gates.append(conf)
        h = torch.stack(toks, dim=2)                           # (B,TP,18,d)
        h = h + self.spatial_embed + self.temp_embed[:, :TP]
        if self.conf_gate:
            gate = torch.stack(gates, dim=2).clamp(0, 1)       # (B,TP,18)
            h = h + (1.0 - gate).unsqueeze(-1) * self.absent
        return h

    @torch.no_grad()
    def token_conf(self, x: torch.Tensor) -> torch.Tensor:
        """(B, TP, 18) mean confidence per token — for masking-valid stats &
        target exclusion (design §4.2: token mean conf < 0.2 → exclude target)."""
        B, T, _ = x.shape
        L = self.layout
        TP = T // self.t_patch
        v = x.reshape(B, TP, self.t_patch, L.n_joints, L.chans)
        out = []
        for js in self.group_idx:
            out.append(v[:, :, :, js, L.gate].reshape(B, TP, -1).mean(dim=-1))
        return torch.stack(out, dim=2)
