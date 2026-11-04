"""Encoder adapter: a uniform frozen-encoder interface for the evaluation suite.

    pool(pose, n_valid)        -> (B, d)      padding-masked mean clip embedding
    time_tokens(pose, n_valid) -> (B, TP, d)  joint-pooled per-time-patch tokens, (B, TP) mask

pose: (B, T, flat) packed input (C77-3D: coordinates + joint mask, foresign.inputs.pack_c77),
zero-padded to clip_len; n_valid: (B,) valid frame counts. Padding exclusion is handled here.
"""
from __future__ import annotations

import os

import torch


class ForeSignAdapter:
    """ForeSign-JEPA downstream encoder = EMA TARGET (tokenizer + context).
    S-JEPA: downstream uses the EMA encoder. Same pool/tokens interface as
    V4Adapter so the entire eval suite (islr/diagnostics/phonology) runs on a
    ForeSign checkpoint with zero probe-code changes.

    For the phonology probe's hand-token readout (R8_BASELINE §5b follow-up),
    `hand_tokens()` returns only the 12 hand-group tokens — the part-structured
    tokenization makes this clean (no pooling-away of handshape)."""

    def __init__(self, ckpt: str, device: str = "cuda", which: str = "ema"):
        torch.backends.cuda.enable_cudnn_sdp(False)
        from foresign.build import build_model
        from foresign.groups import HAND_GROUPS, GROUP_INDEX
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        model = build_model(cfg)
        model.load_state_dict(ck["model"], strict=False)   # boundary_head absent in pre-BA ckpts
        # downstream = EMA target (which="ema") or online context (which="online")
        self.tok = (model.target_tok if which == "ema" else model.tokenizer).to(device).eval()
        self.body = (model.target_ctx if which == "ema" else model.context).to(device).eval()
        self.body.grad_checkpoint = False
        self.device = device
        self.t_patch = cfg["t_patch"]
        self.clip_len = cfg["clip_len"]
        self.d_model = cfg["d_model"]
        from foresign.inputs import layout_for
        self.layout = layout_for(cfg)                    # resona77 | c77 (coords + mask)
        self.in_dim = int(cfg.get("in_dim", 3))
        self._hand_cols = [GROUP_INDEX[g] for g in HAND_GROUPS]
        td = int(os.environ.get("FS_TAP_DEPTH", "0"))
        self.tap_depth = td if 0 < td < self.body.depth else None
        print(f"[adapter] tap_depth={self.tap_depth or 'final'}", flush=True)
        from foresign.confmode import resolve_drop_conf
        self.drop_conf = resolve_drop_conf(cfg)
        print(f"[adapter] drop_conf={self.drop_conf}", flush=True)

    def _body(self, h):
        if self.tap_depth is None:
            return self.body(h)
        N, TP, V, d = h.shape
        x = h.reshape(N * TP, V, d)
        body = self.body
        for li, (st, ts, att) in enumerate(zip(body.blocks_st, body.blocks_ts, body.ts_attn), 1):
            x = body._layer_forward(x, TP, st, ts, att)
            if li == self.tap_depth:
                break
        return x.reshape(N, TP, V, d)

    @torch.no_grad()
    def _grid(self, pose, n_valid):
        if self.drop_conf:
            from foresign.confmode import zero_conf
            pose = zero_conf(pose)
        h = self._body(self.tok(pose))                            # (B,TP,18,d)
        tp = pose.size(1) // self.t_patch
        vtp = torch.div(n_valid + self.t_patch - 1, self.t_patch,
                        rounding_mode="floor").clamp(1, tp)
        tmask = torch.arange(tp, device=pose.device)[None] < vtp[:, None]
        return h, tmask                                          # (B,TP,18,d),(B,TP)

    @torch.no_grad()
    def pool(self, pose, n_valid):
        h, tm = self._grid(pose, n_valid)
        m = tm[:, :, None, None].to(h.dtype)
        return (h * m).sum((1, 2)) / (m.sum((1, 2)) * h.size(2)).clamp(min=1.0)

    @torch.no_grad()
    def time_tokens(self, pose, n_valid):
        h, tm = self._grid(pose, n_valid)
        return h.mean(dim=2), tm                                 # (B,TP,d)

    @torch.no_grad()
    def hand_pool(self, pose, n_valid):
        """Mean over the 12 HAND tokens only — phonology-faithful readout that
        does NOT pool handshape away (fixes the R8 §5b methodological flaw)."""
        h, tm = self._grid(pose, n_valid)
        hh = h[:, :, self._hand_cols, :]
        m = tm[:, :, None, None].to(h.dtype)
        return (hh * m).sum((1, 2)) / (m.sum((1, 2)) * hh.size(2)).clamp(min=1.0)
