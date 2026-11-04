"""ForeSign-JEPA model (design §5.2): continuous latent prediction with an EMA
target encoder. Reuses the v4 DSTformer dual-stream body (foresign/_vendor/resona/models/v4_blocks.py).

Flow per step:
  tokens_full = tokenizer(x)                       # (B,TP,18,d), shared weights
  with no_grad: z_tgt = target_encoder(tokens_full)   # EMA weights, full view
  z_ctx = context_encoder(tokens_full, masked→0)      # context = visible only
  z_pred = predictor(z_ctx, mask)                     # predict at masked slots
  loss = smooth_L1( z_pred[mask], stopgrad(LN(z_tgt[mask])) )   # latent target

Design choices (following S-JEPA / V-JEPA):
  - EMA target is ESSENTIAL (S-JEPA: removing it collapses to 1.6%). momentum
    0.999 start → cosine to 1.0.
  - target = full-sequence target-encoder OUTPUT (not input recon).
  - narrow predictor (dim/2) with zero-init mask token.
  - LayerNorm on targets (centering) — variance-preserving, anti-collapse.
  - collapse monitoring: effective rank / per-dim variance / off-diagonal corr,
    logged every step (R2 risk: 234-d pose latent collapse is unverified).
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from foresign.tokenize import PartLinearTokenizer
from foresign.groups import N_GROUPS

from foresign._vendor.resona.models.v4_blocks import DSTformerBody as _DSTformerBody, RMSNorm


class DSTformerBody(_DSTformerBody):
    """DSTformerBody with partial activation checkpointing: `ckpt_layers` = number of leading
    layers that are checkpointed when grad_checkpoint is on (None = all, the original
    behaviour). Checkpointing recomputes with the saved RNG state, so any value gives the same
    maths; fewer checkpointed layers = more memory, less recompute. Same state_dict keys."""

    ckpt_layers = None

    def forward(self, x):
        n = self.depth if self.ckpt_layers is None else int(self.ckpt_layers)
        if not (self.grad_checkpoint and self.training) or n >= self.depth:
            return super().forward(x)
        B, F_, V_, dim = x.shape
        x = x.reshape(B * F_, V_, dim)
        for i, (st, ts, att) in enumerate(zip(self.blocks_st, self.blocks_ts, self.ts_attn)):
            if i < n:
                x = torch.utils.checkpoint.checkpoint(self._layer_forward, x, F_, st, ts, att,
                                                      use_reentrant=False)
            else:
                x = self._layer_forward(x, F_, st, ts, att)
        return self.norm_final(x).reshape(B, F_, V_, dim)


class Predictor(nn.Module):
    """Narrow DSTformer predictor with a zero-init mask token (V-JEPA style)."""

    def __init__(self, d_model, pred_dim, depth, n_heads, max_tp=64, n_seg_max=8):
        super().__init__()
        self.embed = nn.Linear(d_model, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, pred_dim))   # zero-init
        self.spatial_embed = nn.Parameter(torch.zeros(1, 1, N_GROUPS, pred_dim))
        self.temp_embed = nn.Parameter(torch.zeros(1, max_tp, 1, pred_dim))
        # DSeq-T-JEPA autoregressive step token (zero-init: no-op when step=None)
        self.step_embed = nn.Parameter(torch.zeros(1, n_seg_max, 1, pred_dim))
        nn.init.trunc_normal_(self.spatial_embed, std=0.02)
        nn.init.trunc_normal_(self.temp_embed, std=0.02)
        self.body = DSTformerBody(dim=pred_dim, depth=depth, num_heads=n_heads,
                                  ffn_dim=pred_dim * 2, drop_path_rate=0.0,
                                  grad_checkpoint=True)
        self.norm = RMSNorm(pred_dim)
        self.out = nn.Linear(pred_dim, d_model)

    def forward(self, z_ctx, mask, step=None):
        """z_ctx: (B,TP,V,d) context features (masked slots are 0).
        mask: (B,TP,V) True=masked. step: optional int autoregressive step index;
        when given, adds step_embed[step] to the masked/target slots so the predictor
        knows which AR step it is on. Returns prediction (B,TP,V,d)."""
        B, TP, V, _ = z_ctx.shape
        h = self.embed(z_ctx)
        m = self.mask_token.expand(B, TP, V, -1)
        h = torch.where(mask.unsqueeze(-1), m, h)        # put mask token at targets
        h = h + self.spatial_embed + self.temp_embed[:, :TP]
        if step is not None:
            # add the AR step token only at the masked/target slots
            st = self.step_embed[:, step]                # (1,1,pred_dim)
            h = h + mask.unsqueeze(-1).to(h.dtype) * st
        h = self.body(h)
        return self.out(self.norm(h))


class ForeSignJEPA(nn.Module):
    def __init__(self, d_model=384, depth=12, n_heads=6, pred_dim=192,
                 pred_depth=4, pred_heads=6, t_patch=4, ema_base=0.999,
                 n_seg=5, n_seg_max=8, group_layout="anatomical", layout=None):
        super().__init__()
        self.n_seg = n_seg
        self.n_seg_max = n_seg_max
        self.tokenizer = PartLinearTokenizer(d_model=d_model, t_patch=t_patch,
                                             group_layout=group_layout, layout=layout)
        self.layout = self.tokenizer.layout
        self.context = DSTformerBody(dim=d_model, depth=depth, num_heads=n_heads,
                                     ffn_dim=d_model * 4, drop_path_rate=0.1,
                                     grad_checkpoint=True)
        # EMA target = frozen deep-copy of (tokenizer + context); updated by momentum
        self.target_tok = copy.deepcopy(self.tokenizer)
        self.target_ctx = copy.deepcopy(self.context)
        for p in self.target_tok.parameters():
            p.requires_grad_(False)
        for p in self.target_ctx.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor(d_model, pred_dim, pred_depth, pred_heads,
                                   n_seg_max=n_seg_max)
        # BA-JEPA: per-temporal-patch boundary-prediction head (self-supervised
        # gloss-segment boundary pretext; instils frame-level alignment structure)
        self.boundary_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        # language-SupCon projection head (SimCLR-style: apply the contrastive loss on a
        # projection, keeping the backbone features general). Only used when lang_weight>0
        # and lang_proj=True; cheap (only trained by the language term).
        self.lang_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 128))
        # running per-dataset mean of pooled context features (EMA), for the
        # "centered language SupCon" variant: subtract the dataset's mean before the
        # language contrastive term so it must separate languages BEYOND the
        # per-dataset offset (= the fingerprint) -- the SSL analog of the frozen-head
        # champion's per-dataset centering. Buffer (persists in state_dict).
        self.register_buffer("ds_mean", torch.zeros(32, d_model))
        self.register_buffer("ds_count", torch.zeros(32))
        # Buffers / parameter of earlier auxiliary objectives (content SupCon, pairwise match)
        # that are not part of this release; kept so that checkpoints keep the same state_dict
        # keys and parameter list (they receive no gradient in any released config).
        self.register_buffer("content_ds_mean", torch.zeros(2, d_model))
        self.register_buffer("content_ds_count", torch.zeros(2))
        self.wpp_match_scale = nn.Parameter(torch.tensor(10.0))
        self.ema_base = ema_base
        self.t_patch = t_patch
        self.d_model = d_model

    @torch.no_grad()
    def update_ema(self, m):
        for q, k in zip(self.tokenizer.parameters(), self.target_tok.parameters()):
            k.mul_(m).add_(q.detach(), alpha=1 - m)
        for q, k in zip(self.context.parameters(), self.target_ctx.parameters()):
            k.mul_(m).add_(q.detach(), alpha=1 - m)

    def forward(self, x, mask, valid=None, vreg=0.0, boundary_target=None,
                lambda_bound=0.0, seq_jepa=False, n_seg=5, beta=0.0, tau=0.1,
                sal_lambda=1.0, lang_id=None, lang_weight=0.0, lang_temp=0.125, lang_proj=False, ds_id=None, lang_center=False,
                alpha_pred=1.0, fast_stats=False, want_stats=True):
        """x: (B,T,234), mask: (B,TP,V) True=predict-target, valid: (B,TP,V).
        vreg>0 adds a VICReg-style variance+covariance regularizer on the context
        features — the CAUSAL-TEST knob: directly forces higher eff_rank. If it
        rescues a collapsing condition's downstream → eff_rank→downstream is causal.

        seq_jepa=True switches to the DSeq-T-JEPA objective (design §2): temporal
        saliency segmentation + autoregressive segment prediction + inter-timestep
        contrastive anti-smoothing. The passed-in `mask` is IGNORED in this path
        (per-step masks are built internally)."""
        # target: full view, EMA weights, no grad
        with torch.no_grad():
            ht = self.target_tok(x)
            z_tgt = self.target_ctx(ht)                  # (B,TP,V,d)
            z_tgt = F.layer_norm(z_tgt, (z_tgt.size(-1),))   # centering (anti-collapse)
        if seq_jepa:
            return self._forward_seq(x, z_tgt, valid=valid, boundary_target=boundary_target,
                                     n_seg=n_seg, beta=beta, tau=tau, sal_lambda=sal_lambda,
                                     lambda_bound=lambda_bound, vreg=vreg,
                                     lang_id=lang_id, lang_weight=lang_weight,
                                     lang_temp=lang_temp, lang_proj=lang_proj,
                                     ds_id=ds_id, lang_center=lang_center)
        if alpha_pred == 0 and vreg == 0 and (boundary_target is None or lambda_bound == 0):
            # NCE-only: the masked-context path contributes nothing -> skip it entirely
            l_nce = self._temporal_nce(x, z_tgt, valid, tau) if beta > 0                 else z_tgt.new_zeros((), requires_grad=False)
            loss = beta * l_nce
            with torch.no_grad():
                tv = z_tgt[valid] if valid is not None else z_tgt.reshape(-1, z_tgt.size(-1))
                if fast_stats and not want_stats:
                    return loss, {"feat_std_t": _feat_std(tv)}
                stats = collapse_stats(tv) if tv.size(0) >= 2 else {}
                if fast_stats:
                    stats["feat_std_t"] = _feat_std(tv)
                stats.update({"n_target": 0, "pred_var": 0.0, "l_pred": 0.0,
                              "l_nce": float(l_nce), "l_vreg": 0.0, "l_bound": 0.0})
            return loss, stats
        # context: visible only (zero the masked tokens before the body)
        h = self.tokenizer(x)
        h = h * (~mask).unsqueeze(-1).to(h.dtype)
        z_ctx = self.context(h)
        z_ctx = z_ctx * (~mask).unsqueeze(-1).to(z_ctx.dtype)
        sel = mask & valid if valid is not None else mask
        loss = z_ctx.sum() * 0.0
        l_pred = z_ctx.new_zeros(())
        p = t = None
        if alpha_pred > 0 and sel.sum() > 0:
            z_pred = self.predictor(z_ctx, mask)             # (B,TP,V,d)
            p = z_pred[sel]
            t = z_tgt[sel]
            l_pred = F.smooth_l1_loss(p, t)
            loss = loss + alpha_pred * l_pred
        l_vreg = z_ctx.new_zeros(())
        if vreg > 0:
            # VICReg on context features at VALID tokens (the eff_rank we measure)
            zc = z_ctx[valid] if valid is not None else z_ctx.reshape(-1, z_ctx.size(-1))
            zc = zc - zc.mean(0, keepdim=True)
            std = (zc.var(0) + 1e-4).sqrt()
            var_term = F.relu(1.0 - std).mean()           # push per-dim std -> 1
            d = zc.size(1)
            cov = (zc.T @ zc) / (zc.size(0) - 1)
            cov_term = (cov.pow(2).sum() - cov.diag().pow(2).sum()) / d
            l_vreg = var_term + cov_term
            loss = loss + vreg * l_vreg
        # boundary-prediction aux loss (BA-JEPA / ForeSign-MS): per-patch boundary score
        # from context features at patches with any visible token, supervised by the
        # unsupervised motion change-point target.
        l_bound = z_ctx.new_zeros(())
        if boundary_target is not None and lambda_bound > 0:
            zc_patch = z_ctx.mean(2)                     # (B,TP,d) mean over V
            blogit = self.boundary_head(zc_patch).squeeze(-1)   # (B,TP)
            vis = (~mask).any(-1)                        # (B,TP) patches with any visible token
            if valid is not None:
                vis = vis & valid.any(-1)                # never score padded patches
            if vis.any():
                l_bound = F.binary_cross_entropy_with_logits(
                    blogit[vis], boundary_target[vis].to(blogit.dtype))
                loss = loss + lambda_bound * l_bound
        l_nce = z_ctx.new_zeros(())
        if beta > 0:
            l_nce = self._temporal_nce(x, z_tgt, valid, tau)
            loss = loss + beta * l_nce
        with torch.no_grad():
            tv = z_tgt[valid] if valid is not None else z_tgt.reshape(-1, z_tgt.size(-1))
            mon = t if t is not None else tv             # monitor TARGET space
            if fast_stats and not want_stats:            # no host sync, no SVD (non-log steps)
                return loss, {"feat_std_t": _feat_std(mon)}
            stats = collapse_stats(mon) if mon.size(0) >= 2 else {}
            if fast_stats:
                stats["feat_std_t"] = _feat_std(mon)
            stats.update({"n_target": int(sel.sum()),
                          "pred_var": p.var(0).mean().item() if p is not None else 0.0,
                          "l_pred": float(l_pred), "l_nce": float(l_nce),
                          "l_vreg": float(l_vreg), "l_bound": float(l_bound)})
        return loss, stats

    def _temporal_nce(self, x, z_tgt, valid, tau):
        """Intra-clip temporal InfoNCE (uniform time weights): the all-visible online
        context at time t (final layer, joint-mean) must identify the stop-grad EMA
        target at the same t among all valid t' of the clip."""
        B, TP, V, _ = z_tgt.shape
        full = torch.zeros(B, TP, V, dtype=torch.bool, device=x.device)
        c = self._encode_ctx(x, full).mean(2)                   # (B,TP,d)
        q = z_tgt.mean(2).detach()                              # (B,TP,d)
        tvalid = valid.any(-1) if valid is not None else             torch.ones(B, TP, dtype=torch.bool, device=x.device)
        # similarity / softmax in fp32 (autocast would otherwise run the bmm in bf16)
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = torch.bmm(F.normalize(c.float(), dim=-1),
                               F.normalize(q.float(), dim=-1).transpose(1, 2)) / tau
            logits = logits.masked_fill(~tvalid.unsqueeze(1), torch.finfo(logits.dtype).min)
            diag = torch.log_softmax(logits, dim=-1).diagonal(dim1=1, dim2=2)   # (B,TP)
            w = tvalid.float()
            return -(w * diag).sum() / (w.sum() + 1e-6)

    def _encode_ctx(self, x, mask):
        """Context encode with masked temporal slots zeroed (mirrors the parallel
        path). mask: (B,TP,V) True=masked/predict. Returns z_ctx (B,TP,V,d)."""
        h = self.tokenizer(x)
        h = h * (~mask).unsqueeze(-1).to(h.dtype)
        z_ctx = self.context(h)
        z_ctx = z_ctx * (~mask).unsqueeze(-1).to(z_ctx.dtype)
        return z_ctx

    def _forward_seq(self, x, z_tgt, valid=None, boundary_target=None, n_seg=5,
                     beta=0.0, tau=0.1, sal_lambda=1.0, lambda_bound=0.0, vreg=0.0,
                     lang_id=None, lang_weight=0.0, lang_temp=0.125, lang_proj=False,
                     ds_id=None, lang_center=False):
        """DSeq-T-JEPA objective (design §2). z_tgt: (B,TP,V,d) stopgrad LN target."""
        from foresign.masking import temporal_saliency_segments
        B, TP, V, d = z_tgt.shape
        device = z_tgt.device
        if boundary_target is None:
            boundary_target = z_tgt.new_zeros(B, TP)
        # (a) per-temporal-patch target latent (mean over joints)
        z_tgt_patch = z_tgt.mean(2)                              # (B,TP,d)
        # (b) saliency segmentation (ordered desc by mean saliency)
        segs, s_tilde = temporal_saliency_segments(
            z_tgt_patch, boundary_target, n_seg=n_seg, lam=sal_lambda)

        # (c) L_pred: autoregressive prefix prediction, k=1..n_seg-1
        l_pred = z_tgt.new_zeros(())
        n_steps = 0
        tvalid = valid.any(-1) if valid is not None else torch.ones(B, TP, dtype=torch.bool, device=device)
        for k in range(1, n_seg):
            # mask True at target = segment R_{k+1} (rank index k, 0-based);
            # visible = R_1..R_k. Samples with < k+1 segments contribute nothing.
            mask_k = torch.zeros(B, TP, V, dtype=torch.bool, device=device)
            for b in range(B):
                if len(segs[b]) > k:
                    s0, e0 = segs[b][k]
                    mask_k[b, s0:e0, :] = True
            if valid is not None:
                mask_k = mask_k & valid
            sel = mask_k
            if sel.sum() == 0:
                continue
            z_ctx_k = self._encode_ctx(x, mask_k)
            z_pred_k = self.predictor(z_ctx_k, mask_k, step=min(k, self.n_seg_max - 1))
            l_pred = l_pred + F.smooth_l1_loss(z_pred_k[sel], z_tgt[sel])
            n_steps += 1
        if n_steps > 0:
            l_pred = l_pred / n_steps

        # (d) L_disc: inter-timestep InfoNCE on full-context per-patch features
        full_mask = torch.zeros(B, TP, V, dtype=torch.bool, device=device)
        z_ctx_full = self._encode_ctx(x, full_mask)             # all-visible context
        c = z_ctx_full.mean(2)                                  # (B,TP,d)

        # (d.5) L_lang: SSL-TIME language SupCon (optional; off in all Phase-2 configs: same-language pull /
        # different-language repel baked into the encoder itself, not a post-hoc head).
        # Uses foresign/supcon.py::supcon
        # on a clip-level pooled vector of the ONLINE (gradient-flowing) context path --
        # unlike the post-hoc round-1..7 work, this backpropagates into the encoder.
        l_lang = z_tgt.new_zeros(())
        known = None            # pre-existing bug fix: only defined below when the language
                                 # term actually runs; the stats line further down must not
                                 # reference it unconditionally (was UnboundLocalError whenever
                                 # lang_id is passed but lang_weight==0, e.g. any run that
                                 # doesn't opt into the SSL-time language term -- caught by the
                                 # foresign_D content-SupCon smoke test, 2026-07-12).
        if lang_id is not None and lang_weight > 0:
            from foresign.supcon import supcon as _lang_supcon
            valid_t_f = tvalid.float().unsqueeze(-1)                     # (B,TP,1)
            c_pool = (c.float() * valid_t_f).sum(1) / valid_t_f.sum(1).clamp(min=1.0)  # (B,d)
            if lang_center and ds_id is not None:
                # subtract the running per-dataset mean (detached, PRE-update) so the
                # contrastive term must separate languages beyond the per-dataset offset
                # (the fingerprint). Then EMA-update the running means from this batch.
                nds = self.ds_mean.size(0)
                dsc = ds_id.clamp(0, nds - 1)
                c_pool = c_pool - self.ds_mean[dsc].detach()
                with torch.no_grad():
                    cd = c.float().detach()
                    cd_pool = (cd * valid_t_f).sum(1) / valid_t_f.sum(1).clamp(min=1.0)
                    for d in torch.unique(dsc):
                        m = dsc == d
                        bm = cd_pool[m].mean(0)
                        if self.ds_count[d] < 1:
                            self.ds_mean[d] = bm
                        else:
                            self.ds_mean[d].mul_(0.98).add_(bm, alpha=0.02)
                        self.ds_count[d] += 1
            known = lang_id >= 0                                          # drop UNKNOWN_LANG_ID
            if known.sum() > 1:
                feat = c_pool[known]
                if lang_proj:               # SimCLR-style projection head (candidate better method)
                    feat = self.lang_proj(feat)
                # supcon() expects L2-normalized input (matches the projection-head convention:
                # own F.normalize-before-supcon convention) -- NOT normalizing here
                # would silently change the effective temperature/similarity scale.
                c_norm = F.normalize(feat, dim=-1)
                l_lang = _lang_supcon(c_norm, lang_id[known],
                                      torch.zeros_like(lang_id[known]), lang_temp,
                                      cross_only=False)

        q = z_tgt_patch.detach()                                # (B,TP,d) stopgrad
        cn = F.normalize(c.float(), dim=-1)
        qn = F.normalize(q.float(), dim=-1)
        logits = torch.bmm(cn, qn.transpose(1, 2)) / tau        # (B,TP,TP) sim(c_t,q_t')
        # mask out invalid time-steps (both as anchors and as negatives)
        valid_t = tvalid.float()                                # (B,TP)
        neg_inf = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~tvalid.unsqueeze(1), neg_inf)   # invalid t' columns
        logp = torch.log_softmax(logits, dim=-1)                # (B,TP,TP)
        idx = torch.arange(TP, device=device)
        diag = logp[:, idx, idx]                                # (B,TP) log p(t|t)
        wt = (1.0 + s_tilde) * valid_t                          # saliency weight, valid only
        denom = wt.sum() + 1e-6
        l_disc = -(wt * diag).sum() / denom

        # (e) boundary BCE aux on full-context patch features
        l_bound = z_tgt.new_zeros(())
        if boundary_target is not None and lambda_bound > 0:
            blogit = self.boundary_head(c).squeeze(-1)          # (B,TP)
            if tvalid.any():
                l_bound = F.binary_cross_entropy_with_logits(
                    blogit[tvalid], boundary_target[tvalid].to(blogit.dtype))

        # (f) VICReg on full-context features (kept term)
        l_vreg = z_tgt.new_zeros(())
        if vreg > 0:
            zc = z_ctx_full[valid] if valid is not None else z_ctx_full.reshape(-1, d)
            zc = zc - zc.mean(0, keepdim=True)
            std = (zc.var(0) + 1e-4).sqrt()
            var_term = F.relu(1.0 - std).mean()
            cov = (zc.T @ zc) / (zc.size(0) - 1)
            cov_term = (cov.pow(2).sum() - cov.diag().pow(2).sum()) / d
            l_vreg = var_term + cov_term

        loss = l_pred + beta * l_disc + lambda_bound * l_bound + vreg * l_vreg + lang_weight * l_lang

        with torch.no_grad():
            stats = collapse_stats(z_tgt[tvalid.unsqueeze(-1).expand(-1, -1, V)]
                                   if valid is not None else z_tgt.reshape(-1, d))
            # selfsim_offdiag: mean off-diagonal of normalized c@c^T (anti-smoothing proxy)
            ccn = F.normalize(c.float(), dim=-1)
            sim = torch.bmm(ccn, ccn.transpose(1, 2))           # (B,TP,TP)
            eye = torch.eye(TP, device=device).unsqueeze(0)
            mvalid = tvalid.unsqueeze(2) & tvalid.unsqueeze(1)  # (B,TP,TP)
            off = (sim * (1 - eye) * mvalid).sum() / ((mvalid * (1 - eye)).sum() + 1e-6)
            stats.update({"n_target": int(B * TP), "n_pred_steps": n_steps,
                          "l_pred": float(l_pred), "l_disc": float(l_disc),
                          "l_bound": float(l_bound), "l_vreg": float(l_vreg),
                          "l_lang": float(l_lang),
                          "n_lang_known": int(known.sum()) if known is not None else 0,
                          "selfsim_offdiag": float(off),
                          "pred_var": float(c.var(0).mean())})
        return loss, stats


@torch.no_grad()
def _feat_std(z):
    """Same value as collapse_stats(z)["feat_std"], as a 0-d tensor (no host sync)."""
    z = z.float()
    return (z - z.mean(0, keepdim=True)).std(0).mean()


@torch.no_grad()
def collapse_stats(z):
    """Anti-collapse diagnostics on a (N, d) feature matrix.
    eff_rank (entropy of singular-value spectrum), mean per-dim std, mean |offdiag
    corr|. Collapse → eff_rank→1, std→0."""
    z = z.float()
    z = z - z.mean(0, keepdim=True)
    std = z.std(0).mean().item()
    n = min(z.size(0), 4096)
    zz = z[:n]
    try:
        s = torch.linalg.svdvals(zz)
        p = s / (s.sum() + 1e-9)
        eff_rank = float(torch.exp(-(p * (p + 1e-12).log()).sum()))
    except Exception:
        eff_rank = float("nan")
    d = min(zz.size(1), 256)
    c = torch.corrcoef(zz[:, :d].T)
    offdiag = (c - torch.eye(d, device=c.device)).abs()
    return {"eff_rank": eff_rank, "feat_std": std,
            "offdiag_corr": offdiag.mean().item()}


def ema_momentum(step, total_steps, base=0.999, final=1.0):
    """Cosine momentum schedule base→final (V-JEPA 2)."""
    import math
    return final - (final - base) * (1 + math.cos(math.pi * step / max(1, total_steps))) / 2
