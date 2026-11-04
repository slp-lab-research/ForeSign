"""ForeSign masking families (design §6). Returns a boolean MASK over the
(B, TP, V=18) token grid where True = MASKED (a prediction target) and
False = VISIBLE (context). All masks defined over VALID (non-pad) tokens.

P1 collapse-check uses M3 only. M3′ is the ratio/geometry-matched control that
makes the M1 comparison interpretable (R1 N1). M1/M2 land with the gate.

V-JEPA multiblock (M3): union of several (time × space) blocks spanning the full
temporal extent (temporal scale 1.0), short narrow blocks + long wide blocks, with
an optional confidence/motion saliency bias on block placement.
"""
from __future__ import annotations

import torch

from foresign.groups import (
    N_GROUPS, GROUP_INDEX, HAND_GROUPS, LHAND_GROUPS, RHAND_GROUPS,
    ARM_GROUPS, FACE_GROUPS,
)

_COLS = {
    "hands": [GROUP_INDEX[g] for g in HAND_GROUPS],          # 12
    "arms": [GROUP_INDEX[g] for g in ARM_GROUPS],            # 2
    "face": [GROUP_INDEX[g] for g in FACE_GROUPS],           # 3
    "l_handarm": [GROUP_INDEX[g] for g in LHAND_GROUPS + ["l_arm"]],
    "r_handarm": [GROUP_INDEX[g] for g in RHAND_GROUPS + ["r_arm"]],
}


def _rand_block(B, TP, V, t_frac, s_frac, device, gen=None):
    """One contiguous (time × group) block per sample → bool (B,TP,V)."""
    th = max(1, int(round(TP * t_frac)))
    sh = max(1, int(round(V * s_frac)))
    t0 = torch.randint(0, max(1, TP - th + 1), (B,), device=device, generator=gen)
    s0 = torch.randint(0, max(1, V - sh + 1), (B,), device=device, generator=gen)
    tar = torch.arange(TP, device=device)
    sar = torch.arange(V, device=device)
    tm = (tar[None, :] >= t0[:, None]) & (tar[None, :] < (t0 + th)[:, None])  # (B,TP)
    sm = (sar[None, :] >= s0[:, None]) & (sar[None, :] < (s0 + sh)[:, None])  # (B,V)
    return tm[:, :, None] & sm[:, None, :]                                    # (B,TP,V)


def multiblock_mask(B, TP, V, device, *, n_short=8, short_s=0.15, n_long=2,
                    long_s=0.7, t_frac=1.0, valid=None, min_keep=2, gen=None):
    """M3. Union of n_short narrow + n_long wide full-temporal blocks.
    valid: (B,TP,V) bool of non-pad tokens (mask restricted to valid).
    Guarantees >=min_keep visible valid tokens/sample (predictor needs context)."""
    mask = torch.zeros(B, TP, V, dtype=torch.bool, device=device)
    for _ in range(n_short):
        mask |= _rand_block(B, TP, V, t_frac, short_s, device, gen)
    for _ in range(n_long):
        mask |= _rand_block(B, TP, V, t_frac, long_s, device, gen)
    if valid is not None:
        mask &= valid                                   # never "mask" a pad token
    # ensure some visible context among valid tokens
    if valid is not None:
        vis_valid = valid & ~mask
        deficit = (vis_valid.flatten(1).sum(1) < min_keep)
        if deficit.any():
            # for deficient samples, unmask a few random valid tokens
            for b in deficit.nonzero(as_tuple=True)[0].tolist():
                idx = valid[b].flatten().nonzero(as_tuple=True)[0]
                if idx.numel():
                    pick = idx[torch.randperm(idx.numel(), device=device)[:min_keep]]
                    flat = mask[b].flatten(); flat[pick] = False
                    mask[b] = flat.reshape(TP, V)
    return mask


def ratio_matched_random_mask(B, TP, V, device, *, ratio, valid=None,
                              block_s=None, gen=None):
    """M3′ control. Mask a matched FRACTION of valid tokens as a random set,
    optionally in small contiguous spatial blocks (block_s groups wide) to match
    M1a's spatial-continuity WITHOUT respecting anatomical part boundaries —
    isolates 'part structure' from 'masking a correlated chunk' (R1 N1)."""
    if block_s is None:
        # plain random tokens at the target ratio (over valid tokens)
        noise = torch.rand(B, TP, V, device=device, generator=gen)
        if valid is not None:
            noise = noise.masked_fill(~valid, 2.0)       # push pads to the end
        k = (valid.flatten(1).sum(1) * ratio).long() if valid is not None \
            else torch.full((B,), int(TP * V * ratio), device=device)
        flat = noise.flatten(1)
        order = flat.argsort(1)
        mask = torch.zeros(B, TP * V, dtype=torch.bool, device=device)
        for b in range(B):
            mask[b, order[b, :k[b]]] = True
        return mask.reshape(B, TP, V)
    # geometry-matched: random small blocks until ratio reached
    mask = torch.zeros(B, TP, V, dtype=torch.bool, device=device)
    target = (valid.flatten(1).sum(1) * ratio) if valid is not None \
        else torch.full((B,), TP * V * ratio, device=device)
    for _ in range(64):
        add = _rand_block(B, TP, V, 1.0, block_s / V, device, gen)
        mask |= (add & valid) if valid is not None else add
        if (mask.flatten(1).sum(1) >= target).all():
            break
    return mask


# ---------------------------------------------------------------------------
# M1 — part-structured masks (design §6). Mask whole anatomical group COLUMNS
# across all valid time. Names are anatomical; the phonological interpretation
# is a VALIDATION target (ASL-LEX probe), not a claim.
# ---------------------------------------------------------------------------

def _col_mask(B, TP, V, cols, device, valid=None):
    """Mask the given group columns across all time → (B,TP,V) bool."""
    m = torch.zeros(B, TP, V, dtype=torch.bool, device=device)
    if cols:
        m[:, :, torch.tensor(cols, device=device)] = True
    return (m & valid) if valid is not None else m


def m1a_strict(B, TP, V, device, valid=None):
    """Both hands (12) + both arms (2) masked → predict from trunk+face (5).
    Wrist lives in arm group so hand position/movement is hidden too (no leak)."""
    return _col_mask(B, TP, V, _COLS["hands"] + _COLS["arms"], device, valid)


def m1a_loose(B, TP, V, device, valid=None):
    """Both hands (12) masked, arms (body-wrist) VISIBLE → position given,
    handshape/orientation predicted. strict−loose isolates position vs shape."""
    return _col_mask(B, TP, V, _COLS["hands"], device, valid)


def m1b_face(B, TP, V, device, valid=None):
    return _col_mask(B, TP, V, _COLS["face"], device, valid)


def m1c_one_hand(B, TP, V, device, valid=None, bimanual=None, gen=None):
    """Mask ONE hand+arm per sample (random L/R), only where bimanual=True
    (design §4.3: one-hand masking is meaningless on one-handed signs). Samples
    not bimanual fall back to m1a_loose so the batch still has targets."""
    pickL = torch.rand(B, device=device, generator=gen) < 0.5
    m = torch.zeros(B, TP, V, dtype=torch.bool, device=device)
    lcol = torch.tensor(_COLS["l_handarm"], device=device)
    rcol = torch.tensor(_COLS["r_handarm"], device=device)
    for b in range(B):
        m[b, :, lcol if pickL[b] else rcol] = True
    if bimanual is not None:
        fb = m1a_loose(B, TP, V, device)                     # fallback for 1-handed
        m = torch.where(bimanual[:, None, None], m, fb)
    return (m & valid) if valid is not None else m


# ---------------------------------------------------------------------------
# M2 — anticipation mask (design §6). Continuous clips ONLY (caller gates by
# clip type). Mask all valid tokens after t0 ~ U[t_lo, t_hi]·(valid extent).
# ---------------------------------------------------------------------------

def m2_anticipation(B, TP, V, device, valid=None, t_lo=0.4, t_hi=0.8, gen=None):
    """Predict the rest of the clip from its first 40-80%."""
    if valid is not None:
        # per-sample valid time extent (max valid time-patch index +1)
        tvalid = valid.any(dim=2)                            # (B,TP)
        extent = tvalid.float().sum(dim=1).clamp(min=1)      # (B,)
    else:
        extent = torch.full((B,), TP, device=device, dtype=torch.float)
    frac = t_lo + (t_hi - t_lo) * torch.rand(B, device=device, generator=gen)
    t0 = (frac * extent).long().clamp(min=1)                 # (B,)
    ar = torch.arange(TP, device=device)
    after = ar[None, :] >= t0[:, None]                       # (B,TP)
    m = after[:, :, None].expand(B, TP, V)
    return (m & valid) if valid is not None else m.clone()


# ---------------------------------------------------------------------------
# M3' ratio-matched control for a given M1 variant (R1 N1): random tokens at
# the SAME ratio as the part mask but ignoring anatomical boundaries.
# ---------------------------------------------------------------------------

def m3prime_match(part_mask, valid, gen=None):
    """Given an M1 part_mask (B,TP,V) + valid, return a random mask with the
    SAME per-sample masked-token count over valid tokens."""
    B, TP, V = part_mask.shape
    device = part_mask.device
    k = (part_mask & valid).flatten(1).sum(1) if valid is not None \
        else part_mask.flatten(1).sum(1)
    noise = torch.rand(B, TP, V, device=device, generator=gen)
    if valid is not None:
        noise = noise.masked_fill(~valid, 2.0)
    order = noise.flatten(1).argsort(1)
    out = torch.zeros(B, TP * V, dtype=torch.bool, device=device)
    for b in range(B):
        out[b, order[b, :int(k[b])]] = True
    return out.reshape(B, TP, V)


# ---------------------------------------------------------------------------
# Gate dispatcher: pick a mask per step by condition (design §6 family probs).
# condition ∈ {m3, m3prime, m1, m2, m1m2}. Default m3 = current behavior.
# ---------------------------------------------------------------------------

def _coords_weight(pose, layout):
    """Split a flat pose into motion coordinates and the per-joint weight channel.
    Default (layout None) = resona77: (x, y) weighted by conf."""
    from foresign.inputs import OLD
    L = layout or OLD
    B, T, _ = pose.shape
    v = pose.reshape(B, T, L.n_joints, L.chans)
    return v[..., :L.motion], v[..., L.gate], L.n_joints


def group_motion_energy(pose, t_patch, group_idx, n_joints=78, layout=None):
    """conf-weighted per-(time-patch, group) motion saliency (S-JEPA / design v0.3:
    m = min(c_t, c_{t-1})·|Δxy|). pose:(B,T,234) → (B,TP,n_groups). Confidence
    gating prevents low-conf jitter from masquerading as motion. For c77 input the
    weight is the joint mask and the motion uses all in_dim coordinates."""
    B, T, _ = pose.shape
    TP = T // t_patch
    xy, c, n_joints = _coords_weight(pose, layout)
    dx = (xy[:, 1:] - xy[:, :-1]).norm(dim=-1)            # (B,T-1,J)
    cw = torch.minimum(c[:, 1:], c[:, :-1])               # (B,T-1,J) min conf
    m = cw * dx
    m = torch.nn.functional.pad(m, (0, 0, 1, 0))          # (B,T,J)
    mp = m.reshape(B, TP, t_patch, n_joints).mean(2)      # (B,TP,J)
    return torch.stack([mp[:, :, js].mean(-1) for js in group_idx], dim=2)


def frame_motion_energy(pose, n_joints=78, layout=None):
    """conf-weighted per-frame motion magnitude (B,T). m_t = sum_j min(c_t,c_{t-1})|Δxy|."""
    xy, c, _ = _coords_weight(pose, layout)
    dx = (xy[:, 1:] - xy[:, :-1]).norm(dim=-1)            # (B,T-1,J)
    cw = torch.minimum(c[:, 1:], c[:, :-1])
    m = (cw * dx).sum(-1) / (cw.sum(-1) + 1e-6)           # (B,T-1)
    return torch.nn.functional.pad(m, (1, 0))            # (B,T)


def temporal_boundary_target(pose, t_patch, layout=None):
    """BA-JEPA unsupervised gloss-boundary proxy. Patch-level motion-energy
    acceleration |Δ(patch_motion)| peaks at sign transitions/holds. Returns
    (B,TP) soft boundary score in [0,1] (per-clip min-max) + patch motion."""
    m = frame_motion_energy(pose, layout=layout)          # (B,T)
    B, T = m.shape
    TP = T // t_patch
    mp = m[:, :TP * t_patch].reshape(B, TP, t_patch).mean(-1)   # (B,TP)
    g = (mp[:, 1:] - mp[:, :-1]).abs()
    g = torch.nn.functional.pad(g, (1, 0))                # (B,TP)
    lo = g.min(1, keepdim=True).values
    hi = g.max(1, keepdim=True).values
    return (g - lo) / (hi - lo + 1e-6), mp


def temporal_saliency_segments(z_tgt_patch, s_boundary, n_seg=5, lam=1.0, w=3):
    """DSeq-T-JEPA temporal saliency / ordering (design §1-2). Cut the TP axis into
    n_seg contiguous segments at the top-(n_seg-1) peaks of a blended saliency, and
    return them ordered DESCENDING by mean saliency (most-discriminative first).

      z_tgt_patch : (B,TP,d) target latent per temporal patch (mean over joints).
      s_boundary  : (B,TP) temporal_boundary_target warmup prior in [0,1].
      returns     : (segs, s) where
                      segs = list over B of lists of (start,end) spans, each list
                             sorted DESC by mean blended saliency s over the span,
                      s    = (B,TP) blended saliency in [0,1].

    Novelty score s_t = ||z_t - movavg(z, window=w)||_2 (per-patch temporal change),
    min-max normalized per clip; blended s = (1-lam)*s_boundary + lam*s_novelty.
    Robust to small TP and degenerate (all-equal) saliency."""
    B, TP, d = z_tgt_patch.shape
    z = z_tgt_patch.float()
    # causal-symmetric moving average over the temporal axis (window w, reflect via clamp)
    if w > 1 and TP > 1:
        pad = w // 2
        zt = z.transpose(1, 2)                                   # (B,d,TP)
        zt = torch.nn.functional.pad(zt, (pad, pad), mode="replicate")
        mov = torch.nn.functional.avg_pool1d(zt, kernel_size=w, stride=1)
        mov = mov[..., :TP].transpose(1, 2)                      # (B,TP,d)
    else:
        mov = z
    s_nov = (z - mov).norm(dim=-1)                               # (B,TP)
    lo = s_nov.min(1, keepdim=True).values
    hi = s_nov.max(1, keepdim=True).values
    s_nov = (s_nov - lo) / (hi - lo + 1e-6)                      # per-clip min-max
    s = (1.0 - lam) * s_boundary.float() + lam * s_nov           # (B,TP)
    # cut into n_seg contiguous segments at the top-(n_seg-1) peaks of s (per sample)
    n_cut = max(0, min(n_seg - 1, TP - 1))
    segs = []
    for b in range(B):
        sb = s[b]
        if n_cut > 0:
            # candidate cut points are interior indices 1..TP-1 (cut BEFORE that index)
            cand = torch.arange(1, TP, device=s.device)
            cand_scores = sb[cand]                               # saliency at cut points
            order = cand_scores.argsort(descending=True)
            picked = sorted(set(cand[order[:n_cut]].tolist()))
        else:
            picked = []
        cuts = sorted(set([0, TP] + picked))
        spans = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)
                 if cuts[i + 1] > cuts[i]]
        # order segments DESCENDING by mean saliency
        spans.sort(key=lambda se: sb[se[0]:se[1]].mean().item(), reverse=True)
        segs.append(spans)
    return segs, s


def boundary_mask(pose, t_patch, V, ratio=0.7, valid=None, gen=None, thr_q=0.6, layout=None):
    """BA-JEPA boundary-AWARE masking: split the TP axis into temporal SEGMENTS at
    high-boundary patches, then mask WHOLE segments (all V) until ~ratio of patches
    are masked. Forces predicting entire missing gloss-segments from neighbouring
    visible segments. A patch is fully-visible or fully-masked → clean per-patch
    boundary supervision. Returns (mask (B,TP,V) True=predict, boundary_tgt (B,TP))."""
    bt, _ = temporal_boundary_target(pose, t_patch, layout=layout)   # (B,TP)
    B, TP = bt.shape
    mask = torch.zeros(B, TP, V, dtype=torch.bool, device=pose.device)
    need = max(1, int(ratio * TP))
    for b in range(B):
        thr = torch.quantile(bt[b], thr_q)
        bnd = (bt[b] >= thr).nonzero().flatten().tolist()
        cuts = sorted(set([0, TP] + bnd))
        segs = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] > cuts[i]]
        perm = (torch.randperm(len(segs), generator=gen) if gen is not None
                else torch.randperm(len(segs))).tolist()
        got = 0
        for si in perm:
            s, e = segs[si]
            mask[b, s:e, :] = True
            got += (e - s)
            if got >= need:
                break
    if valid is not None:
        mask = mask & valid
    return mask, bt


def saliency_mask(motion, ratio, valid=None, gen=None, invert=False, tau=0.5):
    """S-JEPA motion-aware masking, SCATTER geometry (per-token selection — only
    the SELECTION axis differs from rand78; ratio/geometry identical). Stochastic:
    Gumbel-top-k over softmax(motion/tau) so high-motion tokens are PREFERENTIALLY
    (not deterministically) masked, faithful to S-JEPA. invert=True → low-motion."""
    B, TP, V = motion.shape
    s = (-motion if invert else motion).flatten(1)        # (B, TP*V)
    s = s / (s.abs().max(1, keepdim=True).values * tau + 1e-9)
    logp = torch.log_softmax(s, dim=1)
    u = (torch.rand(s.shape, device=s.device, generator=gen)
         if gen is not None else torch.rand_like(s))
    score = logp - torch.log(-torch.log(u + 1e-10) + 1e-10)   # Gumbel
    if valid is not None:
        score = score.masked_fill(~valid.flatten(1), float("-inf"))
    k = ((valid.flatten(1).sum(1) if valid is not None
          else torch.full((B,), TP * V, device=motion.device)) * ratio).long()
    order = score.argsort(1, descending=True)
    mask = torch.zeros(B, TP * V, dtype=torch.bool, device=motion.device)
    for b in range(B):
        mask[b, order[b, :int(k[b])]] = True
    return mask.reshape(B, TP, V)


def build_mask(condition, B, TP, V, device, *, valid=None, cfg=None,
               is_continuous=None, motion=None, gen=None):
    cfg = cfg or {}
    def m3():
        return multiblock_mask(B, TP, V, device,
                               n_short=cfg.get("n_short", 8), short_s=cfg.get("short_s", 0.15),
                               n_long=cfg.get("n_long", 2), long_s=cfg.get("long_s", 0.7),
                               valid=valid, gen=gen)
    def m1_family():
        r = torch.rand(1, generator=gen).item() if gen is not None else torch.rand(1).item()
        if r < 0.45:   return m1a_strict(B, TP, V, device, valid)
        if r < 0.70:   return m1a_loose(B, TP, V, device, valid)
        if r < 0.85:   return m1b_face(B, TP, V, device, valid)
        return m1c_one_hand(B, TP, V, device, valid, bimanual=is_continuous, gen=gen)
    def m2():
        return m2_anticipation(B, TP, V, device, valid, gen=gen)

    if condition == "m3":
        return m3()
    if condition == "m3prime":
        return m3prime_match(m1a_strict(B, TP, V, device, valid), valid, gen=gen)
    # ---- ratio-controlled confirmation conditions (all ≈ m1a-strict 14/18≈0.78
    # of valid tokens; vary ONLY spatial structure) — isolates structure vs ratio
    if condition == "part78":     # anatomical part masking (hands+arms), 78%
        return m1a_strict(B, TP, V, device, valid)
    if condition == "rand78":     # scattered random tokens, matched 78%
        return m3prime_match(m1a_strict(B, TP, V, device, valid), valid, gen=gen)
    if condition == "block78":    # contiguous spatial blocks, matched 78%
        return ratio_matched_random_mask(B, TP, V, device, ratio=14.0 / 18.0,
                                         valid=valid, block_s=4, gen=gen)
    if condition == "saliency78": # S-JEPA motion-aware: mask highest-motion 78%
        if motion is None:
            raise ValueError("saliency78 needs motion (pass build_mask(..., motion=...))")
        return saliency_mask(motion, 14.0 / 18.0, valid=valid, gen=gen)
    if condition == "antisaliency78":  # control: mask LOWEST-motion 78%
        if motion is None:
            raise ValueError("antisaliency78 needs motion")
        return saliency_mask(motion, 14.0 / 18.0, valid=valid, gen=gen, invert=True)
    r = (torch.rand(1, generator=gen).item() if gen is not None else torch.rand(1).item())
    if condition == "m1":          # M3 0.6 / M1 0.4
        return m3() if r < 0.6 else m1_family()
    if condition == "m2":          # M3 0.6 / M2 0.4
        return m3() if r < 0.6 else m2()
    if condition == "m1m2":        # M3 0.6 / M1 0.25 / M2 0.15
        return m3() if r < 0.6 else (m1_family() if r < 0.85 else m2())
    raise ValueError(f"unknown masking condition {condition!r}")
