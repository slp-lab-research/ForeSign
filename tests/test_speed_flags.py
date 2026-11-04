import copy

import pytest
import torch

from foresign.build import build_model

CFG = dict(d_model=24, depth=3, n_heads=2, pred_dim=12, pred_depth=2, pred_heads=2,
           t_patch=2, ema_base=0.99, input_format="c77", in_dim=3, lr=1e-3, weight_decay=0.05)


def _batch(B=3, T=16):
    torch.manual_seed(1)
    x = torch.randn(B, T, 77 * 4)
    x.view(B, T, 77, 4)[..., 3] = 1.0
    mask = torch.rand(B, T // 2, 18) < 0.7
    valid = torch.ones(B, T // 2, 18, dtype=torch.bool)
    return x, mask, valid


def test_foreach_ema_matches_loop_bitwise():
    from foresign.train import ema_update
    torch.manual_seed(0)
    a = build_model(CFG)
    for p in a.tokenizer.parameters():
        p.data.add_(torch.randn_like(p))
    for p in a.context.parameters():
        p.data.add_(0.1 * torch.randn_like(p))
    b = copy.deepcopy(a)
    ema_update(a, 0.995, {})
    ema_update(b, 0.995, {"foreach_ema": True})
    for (n, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(p, q), n


@pytest.mark.parametrize("n_ckpt", [0, 1, 2])
def test_partial_checkpointing_same_loss_and_grads(n_ckpt):
    x, mask, valid = _batch()
    torch.manual_seed(0)
    full = build_model(CFG)
    part = copy.deepcopy(full)
    part.context.ckpt_layers = n_ckpt
    part.predictor.body.ckpt_layers = n_ckpt
    out = []
    for m in (full, part):
        m.train()
        torch.manual_seed(7)                                      # same drop-path draws
        loss, _ = m(x, mask, valid, beta=0.3, alpha_pred=1.0)
        loss.backward()
        out.append((loss.detach(), {n: p.grad.clone() for n, p in m.named_parameters()
                                    if p.grad is not None}))
    assert torch.allclose(out[0][0], out[1][0], atol=1e-6)
    assert out[0][1].keys() == out[1][1].keys()
    for n in out[0][1]:
        assert torch.allclose(out[0][1][n], out[1][1][n], atol=1e-5, rtol=1e-4), n


def test_build_sets_ckpt_layers():
    m = build_model({**CFG, "ckpt_layers": 1, "pred_ckpt_layers": 0})
    assert m.context.ckpt_layers == 1 and m.predictor.body.ckpt_layers == 0
    assert build_model(CFG).context.ckpt_layers is None
    # state_dict keys unchanged by the subclass
    from foresign._vendor.resona.models.v4_blocks import DSTformerBody as Orig
    o = Orig(dim=24, depth=3, num_heads=2, ffn_dim=96)
    assert set(o.state_dict()) == set(m.context.state_dict())


@pytest.mark.parametrize("obj", [dict(alpha_pred=1.0, beta=0.3), dict(alpha_pred=0.0, beta=1.0)])
def test_fast_stats_same_loss_and_feat_std(obj):
    x, mask, valid = _batch()
    torch.manual_seed(0)
    m = build_model(CFG)
    m.eval()
    l0, s0 = m(x, mask, valid, **obj)
    l1, s1 = m(x, mask, valid, fast_stats=True, want_stats=False, **obj)
    l2, s2 = m(x, mask, valid, fast_stats=True, want_stats=True, **obj)
    assert torch.equal(l0, l1) and torch.equal(l0, l2)
    assert set(s1) == {"feat_std_t"} and torch.is_tensor(s1["feat_std_t"])
    assert s1["feat_std_t"].item() == pytest.approx(s0["feat_std"], rel=1e-6)
    assert s2["eff_rank"] == pytest.approx(s0["eff_rank"]) and "feat_std_t" in s2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused AdamW needs CUDA")
def test_fused_adamw_close_to_unfused():
    from foresign.train import make_optimizer
    torch.manual_seed(0)
    w = torch.randn(64, 64, device="cuda")
    ps = [torch.nn.Parameter(w.clone()), torch.nn.Parameter(w.clone())]
    o1 = make_optimizer([ps[0]], {"lr": 1e-3, "weight_decay": 0.05})
    o2 = make_optimizer([ps[1]], {"lr": 1e-3, "weight_decay": 0.05, "fused_adamw": True})
    for _ in range(20):
        g = torch.randn(64, 64, device="cuda")
        for p, o in zip(ps, (o1, o2)):
            p.grad = g.clone(); o.step()
    assert torch.allclose(ps[0], ps[1], atol=1e-6)


def test_fast_temporal_attention_matches_original():
    from foresign._vendor.resona.models.v4_blocks import Attention
    from foresign.fastops import FastTemporalAttention, enable
    torch.manual_seed(0)
    a = Attention(24, num_heads=3, mode="temporal")
    b = copy.deepcopy(a)
    b.__class__ = FastTemporalAttention
    x = torch.randn(4 * 10, 18, 24)                 # (B*F, N, C), B=4, F=10
    assert torch.allclose(a(x, 10), b(x, 10), atol=1e-5)
    s = Attention(24, num_heads=3, mode="spatial")
    t = copy.deepcopy(s); t.__class__ = FastTemporalAttention
    assert torch.equal(s(x, 10), t(x, 10))          # spatial path untouched
    m = build_model({**CFG, "fast_attn": True})
    assert sum(isinstance(mm, FastTemporalAttention) for mm in m.modules()) == \
        2 * (3 + 3) + 2 * 2                          # context + target (3 layers) + predictor (2)
    m2 = build_model(CFG)
    x2, mask, valid = _batch()
    m2.load_state_dict(m.state_dict())
    m.eval(); m2.eval()
    assert torch.allclose(m(x2, mask, valid)[0], m2(x2, mask, valid)[0], atol=1e-5)
    assert enable(m) == 0                           # already swapped


def test_fast_norm_matches_and_swaps():
    from foresign._vendor.resona.models.v4_blocks import RMSNorm
    from foresign.fastops import FastRMSNorm
    torch.manual_seed(0)
    a = RMSNorm(24); a.weight.data.uniform_(0.5, 1.5)
    b = copy.deepcopy(a); b.__class__ = FastRMSNorm
    x = torch.randn(50, 24)
    assert torch.allclose(a(x), b(x), atol=1e-5)
    m = build_model({**CFG, "fast_norm": True})
    assert not any(type(mm) is RMSNorm for mm in m.modules())
    m2 = build_model(CFG); m2.load_state_dict(m.state_dict())
    x2, mask, valid = _batch()
    m.eval(); m2.eval()
    assert torch.allclose(m(x2, mask, valid)[0], m2(x2, mask, valid)[0], atol=1e-5)


def test_speed_flag_defaults_and_overrides(monkeypatch):
    from foresign.build import speed_flag, speed_flags
    from foresign.fastops import FastRMSNorm, FastTemporalAttention
    monkeypatch.delenv("FS_FAST_OPS", raising=False)
    old = {k: v for k, v in CFG.items() if k not in ("input_format", "in_dim")}
    assert not any(speed_flags(old).values())                     # resona77: original kernels
    assert all(speed_flags(CFG).values())                         # c77: fast by default
    assert speed_flag({**CFG, "fast_norm": False}, "fast_norm") is False
    monkeypatch.setenv("FS_FAST_OPS", "0")
    assert not any(speed_flags(CFG).values())
    assert not any(isinstance(m, (FastRMSNorm, FastTemporalAttention))
                   for m in build_model(CFG).modules())
    monkeypatch.setenv("FS_FAST_OPS", "1")
    assert all(speed_flags(old).values())
