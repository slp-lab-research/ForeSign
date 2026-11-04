import torch

from foresign.build import build_model

CFG = dict(d_model=64, depth=2, n_heads=4, pred_dim=32, pred_depth=1, pred_heads=4,
           t_patch=2, ema_base=0.999, clip_len=64)


def _batch():
    torch.manual_seed(0)
    x = torch.randn(2, 64, 234)
    mask = torch.rand(2, 32, 18) < 0.78
    valid = torch.ones(2, 32, 18, dtype=torch.bool)
    return x, mask, valid


def test_pred_only_has_no_nce():
    m = build_model(CFG); x, mask, valid = _batch()
    loss, st = m(x, mask, valid)
    assert torch.isfinite(loss) and st["l_nce"] == 0.0


def test_nce_only_excludes_pred():
    m = build_model(CFG); x, mask, valid = _batch()
    loss, st = m(x, mask, valid, alpha_pred=0.0, beta=1.0, tau=0.1)
    assert abs(float(loss) - st["l_nce"]) < 1e-4
    assert st["l_nce"] > 0


def test_ms_combines_terms():
    m = build_model(CFG); x, mask, valid = _batch()
    bt = torch.rand(2, 32)
    loss, st = m(x, mask, valid, alpha_pred=1.0, beta=0.3, tau=0.1,
                 boundary_target=bt, lambda_bound=0.2)
    expect = st["l_pred"] + 0.3 * st["l_nce"] + 0.2 * st["l_bound"]
    assert abs(float(loss) - expect) < 1e-3
    loss.backward()
    assert any(p.grad is not None for p in m.context.parameters())


def test_nce_only_skips_masked_path():
    m = build_model(CFG); x, mask, valid = _batch()

    def _boom(*a, **k):
        raise AssertionError("predictor must not run when alpha_pred == 0")
    m.predictor.forward = _boom
    calls = []
    m.context.register_forward_hook(lambda mod, i, o: calls.append(1))
    loss, st = m(x, mask, valid, alpha_pred=0.0, beta=1.0, tau=0.1)
    assert len(calls) == 1
    assert abs(float(loss) - st["l_nce"]) < 1e-4 and st["l_nce"] > 0
    assert st["l_pred"] == 0.0 and st["l_bound"] == 0.0 and st["l_vreg"] == 0.0
    assert st["n_target"] == 0 and st["pred_var"] == 0.0 and "eff_rank" in st
    loss.backward()
    assert any(p.grad is not None for p in m.context.parameters())


def test_temporal_nce_is_fp32_under_autocast():
    m = build_model(CFG); x, mask, valid = _batch()
    torch.manual_seed(1)
    z = torch.randn(2, 32, 18, 64)
    c = torch.randn(2, 32, 18, 64) * 3.0
    m._encode_ctx = lambda x_, mask_: c          # isolate the similarity/softmax math
    ref = m._temporal_nce(x, z, valid, 0.1)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        got = m._temporal_nce(x, z, valid, 0.1)
    assert got.dtype == torch.float32
    assert abs(float(got) - float(ref)) < 1e-5


def test_boundary_loss_ignores_padded_patches():
    """L_bound must only score valid (non-padded) time patches: changing the
    boundary target on padded patches must not change the loss."""
    m = build_model(CFG); x, mask, valid = _batch()
    valid[:, 20:, :] = False                     # last 12 patches are padding
    mask = mask & valid                          # masking never selects padding
    bt = torch.rand(2, 32)
    bt2 = bt.clone(); bt2[:, 20:] = 1.0 - bt2[:, 20:]
    torch.manual_seed(1); _, s1 = m(x, mask, valid, boundary_target=bt, lambda_bound=0.2)
    torch.manual_seed(1); _, s2 = m(x, mask, valid, boundary_target=bt2, lambda_bound=0.2)
    assert abs(s1["l_bound"] - s2["l_bound"]) < 1e-6
