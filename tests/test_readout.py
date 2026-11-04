import torch

from foresign.build import build_model

CFG = dict(d_model=64, depth=3, n_heads=4, pred_dim=32, pred_depth=1, pred_heads=4,
           t_patch=2, ema_base=0.999, clip_len=64, drop_conf=False)


def _ckpt(tmp_path):
    m = build_model(CFG)
    p = tmp_path / "c.pt"
    torch.save({"model": m.state_dict(), "cfg": CFG, "step": 0}, p)
    return str(p)


def test_adapter_tap_changes_features(tmp_path, monkeypatch):
    from foresign.adapters import ForeSignAdapter
    ck = _ckpt(tmp_path)
    x = torch.randn(2, 64, 234); nv = torch.tensor([64, 40])
    monkeypatch.setenv("FS_TAP_DEPTH", "0")
    a0 = ForeSignAdapter(ck, device="cpu").pool(x, nv)
    monkeypatch.setenv("FS_TAP_DEPTH", "1")
    a1 = ForeSignAdapter(ck, device="cpu").pool(x, nv)
    assert a0.shape == a1.shape == (2, 64)
    assert not torch.allclose(a0, a1)


def test_window_encoder_uses_ckpt_clip_len(tmp_path):
    from evals.cslr_phoenix import WindowEncoder
    enc = WindowEncoder(_ckpt(tmp_path), freeze=True)
    assert enc.win == 64 and enc.win_tok == 32
