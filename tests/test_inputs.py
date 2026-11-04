import numpy as np
import pytest
import torch

from foresign.inputs import C77, RESONA77, Layout, layout_for, pack_c77


def test_layouts():
    old = layout_for({})
    assert old == Layout(78, 3, 3, 2, 2) and old.flat == 234 and old.fmt == RESONA77
    new = layout_for({"input_format": "c77", "in_dim": 3})
    assert new == Layout(77, 4, 3, 3, 3, C77) and new.flat == 308
    assert layout_for({"input_format": "c77", "in_dim": 2}).flat == 77 * 3


def test_c77_rejects_drop_conf_and_bad_in_dim():
    with pytest.raises(ValueError):
        layout_for({"input_format": "c77", "drop_conf": True})
    with pytest.raises(ValueError):
        layout_for({"input_format": "c77", "in_dim": 4})
    with pytest.raises(ValueError):
        layout_for({"input_format": "bogus"})


def test_pack_c77_zeroes_invalid_and_appends_mask():
    T = 5
    x = np.random.default_rng(0).normal(size=(T, 77, 3)).astype(np.float16)
    m = np.ones((T, 77), np.uint8)
    m[1, 10] = 0
    p = pack_c77(x, m, 3)
    assert p.shape == (T, 77 * 4) and p.dtype == np.float32
    v = p.reshape(T, 77, 4)
    assert np.all(v[1, 10] == 0)
    assert np.allclose(v[0, 10, :3], x[0, 10].astype(np.float32))
    assert v[0, 10, 3] == 1 and v[1, 10, 3] == 0
    p2 = pack_c77(x, m, 2).reshape(T, 77, 3)
    assert np.allclose(p2[..., :2], v[..., :2]) and np.all(p2[..., 2] == v[..., 3])


def test_tokenizer_old_layout_unchanged_shapes():
    from foresign.tokenize import PartLinearTokenizer
    tok = PartLinearTokenizer(d_model=16, t_patch=2)
    assert tok.proj[0].in_features == 3 * 2 * 3          # trunk 3 joints x t_patch 2 x 3 ch
    out = tok(torch.randn(2, 8, 234))
    assert out.shape == (2, 4, 18, 16)


def test_tokenizer_c77_projects_coords_and_gates_on_mask():
    from foresign.tokenize import PartLinearTokenizer
    lay = layout_for({"input_format": "c77", "in_dim": 3})
    torch.manual_seed(0)
    tok = PartLinearTokenizer(d_model=16, t_patch=2, layout=lay)
    assert tok.proj[0].in_features == 3 * 2 * 3
    x = torch.randn(2, 8, 77, 4)
    x[..., 3] = 1.0
    full = tok(x.reshape(2, 8, -1))
    # changing only the mask channel changes the output by exactly the absent-vector term
    x2 = x.clone()
    x2[:, :2, :, 3] = 0.0                                  # first time patch fully absent
    out2 = tok(x2.reshape(2, 8, -1))
    diff = out2 - full
    assert torch.allclose(diff[:, 0], tok.absent[0, 0].expand(2, 18, 16), atol=1e-6)
    assert torch.allclose(diff[:, 1:], torch.zeros_like(diff[:, 1:]))
    conf = tok.token_conf(x2.reshape(2, 8, -1))
    assert conf.shape == (2, 4, 18) and torch.all(conf[:, 0] == 0) and torch.all(conf[:, 1:] == 1)


def test_build_model_takes_layout_from_cfg():
    from foresign.build import build_model
    cfg = dict(d_model=24, depth=1, n_heads=2, pred_dim=12, pred_depth=1, pred_heads=2,
               t_patch=2, ema_base=0.99, input_format="c77", in_dim=3)
    m = build_model(cfg)
    assert m.tokenizer.layout.flat == 308 and m.target_tok.layout.flat == 308
    old = build_model({k: v for k, v in cfg.items() if k not in ("input_format", "in_dim")})
    assert old.tokenizer.layout.flat == 234


def test_motion_energy_on_c77_layout():
    from foresign.masking import frame_motion_energy, temporal_boundary_target
    lay = layout_for({"input_format": "c77", "in_dim": 3})
    x = torch.zeros(1, 8, 77, 4)
    x[..., 3] = 1.0
    x[0, 4:, 9, 2] = 1.0                                   # z-only jump of one joint at t=4
    e = frame_motion_energy(x.reshape(1, 8, -1), layout=lay)
    assert e.shape == (1, 8) and e[0, 4] > 0 and e[0, 3] == 0 and e[0, 5] == 0
    x[0, :, 9, 3] = 0.0                                    # joint absent -> no motion counted
    assert frame_motion_energy(x.reshape(1, 8, -1), layout=lay).abs().sum() == 0
    bt, mp = temporal_boundary_target(x.reshape(1, 8, -1), 2, layout=lay)
    assert bt.shape == (1, 4)
