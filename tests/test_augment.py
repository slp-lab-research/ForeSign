import math
import random

import numpy as np
import pytest
import torch
import yaml

from foresign.augment import random_rotation, rotate
from tests.test_canon_data import canon_root  # noqa: F401  (fixture)


def test_rotation_is_proper_and_within_bounds():
    rng = random.Random(0)
    for _ in range(200):
        R, yaw, pitch = random_rotation(rng, 30.0, 15.0)
        assert R.dtype == np.float32
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-5)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)
        assert abs(yaw) <= 30.0 and abs(pitch) <= 15.0


def test_rotation_composition_yaw_then_pitch():
    rng = random.Random(1)
    R, yaw, pitch = random_rotation(rng, 30.0, 15.0)
    a, b = math.radians(yaw), math.radians(pitch)
    Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    assert np.allclose(R, Rx @ Ry, atol=1e-6)
    # pure yaw keeps y, pure pitch keeps x
    Ryo, _, p0 = random_rotation(random.Random(2), 30.0, 0.0)
    assert p0 == 0.0 and np.allclose(Ryo[1], [0, 1, 0])


def test_off_draws_nothing():
    rng = random.Random(5)
    state = rng.getstate()
    R, yaw, pitch = random_rotation(rng, 0.0, 0.0)
    assert R is None and yaw == 0.0 and pitch == 0.0 and rng.getstate() == state


def test_rotate_preserves_masked_and_norms():
    x = np.random.default_rng(0).normal(size=(10, 77, 3)).astype(np.float16)
    x[:, 7:9] = 0
    R, _, _ = random_rotation(random.Random(3), 30.0, 15.0)
    y = rotate(x, R)
    assert y.dtype == np.float32 and y.shape == x.shape
    assert np.all(y[:, 7:9] == 0)
    assert np.allclose(np.linalg.norm(y, axis=-1), np.linalg.norm(x.astype(np.float32), axis=-1),
                       atol=1e-3)


def _stream(root, ex, tmp_path, **kw):
    from foresign.canon_data import CanonClips
    return CanonClips(datasets=["autsl", "youtube_asl", "phoenix14t"], clip_len=32, root=root,
                      exclude_path=ex, cache_dir=str(tmp_path / "cache"), seed=0, **kw)


def test_stream_off_by_default_and_rotation_applied(canon_root, tmp_path):  # noqa: F811
    root, ex = canon_root
    base = _stream(root, ex, tmp_path)
    assert base.aug_yaw == 0.0 and base.aug_pitch == 0.0
    off = [next(iter(_stream(root, ex, tmp_path)))[0] for _ in range(1)]
    on_it = iter(_stream(root, ex, tmp_path, aug_rot_yaw_deg=30.0, aug_rot_pitch_deg=15.0))
    on = next(on_it)[0]
    v_off, v_on = off[0].reshape(32, 77, 4), on.reshape(32, 77, 4)
    assert torch.equal(v_off[..., 3], v_on[..., 3])                    # masks unchanged
    assert not torch.allclose(v_off[..., :3], v_on[..., :3])           # coords rotated
    assert torch.all(v_on[:, 7:9] == 0)                                 # absent joints stay 0
    # norms per joint preserved (rotation about the origin)
    assert torch.allclose(v_off[..., :3].norm(dim=-1), v_on[..., :3].norm(dim=-1), atol=1e-2)


def test_stream_deterministic_given_seed(canon_root, tmp_path):  # noqa: F811
    root, ex = canon_root
    a = [next(iter(_stream(root, ex, tmp_path, aug_rot_yaw_deg=30.0, aug_rot_pitch_deg=15.0)))[0]]
    b = [next(iter(_stream(root, ex, tmp_path, aug_rot_yaw_deg=30.0, aug_rot_pitch_deg=15.0)))[0]]
    assert torch.equal(a[0], b[0])


def test_context_and_target_see_identical_input():
    """The rotated window is the single model input: tokenizer (context) and target_tok
    receive the same tensor."""
    from foresign.build import build_model
    cfg = dict(d_model=24, depth=1, n_heads=2, pred_dim=12, pred_depth=1, pred_heads=2,
               t_patch=2, ema_base=0.99, input_format="c77", in_dim=3)
    m = build_model(cfg)
    seen = {}
    def hook(key):
        def f(mod, inp, out):
            seen.setdefault(key, inp[0].clone())       # returns None: output untouched
        return f
    m.tokenizer.register_forward_hook(hook("ctx"))
    m.target_tok.register_forward_hook(hook("tgt"))
    x = torch.randn(2, 8, 77 * 4)
    mask = torch.zeros(2, 4, 18, dtype=torch.bool)
    mask[:, :2] = True
    m(x, mask, torch.ones(2, 4, 18, dtype=torch.bool))
    assert torch.equal(seen["ctx"], seen["tgt"])


def test_rot_configs():
    import os
    d = os.path.join(os.path.dirname(__file__), "..", "configs", "p2")
    for obj in ("pred", "ms"):
        rot = yaml.safe_load(open(os.path.join(d, f"p2_{obj}_rot_s0.yaml"), encoding="utf-8"))
        long = yaml.safe_load(open(os.path.join(d, f"p2_{obj}_long_s0.yaml"), encoding="utf-8"))
        assert rot["aug_rot_yaw_deg"] == 30.0 and rot["aug_rot_pitch_deg"] == 15.0
        assert rot["output_dir"] == f"runs/p2/p2_{obj}_rot_s0"
        diff = {k for k in set(rot) | set(long) if rot.get(k) != long.get(k)}
        assert diff == {"aug_rot_yaw_deg", "aug_rot_pitch_deg", "output_dir"}
        assert rot["total_steps"] == 64000 and rot["const_until"] == 51200
        assert rot["ckpt_every"] == 4000
    for name in ("p2_ms_s0", "p2_pred_long_s0"):
        c = yaml.safe_load(open(os.path.join(d, f"{name}.yaml"), encoding="utf-8"))
        assert not c.get("aug_rot_yaw_deg") and not c.get("aug_rot_pitch_deg")


def test_finetune_launcher_accepts_rot_ckpt(tmp_path):
    from foresign.build import build_model
    from tests.test_canon_evals import TINY
    import tools.run_finetune_p2 as L
    cfg = {**TINY, "total_steps": 64000}
    run = tmp_path / "p2_ms_rot_s0"
    run.mkdir()
    torch.save({"model": build_model(cfg).state_dict(), "step": 16000, "cfg": cfg},
               run / "step_16000.pt")
    torch.save({"model": build_model(cfg).state_dict(), "step": 64000, "cfg": cfg},
               run / "last.pt")
    assert L.ckpt_tag(str(run / "step_16000.pt"))[0] == "p2_ms_rot_s0_k16000_L0"
    assert L.ckpt_tag(str(run / "last.pt"))[0] == "p2_ms_rot_s0_k64000_L0"
