"""canon-v2a / v2b (spec 2026-10-29 section 6)."""
import numpy as np
import pytest

from tests.test_canon import synth
from tools.canon import canon as C


def _rot(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = (np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch),
                              np.cos(roll), np.sin(roll))
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return (Rz @ Rx @ Ry).astype(np.float32)


def test_unknown_version():
    kp, world, flags, W, H = synth()
    with pytest.raises(ValueError):
        C.canon_clip(kp, world, flags, W, H, 25.0, version="canon-v9")


@pytest.mark.parametrize("ver", ["canon-v2a", "canon-v2b", "canon-v2c", "canon-v3.1"])
@pytest.mark.parametrize("ang", [(0.7, 0, 0), (0, 0.5, 0), (0, 0, 0.4), (0.6, -0.4, 0.3)])
def test_rotation_invariance_body(ver, ang):
    kp, world, flags, W, H = synth()
    x0, m0, _ = C.canon_clip(kp, world, flags, W, H, 25.0, version=ver)
    w2 = world.copy()
    w2[..., :3] = 1.2 * world[..., :3] @ _rot(*ang).T + np.array([0.3, -0.5, 2.0])
    x1, m1, info = C.canon_clip(kp, w2, flags, W, H, 25.0, version=ver)
    np.testing.assert_array_equal(m0, m1)
    np.testing.assert_allclose(x1[:, :9], x0[:, :9], atol=1e-4)
    assert not info["torso_degenerate"]


def test_v1_is_not_pitch_invariant():
    kp, world, flags, W, H = synth()
    x0, _, _ = C.canon_clip(kp, world, flags, W, H, 25.0)
    w2 = world.copy()
    w2[..., :3] = world[..., :3] @ _rot(0, 0.5, 0).T
    x1, _, _ = C.canon_clip(kp, w2, flags, W, H, 25.0)
    assert np.abs(x1[:, :9] - x0[:, :9]).max() > 0.1


def test_v2a_axes():
    kp, world, flags, W, H = synth()
    w2 = world.copy()
    w2[..., :3] = world[..., :3] @ _rot(0.6, -0.4, 0.3).T
    x, m, _ = C.canon_clip(kp, w2, flags, W, H, 25.0, version="canon-v2a")
    v = np.median(x[:, 1] - x[:, 2], 0)
    t = np.median(0.5 * (x[:, 7] + x[:, 8]) - 0.5 * (x[:, 1] + x[:, 2]), 0)
    assert v[0] > 0.9 and abs(v[1]) < 0.05 and abs(v[2]) < 0.05
    assert t[1] > 0.5 and abs(t[0]) < 0.05 and abs(t[2]) < 0.05
    np.testing.assert_allclose(np.median(0.5 * (x[:, 1] + x[:, 2]), 0), 0, atol=0.05)
    # v2a keeps the hips and a per-clip origin: shoulder midpoints vary per frame
    assert m[:, 7:9].all()
    assert np.abs(0.5 * (x[:, 1] + x[:, 2])).max() > 1e-3


def test_v2b_per_frame_centring_and_hip_mask():
    kp, world, flags, W, H = synth()
    flags[5:8, 0] = 0
    x, m, _ = C.canon_clip(kp, world, flags, W, H, 25.0, version="canon-v2b")
    ok = m[:, 1] > 0
    np.testing.assert_allclose(0.5 * (x[ok, 1] + x[ok, 2]), 0, atol=1e-5)
    assert m[:, 7:9].sum() == 0 and np.abs(x[:, 7:9]).sum() == 0
    assert m[ok][:, [0, 1, 2, 3, 4, 5, 6]].all()


def test_v2b_invariant_to_per_frame_translation():
    kp, world, flags, W, H = synth()
    x0, m0, _ = C.canon_clip(kp, world, flags, W, H, 25.0, version="canon-v2b")
    w2 = world.copy()
    jitter = np.random.default_rng(1).normal(0, 0.05, (world.shape[0], 1, 3))
    w2[..., :3] += jitter
    x1, m1, _ = C.canon_clip(kp, w2, flags, W, H, 25.0, version="canon-v2b")
    np.testing.assert_allclose(x1, x0, atol=1e-4)
    xa, _, _ = C.canon_clip(kp, w2, flags, W, H, 25.0, version="canon-v2a")
    xa0, _, _ = C.canon_clip(kp, world, flags, W, H, 25.0, version="canon-v2a")
    assert np.abs(xa - xa0).max() > 0.01


def test_v2_hands_still_reattached():
    kp, world, flags, W, H = synth()
    for ver in ("canon-v2a", "canon-v2b", "canon-v2c"):
        x, m, _ = C.canon_clip(kp, world, flags, W, H, 25.0, version=ver)
        np.testing.assert_allclose(x[:, C.J_LHAND], x[:, C.J_LWRIST], atol=1e-5)
        np.testing.assert_allclose(x[:, C.J_RHAND], x[:, C.J_RWRIST], atol=1e-5)
        assert x[:, 9:30, 0].mean() > 0 > x[:, 30:51, 0].mean()


def test_degenerate_torso_falls_back():
    kp, world, flags, W, H = synth()
    mid = 0.5 * (world[:, 11, :3] + world[:, 12, :3])
    world[:, 23, :3] = mid
    world[:, 24, :3] = mid
    x, m, info = C.canon_clip(kp, world, flags, W, H, 25.0, version="canon-v2a")
    assert info["torso_degenerate"] and np.isfinite(x).all()


def test_v2_no_pose():
    kp, world, flags, W, H = synth()
    flags[:, 0] = 0
    x, m, info = C.canon_clip(kp, world, flags, W, H, 25.0, version="canon-v2b")
    assert info["no_pose"] and m.sum() == 0


def test_v2c_hips_always_masked_and_never_read():
    kp, world, flags, W, H = synth()
    x0, m0, info = C.canon_clip(kp, world, flags, W, H, 25.0, version="canon-v2c")
    assert m0[:, 7:9].sum() == 0 and np.abs(x0[:, 7:9]).sum() == 0
    assert m0[:, [0, 1, 2, 3, 4, 5, 6]].all() and not info["torso_degenerate"]
    w2 = world.copy()                       # arbitrary / extrapolated hips change nothing
    w2[:, 23:25, :3] = np.random.default_rng(3).normal(0, 2, (world.shape[0], 2, 3))
    kp2 = kp.copy()
    kp2[:, 23:25, :2] = 0.99
    x1, m1, _ = C.canon_clip(kp2, w2, flags, W, H, 25.0, version="canon-v2c")
    np.testing.assert_array_equal(m0, m1)
    np.testing.assert_allclose(x1, x0, atol=1e-6)
    # v2a / v2b do depend on the hips (frame)
    xa0, _, _ = C.canon_clip(kp, world, flags, W, H, 25.0, version="canon-v2b")
    xa1, _, _ = C.canon_clip(kp2, w2, flags, W, H, 25.0, version="canon-v2b")
    assert np.abs(xa1 - xa0).max() > 1e-3


def test_v2c_axes_and_centring():
    kp, world, flags, W, H = synth()
    w2 = world.copy()
    w2[..., :3] = world[..., :3] @ _rot(-0.5, 0.4, 0.2).T
    x, m, _ = C.canon_clip(kp, w2, flags, W, H, 25.0, version="canon-v2c")
    v = np.median(x[:, 1] - x[:, 2], 0)
    up = np.median(x[:, 0] - 0.5 * (x[:, 1] + x[:, 2]), 0)
    assert v[0] > 0.9 and abs(v[1]) < 0.05 and abs(v[2]) < 0.05
    assert up[1] < -0.3 and abs(up[0]) < 0.05 and abs(up[2]) < 1e-3   # nose up = -y, z 0
    np.testing.assert_allclose(0.5 * (x[:, 1] + x[:, 2]), 0, atol=1e-5)


def test_v2c_invariant_to_per_frame_translation():
    kp, world, flags, W, H = synth()
    x0, _, _ = C.canon_clip(kp, world, flags, W, H, 25.0, version="canon-v2c")
    w2 = world.copy()
    w2[..., :3] += np.random.default_rng(2).normal(0, 0.05, (world.shape[0], 1, 3))
    x1, _, _ = C.canon_clip(kp, w2, flags, W, H, 25.0, version="canon-v2c")
    np.testing.assert_allclose(x1, x0, atol=1e-4)
