"""canon-v3 (spec section 7) and canon-v3.1 (section 8: ear-midpoint up axis)."""
import numpy as np
import pytest

from tests.test_canon import synth
from tests.test_canon_v2 import _rot
from tools.canon import canon as C

V3S = pytest.mark.parametrize("ver", ["canon-v3", "canon-v3.1", "canon-v4"])


def test_sg_matches_scipy_interp():
    signal = pytest.importorskip("scipy.signal")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 2, 3)).astype(np.float32)
    m = np.ones((40, 2), np.uint8)
    y = C.smooth_valid_runs(x, m)
    ref = signal.savgol_filter(x.astype(np.float64), 7, 2, axis=0, mode="interp")
    np.testing.assert_allclose(y, ref, atol=1e-5)


def test_sg_runs_masked_and_short_untouched():
    rng = np.random.default_rng(1)
    T = 60
    x = rng.standard_normal((T, 1, 3)).astype(np.float32)
    m = np.ones((T, 1), np.uint8)
    m[10:12] = 0            # run 0..9 (10 frames), gap
    m[17:20] = 0            # run 12..16 (5 frames: too short)
    x[m == 0] = 0
    y = C.smooth_valid_runs(x, m)
    np.testing.assert_array_equal(y[m[:, 0] == 0], x[m[:, 0] == 0])      # masked untouched
    np.testing.assert_array_equal(y[12:17], x[12:17])                    # short run untouched
    assert np.abs(y[0:10] - x[0:10]).max() > 0                           # long runs smoothed
    assert np.abs(y[20:] - x[20:]).max() > 0
    # a valid run is smoothed on its own: changing values outside it does not affect it
    x2 = x.copy()
    x2[12:17] += 100
    np.testing.assert_allclose(C.smooth_valid_runs(x2, m)[20:], y[20:], atol=1e-5)
    np.testing.assert_allclose(C.smooth_valid_runs(x2, m)[:10], y[:10], atol=1e-5)


def test_sg_keeps_quadratics_and_reduces_jitter():
    t = np.arange(200, dtype=np.float64)
    q = (0.001 * t ** 2 - 0.3 * t + 2)[:, None, None] * np.ones((1, 1, 3))
    m = np.ones((200, 1), np.uint8)
    np.testing.assert_allclose(C.smooth_valid_runs(q, m), q, atol=1e-8)
    truth = np.sin(t / 15)[:, None, None] * np.ones((1, 1, 3))
    noisy = truth + np.random.default_rng(0).normal(0, 0.05, truth.shape)
    y = C.smooth_valid_runs(noisy, m)
    assert np.sqrt(((y - truth) ** 2).mean()) < 0.7 * np.sqrt(((noisy - truth) ** 2).mean())
    acc = lambda a: np.abs(np.diff(a, 2, axis=0)).mean()  # noqa: E731
    assert acc(y) < 0.3 * acc(noisy)


def _scale_face(kp, k, shift=(0.0, 0.0)):
    """Face k times larger in the image (closer / bigger head), about the mesh nose."""
    kp = kp.copy()
    f = kp[:, C.FACE_OFF:C.FACE_OFF + 478, :3]
    nose = f[:, C.FACE_NOSE:C.FACE_NOSE + 1].copy()
    f[:] = nose + k * (f - nose)
    f[..., 0] += shift[0]
    f[..., 1] += shift[1]
    return kp


@V3S
def test_face_independent_of_face_scale_and_distance(ver):
    kp, world, flags, W, H = synth()
    x0, m0, info = C.canon_clip(kp, world, flags, W, H, 25.0, version=ver)
    x1, m1, _ = C.canon_clip(_scale_face(kp, 1.7, (0.05, -0.02)), world, flags, W, H, 25.0,
                             version=ver)
    np.testing.assert_array_equal(m0, m1)
    np.testing.assert_allclose(x1, x0, atol=1e-4)
    assert info["iod_px"] > 0 and not info["iod_floor"]
    # v2c does see the face size
    y0, _, _ = C.canon_clip(kp, world, flags, W, H, 25.0, version="canon-v2c")
    y1, _, _ = C.canon_clip(_scale_face(kp, 1.7), world, flags, W, H, 25.0, version="canon-v2c")
    assert np.abs(y1[:, 51:] - y0[:, 51:]).max() > 0.05


@V3S
def test_face_size_fixed(ver):
    kp, world, flags, W, H = synth()
    x, m, _ = C.canon_clip(kp, world, flags, W, H, 25.0, version=ver)
    i, k = 51 + C.FACE26.index(C.EYE_L), 51 + C.FACE26.index(C.EYE_R)
    iod = np.median(np.linalg.norm(x[:, i] - x[:, k], axis=-1))
    assert iod == pytest.approx(C.FACE_SIZE, rel=0.05)


@pytest.mark.parametrize("ver", ["canon-v3", "canon-v3.1"])
@pytest.mark.parametrize("ang", [(0.7, 0, 0), (0, 0.5, 0), (0, 0, 0.4), (0.6, -0.4, 0.3)])
def test_v3_rotation_invariance_body(ver, ang):
    kp, world, flags, W, H = synth()
    x0, m0, _ = C.canon_clip(kp, world, flags, W, H, 25.0, version=ver)
    w2 = world.copy()
    w2[..., :3] = 1.2 * world[..., :3] @ _rot(*ang).T + np.array([0.3, -0.5, 2.0])
    x1, m1, _ = C.canon_clip(kp, w2, flags, W, H, 25.0, version=ver)
    np.testing.assert_array_equal(m0, m1)
    np.testing.assert_allclose(x1[:, :9], x0[:, :9], atol=1e-4)


@V3S
def test_v3_v2c_properties(ver):
    kp, world, flags, W, H = synth()
    x0, m0, _ = C.canon_clip(kp, world, flags, W, H, 25.0, version=ver)
    assert m0[:, 7:9].sum() == 0 and np.abs(x0[:, 7:9]).sum() == 0
    w2 = world.copy()
    w2[:, 23:25, :3] = 5.0
    w2[..., :3] += np.random.default_rng(2).normal(0, 0.05, (world.shape[0], 1, 3))
    x1, _, _ = C.canon_clip(kp, w2, flags, W, H, 25.0, version=ver)
    np.testing.assert_allclose(x1, x0, atol=1e-4)          # hips unread, per-frame centring
    # translation + scale of the whole image / world still cancel (hands and face)
    kp2 = kp.copy()
    kp2[..., 0] = 0.1 + 0.5 * kp[..., 0]
    kp2[..., 1] = 0.2 + 0.5 * kp[..., 1]
    kp2[..., 2] = 0.5 * kp[..., 2]
    w3 = world.copy()
    w3[..., :3] = 1.3 * world[..., :3] + 0.4
    x3, _, _ = C.canon_clip(kp2, w3, flags, W, H, 25.0, version=ver)
    np.testing.assert_allclose(x3, x0, atol=1e-4)
    np.testing.assert_allclose(x0[:, C.J_LHAND], x0[:, C.J_LWRIST], atol=1e-5)


@V3S
def test_v3_smoothing_applied_and_mask_respected(ver):
    kp, world, flags, W, H = synth(T=100)
    flags[40:45, 1] = 0
    kp[40:45, C.LHAND_OFF:C.LHAND_OFF + 21] = 0
    x, m, _ = C.canon_clip(kp, world, flags, W, H, 25.0, version=ver)
    xc, mc, _ = C.canon_clip(kp, world, flags, W, H, 25.0, version="canon-v2c")
    np.testing.assert_array_equal(m, mc)
    assert np.abs(x[m == 0]).sum() == 0
    acc = lambda a: np.abs(np.diff(a[:, 1:7], 2, axis=0)).mean()  # noqa: E731
    assert acc(x) < 0.5 * acc(xc)                         # synth body has white jitter


def _upright(T=60, seed=0):
    """An upright signer as MediaPipe world sees it: ears above the shoulders in their
    coronal plane, the nose well in front (z -0.3) and only a little above the shoulders."""
    kp, world, flags, W, H = synth(T=T, seed=seed)
    rng = np.random.default_rng(seed + 7)
    for j, p in {0: (0, -0.48, -0.3), 7: (0.08, -0.58, 0.0), 8: (-0.08, -0.58, 0.0)}.items():
        world[:, j, :3] = np.array(p) + 0.005 * rng.standard_normal((T, 3))
    return kp, world, flags, W, H


def _up_vs_torso_deg(info, world):
    """Angle between the canonical down axis (row 1 of R) and the anatomical torso
    direction (shoulder midpoint -> hip midpoint) in world coordinates."""
    R = np.asarray(info["R"])
    t = 0.5 * (world[:, 23, :3] + world[:, 24, :3]) - 0.5 * (world[:, 11, :3] + world[:, 12, :3])
    t = np.median(t, 0)
    return np.degrees(np.arccos(np.clip(R[1] @ t / np.linalg.norm(t), -1, 1)))


def test_v31_upright_body_stays_upright():
    kp, world, flags, W, H = _upright()
    x, m, info = C.canon_clip(kp, world, flags, W, H, 25.0, version="canon-v3.1")
    assert _up_vs_torso_deg(info, world) < 5
    # in canonical coordinates the arms hang down (+y), not along z (no backward tilt)
    wr = np.median(0.5 * (x[:, 5] + x[:, 6]), 0)
    assert wr[1] > 0.3 and abs(wr[2]) < 0.6 * wr[1]
    nose = np.median(x[:, 0], 0)
    assert nose[1] < -0.2 and nose[2] < -0.3          # head up, nose in front (-z)
    # canon-v3 (nose up axis) is tilted back by the head-forward angle
    _, _, i3 = C.canon_clip(kp, world, flags, W, H, 25.0, version="canon-v3")
    assert _up_vs_torso_deg(i3, world) > 30


def test_v31_rotation_invariance_full_axes():
    kp, world, flags, W, H = _upright()
    x0, _, _ = C.canon_clip(kp, world, flags, W, H, 25.0, version="canon-v3.1")
    w2 = world.copy()
    w2[..., :3] = world[..., :3] @ _rot(0.5, -0.6, 0.3).T
    x1, _, info = C.canon_clip(kp, w2, flags, W, H, 25.0, version="canon-v3.1")
    np.testing.assert_allclose(x1[:, :9], x0[:, :9], atol=1e-4)
    assert _up_vs_torso_deg(info, w2) < 5


def test_v31_reads_ears_not_hips():
    kp, world, flags, W, H = _upright()
    x0, _, _ = C.canon_clip(kp, world, flags, W, H, 25.0, version="canon-v3.1")
    w2 = world.copy()
    w2[:, 23:25, :3] = 9.0
    x1, _, _ = C.canon_clip(kp, w2, flags, W, H, 25.0, version="canon-v3.1")
    np.testing.assert_allclose(x1, x0, atol=1e-6)
    w3 = world.copy()
    w3[:, 7:9, 2] -= 0.3                                # ears moved forward: frame pitches
    x3, _, _ = C.canon_clip(kp, w3, flags, W, H, 25.0, version="canon-v3.1")
    assert np.abs(x3[:, :7] - x0[:, :7]).max() > 0.05


def _torso_pitch_deg(info, world):
    """Angle of the anatomical torso (shoulder mid -> hip mid) to +y after the clip rotation,
    measured in the y-z plane (positive = hips behind)."""
    R = np.asarray(info["R"])
    t = np.median(0.5 * (world[:, 23, :3] + world[:, 24, :3])
                  - 0.5 * (world[:, 11, :3] + world[:, 12, :3]), 0) @ R.T
    return np.degrees(np.arctan2(t[2], t[1]))


@pytest.mark.parametrize("ang", [(0.7, 0, 0), (0, 0, 0.4), (-0.5, 0, 0.0), (0, 0, -0.3)])
def test_v4_yaw_roll_invariance(ang):
    kp, world, flags, W, H = _upright()
    x0, m0, _ = C.canon_clip(kp, world, flags, W, H, 25.0, version="canon-v4")
    w2 = world.copy()
    w2[..., :3] = 1.2 * world[..., :3] @ _rot(*ang).T + np.array([0.3, -0.5, 2.0])
    x1, m1, _ = C.canon_clip(kp, w2, flags, W, H, 25.0, version="canon-v4")
    np.testing.assert_array_equal(m0, m1)
    np.testing.assert_allclose(x1[:, :9], x0[:, :9], atol=2e-3)


def test_v4_yaw_plus_roll_combined():
    kp, world, flags, W, H = _upright()
    x0, _, _ = C.canon_clip(kp, world, flags, W, H, 25.0, version="canon-v4")
    w2 = world.copy()
    w2[..., :3] = world[..., :3] @ _rot(0.5, 0, 0.25).T
    x1, _, _ = C.canon_clip(kp, w2, flags, W, H, 25.0, version="canon-v4")
    # yaw + roll of a camera are recovered up to a rotation about the shoulder (x) axis,
    # i.e. the pitch that v4 keeps by design: x and the y-z radius of every joint match
    np.testing.assert_allclose(x1[:, :9, 0], x0[:, :9, 0], atol=2e-2)
    np.testing.assert_allclose(np.linalg.norm(x1[:, :9, 1:], axis=-1),
                               np.linalg.norm(x0[:, :9, 1:], axis=-1), atol=2e-2)


def test_v4_keeps_camera_pitch():
    kp, world, flags, W, H = _upright()
    _, _, i0 = C.canon_clip(kp, world, flags, W, H, 25.0, version="canon-v4")
    p0 = _torso_pitch_deg(i0, world)
    assert abs(p0) < 3 and _up_vs_torso_deg(i0, world) < 3       # upright body stays upright
    w2 = world.copy()
    w2[..., :3] = world[..., :3] @ _rot(0, 0.35, 0).T          # camera pitched 20 degrees
    x1, _, i1 = C.canon_clip(kp, w2, flags, W, H, 25.0, version="canon-v4")
    assert abs(abs(_torso_pitch_deg(i1, w2) - p0) - np.degrees(0.35)) < 2  # pitch kept
    v = np.median(x1[:, 1] - x1[:, 2], 0)
    assert v[0] > 0.95 and abs(v[1]) < 0.02 and abs(v[2]) < 0.02          # shoulders on +x


def test_v4_removes_shoulder_tilt():
    kp, world, flags, W, H = _upright()
    w2 = world.copy()
    w2[..., :3] = world[..., :3] @ _rot(0.3, 0, 0.3).T          # yaw + roll
    x, m, _ = C.canon_clip(kp, w2, flags, W, H, 25.0, version="canon-v4")
    sh = x[:, 1] - x[:, 2]
    tilt = np.degrees(np.arccos(np.clip(sh[:, 0] / np.linalg.norm(sh, axis=-1), -1, 1)))
    assert np.median(tilt) < 2
    np.testing.assert_allclose(0.5 * (x[:, 1] + x[:, 2]), 0, atol=1e-5)  # per-frame centring


def test_v4_hips_never_read():
    kp, world, flags, W, H = _upright()
    x0, m0, _ = C.canon_clip(kp, world, flags, W, H, 25.0, version="canon-v4")
    w2 = world.copy()
    w2[:, 23:25, :3] = 7.0
    x1, _, _ = C.canon_clip(kp, w2, flags, W, H, 25.0, version="canon-v4")
    np.testing.assert_allclose(x1, x0, atol=1e-6)
    assert m0[:, 7:9].sum() == 0
