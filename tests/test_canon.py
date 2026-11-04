"""canon-v1 (spec 2026-10-29 section 2): pure-function tests on synthetic clips."""
import numpy as np
import pytest

from tools.canon import canon as C


def _yaw(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)


def synth(T=60, W=640, H=480, seed=0, hands=True, face=True):
    """A frontal signer: world pose in metres, image landmarks consistent with a
    pinhole-ish projection of 400 px per metre centred in the image."""
    rng = np.random.default_rng(seed)
    world = np.zeros((T, 33, 4), np.float32)
    base = {0: (0, -0.55, -0.1), 7: (0.07, -0.55, 0.0), 8: (-0.07, -0.55, 0.0),
            11: (0.2, -0.4, 0), 12: (-0.2, -0.4, 0),
            13: (0.25, -0.15, -0.05), 14: (-0.25, -0.15, -0.05),
            15: (0.2, 0.05, -0.2), 16: (-0.2, 0.05, -0.2), 23: (0.1, 0, 0), 24: (-0.1, 0, 0)}
    for j, p in base.items():
        world[:, j, :3] = np.array(p) + 0.01 * rng.standard_normal((T, 3))
    world[..., 3] = 1
    ppm = 400.0                                    # pixels per metre
    kp = np.zeros((T, 553, 4), np.float32)

    def img(pw):                                   # world (.., 3) -> normalised image
        return np.stack([(W / 2 + ppm * pw[..., 0]) / W, (H / 2 + ppm * pw[..., 1]) / H,
                         ppm * pw[..., 2] / W], -1)
    kp[:, :33, :3] = img(world[:, :33, :3])
    off = 0.02 * rng.standard_normal((21, 3)).astype(np.float32)
    off[0] = 0
    flags = np.zeros((T, 3), np.uint8)
    flags[:, 0] = 1
    if hands:
        # image hands sit 0.03 m away from the body wrist (re-attachment must remove this)
        kp[:, C.LHAND_OFF:C.LHAND_OFF + 21, :3] = img(world[:, 15:16, :3] + 0.03 + off)
        kp[:, C.RHAND_OFF:C.RHAND_OFF + 21, :3] = img(world[:, 16:17, :3] - 0.03 + off)
        flags[:, 1] = 1
        flags[:, 2] = 2
    if face:
        fo = 0.05 * rng.standard_normal((478, 3)).astype(np.float32)
        fo[1] = 0
        kp[:, C.FACE_OFF:C.FACE_OFF + 478, :3] = img(world[:, 0:1, :3] + 0.01 + fo)
    kp[..., 3] = 0.9
    return kp, world, flags, W, H


def test_layout_constants():
    assert (C.POSE_OFF, C.FACE_OFF, C.LHAND_OFF, C.RHAND_OFF) == (0, 33, 511, 532)
    assert C.BODY9 == [0, 11, 12, 13, 14, 15, 16, 23, 24]
    assert C.MOUTH8 == [61, 40, 0, 270, 291, 321, 17, 91]
    assert len(C.FACE18) == 18 and C.FACE18[:6] == [70, 105, 107, 336, 334, 300]
    assert C.FACE18[-6:] == [234, 454, 172, 397, 152, 10]


def test_shapes_and_resample_length():
    kp, world, flags, W, H = synth(T=60)
    x, m, info = C.canon_clip(kp, world, flags, W, H, fps=30.0)
    assert x.shape == (50, 77, 3) and x.dtype == np.float32
    assert m.shape == (50, 77) and m.dtype == np.uint8
    assert info["n_in"] == 60 and info["n_out"] == 50 and info["fps_src"] == 30.0
    assert not info["no_pose"]


def test_resample_25_is_identity_length():
    assert C.resample_len(100, 25.0) == 100
    assert C.resample_len(60, 30.0) == 50
    assert C.resample_len(0, 30.0) == 0


def test_resample_linear_and_nearest_mask():
    T = 6
    x = np.arange(T, dtype=np.float32)[:, None, None] * np.ones((T, 2, 3), np.float32)
    m = np.ones((T, 2), np.uint8)
    m[3, 1] = 0
    xr, mr = C.resample(x, m, 50.0, 25.0)            # every other frame
    assert xr.shape[0] == 3
    np.testing.assert_allclose(xr[:, 0, 0], [0, 2, 4])
    xr, mr = C.resample(x, m, 20.0, 25.0)            # upsample: 6 frames @20 -> 8 @25 (0.3 s)
    assert xr.shape[0] == 8
    np.testing.assert_allclose(xr[1, 0, 0], 0.8, atol=1e-5)   # t=0.04 s -> src 0.8
    # joint 1 is invalid at frame 3: nearest frame in mask; invalid joints are zero
    assert ((mr == 0) <= (np.abs(xr).sum(-1) == 0)).all()


def test_frame_is_canonical():
    kp, world, flags, W, H = synth()
    x, m, info = C.canon_clip(kp, world, flags, W, H, fps=25.0)
    mid = 0.5 * (x[:, 1] + x[:, 2])
    width = np.linalg.norm(x[:, 1] - x[:, 2], axis=-1)
    np.testing.assert_allclose(np.median(mid, 0), 0, atol=0.05)
    np.testing.assert_allclose(np.median(width), 1, atol=0.05)
    v = np.median(x[:, 1] - x[:, 2], 0)            # l_sh - r_sh on +x
    assert v[0] > 0.9 and abs(v[2]) < 0.05


def test_translation_scale_yaw_invariance():
    kp, world, flags, W, H = synth()
    x0, m0, _ = C.canon_clip(kp, world, flags, W, H, fps=25.0)
    # world: scale by 1.3, yaw by 40 degrees, translate; image: zoomed-out camera
    w2 = world.copy()
    w2[..., :3] = 1.3 * world[..., :3] @ _yaw(np.deg2rad(40)).T + np.array([0.5, -0.2, 1.0])
    kp2 = kp.copy()
    kp2[..., 0] = 0.1 + 0.5 * kp[..., 0]
    kp2[..., 1] = 0.2 + 0.5 * kp[..., 1]
    kp2[..., 2] = 0.5 * kp[..., 2]
    x1, m1, _ = C.canon_clip(kp2, w2, flags, W, H, fps=25.0)
    np.testing.assert_array_equal(m0, m1)
    np.testing.assert_allclose(x1[:, :9], x0[:, :9], atol=1e-4)
    # hands/face offsets come from the image and are camera-aligned: translation and
    # scale must cancel exactly (yaw is a body-only test, so check a pure scale+shift)
    w3 = world.copy()
    w3[..., :3] = 1.3 * world[..., :3] + np.array([0.5, -0.2, 1.0])
    x3, _, _ = C.canon_clip(kp2, w3, flags, W, H, fps=25.0)
    np.testing.assert_allclose(x3, x0, atol=1e-4)


def test_hand_reattached_at_body_wrist():
    kp, world, flags, W, H = synth()
    x, m, info = C.canon_clip(kp, world, flags, W, H, fps=25.0)
    np.testing.assert_allclose(x[:, C.J_LHAND], x[:, C.J_LWRIST], atol=1e-5)
    np.testing.assert_allclose(x[:, C.J_RHAND], x[:, C.J_RWRIST], atol=1e-5)
    # metres-per-pixel recovers the synthetic 1/400 m/px
    assert info["s"] == pytest.approx(1 / 400, rel=0.02)
    # face nose (face mesh 1 is not in face18, but the nose joint stays the world nose):
    # face offsets are scaled by s; contour points stay within a head-size of the nose
    assert np.abs(x[:, 59:77] - x[:, 0:1]).max() < 1.5


def test_left_right_order():
    kp, world, flags, W, H = synth()
    x, m, _ = C.canon_clip(kp, world, flags, W, H, fps=25.0)
    # signer's left shoulder/wrist/hand on +x, right on -x
    assert x[:, 1, 0].mean() > 0 > x[:, 2, 0].mean()
    assert x[:, 5, 0].mean() > 0 > x[:, 6, 0].mean()
    assert x[:, 9:30, 0].mean() > 0 > x[:, 30:51, 0].mean()


def test_mask_and_zeros_missing_parts():
    kp, world, flags, W, H = synth(hands=False, face=False)
    flags[10:20, 0] = 0
    world[10:20] = 0
    kp[10:20, :33] = 0
    x, m, info = C.canon_clip(kp, world, flags, W, H, fps=25.0)
    assert m[:, 9:].sum() == 0 and np.abs(x[:, 9:]).sum() == 0
    assert m[10:20].sum() == 0 and np.abs(x[10:20]).sum() == 0
    assert m[:10, :9].all() and m[20:, :9].all()
    assert info["mask_rate"]["lhand"] == 0 and info["mask_rate"]["body"] == pytest.approx(50 / 60)


def test_one_hand_missing_frames():
    kp, world, flags, W, H = synth()
    flags[:30, 1] = 0
    kp[:30, C.LHAND_OFF:C.LHAND_OFF + 21] = 0
    x, m, _ = C.canon_clip(kp, world, flags, W, H, fps=25.0)
    assert m[:30, 9:30].sum() == 0 and m[30:, 9:30].all() and m[:, 30:51].all()


def test_no_pose():
    kp, world, flags, W, H = synth()
    flags[:, 0] = 0
    x, m, info = C.canon_clip(kp, world, flags, W, H, fps=30.0)
    assert info["no_pose"] and m.sum() == 0 and np.abs(x).sum() == 0
    assert x.shape == (50, 77, 3)


def test_empty_clip():
    kp, world, flags, W, H = synth(T=1)
    x, m, info = C.canon_clip(kp[:0], world[:0], flags[:0], W, H, fps=30.0)
    assert x.shape == (0, 77, 3) and m.shape == (0, 77) and info["no_pose"]


def test_zero_shoulder_width_floored():
    kp, world, flags, W, H = synth()
    world[:, 12, :3] = world[:, 11, :3]
    kp[:, 12, :3] = kp[:, 11, :3]
    x, m, info = C.canon_clip(kp, world, flags, W, H, fps=25.0)
    assert info["shoulder_floor"]
    assert np.isfinite(x).all() and np.abs(x).max() < 1e3


def test_float16_input():
    kp, world, flags, W, H = synth()
    x, m, _ = C.canon_clip(kp.astype(np.float16), world.astype(np.float16), flags, W, H, 25.0)
    assert np.isfinite(x).all()


def test_raw77_2d():
    kp, world, flags, W, H = synth()
    xy, m = C.raw77_2d(kp, flags)
    assert xy.shape == (60, 77, 2) and m.shape == (60, 77)
    np.testing.assert_allclose(xy[:, 1], kp[:, 11, :2])
    np.testing.assert_allclose(xy[:, 9], kp[:, C.LHAND_OFF, :2])
    np.testing.assert_allclose(xy[:, 51], kp[:, C.FACE_OFF + 61, :2])
    assert m.all()


def test_select_kp_chunked_matches_whole():
    kp, world, flags, W, H = synth(T=50)
    kp[5:9, C.FACE_OFF:C.FACE_OFF + 478] = 0
    a, fa = C.select_kp(kp, chunk=7)
    b, fb = C.select_kp(kp, chunk=1000)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(fa, fb)
    assert not fa[5:9].any() and fa[:5].all()
    x, m, _ = C.canon_clip(kp, world, flags, W, H, 25.0)
    assert m[5:9, 51:].sum() == 0 and m[10, 51:].all()
