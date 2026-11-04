"""canon-v1: MP-L-V4V0 raw clip -> C77-3D (spec 2026-10-29 section 2). Pure numpy.

Input per clip (``extract_dataset.py``): kp (T,553,4) = pose33 | face478 | lhand21 |
rhand21 in normalised image coordinates (x, y, z, score); world (T,33,4) pose world
landmarks in metres (camera-aligned axes, hip origin); flags (T,3) = pose found,
left/right hand source (0 none). Missing parts are zeros.

Output: x (T',77,3) float32 in the RESONA-77 joint order (``foresign/groups.py``),
mask (T',77) uint8, T' = frames after resampling to 25 fps. "l_" is the signer's left
(MediaPipe LEFT_*). Steps:
  1. body9 from the world landmarks;
  2. hands / face from image landmarks made isotropic in pixels, p = (x W, y H, z W),
     scaled by s = median|world shoulder vec| / median|image shoulder vec| (m/px) and
     re-attached at the body wrist (hands, hand landmark 0) or world nose (face, mesh 1);
  3. clip frame over pose-found frames: origin = median shoulder midpoint, yaw about y
     so the median l-r shoulder vector lies on +x, unit = median shoulder width;
  4. resample to 25 fps (linear on coordinates between two valid frames, otherwise the
     nearest frame; nearest on the mask); invalid joints are 0.
"""
from __future__ import annotations

import numpy as np

CANON_VERSION = "canon-v1"
VERSIONS = ("canon-v1", "canon-v2a", "canon-v2b", "canon-v2c", "canon-v3",
            "canon-v3.1", "canon-v4")                      # spec sections 2, 6-9
V3_LIKE = ("canon-v3", "canon-v3.1", "canon-v4")  # face own scale + smoothing
CENTRED = ("canon-v2b", "canon-v2c", "canon-v3", "canon-v3.1", "canon-v4")  # per-frame centring,
                                                 # hips masked
POSE_EARS = (7, 8)                               # MediaPipe pose LEFT_EAR, RIGHT_EAR
FPS_OUT = 25.0

# raw kp layout
POSE_OFF, FACE_OFF, LHAND_OFF, RHAND_OFF = 0, 33, 511, 532
N_KP = 553

# C77 sources
BODY9 = [0, 11, 12, 13, 14, 15, 16, 23, 24]          # nose, l/r sh, l/r el, l/r wr, l/r hip
MOUTH8 = [61, 40, 0, 270, 291, 321, 17, 91]           # outer lip ring
FACE18 = [70, 105, 107, 336, 334, 300,                # brows
          33, 133, 362, 263,                          # eyes
          1, 168,                                     # nose
          234, 454, 172, 397, 152, 10]                # contour
FACE26 = MOUTH8 + FACE18
FACE_NOSE = 1                                         # face-mesh anchor

# C77 joint indices
J_NOSE, J_LSH, J_RSH, J_LWRIST, J_RWRIST = 0, 1, 2, 5, 6
J_LHAND, J_RHAND, J_FACE = 9, 30, 51
J_HIPS = [7, 8]
PARTS = {"body": slice(0, 9), "lhand": slice(9, 30), "rhand": slice(30, 51),
         "face": slice(51, 77)}

# canon-v3 (spec section 7)
SG_WIN, SG_ORDER = 7, 2                 # Savitzky-Golay after the 25 fps resample
FACE_SIZE = 0.35                        # shoulder widths per inter-ocular distance
EYE_L, EYE_R = 33, 263                  # face-mesh outer eye corners
MIN_IOD_PX = 1.0

MIN_IMG_SHOULDER_PX = 1.0     # floors for degenerate (collapsed) shoulders
MIN_WORLD_SHOULDER_M = 0.05


def metres_per_pixel(world_sh: np.ndarray, img_sh: np.ndarray) -> tuple[float, bool]:
    """s from (n,2,3) left/right shoulders in world metres and isotropic image pixels
    (pose-found frames only). Returns (s, floored)."""
    wd = float(np.median(np.linalg.norm(world_sh[:, 0] - world_sh[:, 1], axis=-1)))
    pd = float(np.median(np.linalg.norm(img_sh[:, 0] - img_sh[:, 1], axis=-1)))
    floored = pd < MIN_IMG_SHOULDER_PX or wd < MIN_WORLD_SHOULDER_M
    return max(wd, MIN_WORLD_SHOULDER_M) / max(pd, MIN_IMG_SHOULDER_PX), floored


def isotropic_px(kp_xyz: np.ndarray, width: float, height: float) -> np.ndarray:
    return kp_xyz * np.array([width, height, width], np.float32)


def attach(anchor_world: np.ndarray, part_px: np.ndarray, root_px: np.ndarray,
           s: float) -> np.ndarray:
    """anchor (T,3) + s * (part (T,K,3) - root (T,3))."""
    return anchor_world[:, None, :] + s * (part_px - root_px[:, None, :])


def clip_frame(body_world: np.ndarray, ok: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Origin (3,), rotation (3,3) and unit from the pose-found frames of body9 (T,9,3).
    Canonical coordinates are ``(x @ R.T - origin) / unit``.

    The yaw is the median per-frame azimuth of the l-r shoulder vector in the x-z plane
    (taken around the mean vector's azimuth, so no wrap-around), and the origin is the
    median shoulder midpoint in the rotated frame: both are then exactly equivariant to
    a yaw of the input (a component-wise median of raw vectors is not)."""
    ls, rs = body_world[ok, 1], body_world[ok, 2]
    v = ls - rs
    m = v.mean(0)
    ref = float(np.arctan2(m[2], m[0]))
    d = np.arctan2(v[:, 2], v[:, 0]) - ref
    a = ref + float(np.median(np.angle(np.exp(1j * d))))
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)   # azimuth a -> +x
    origin = np.median(0.5 * (ls + rs) @ R.T, axis=0)
    w = float(np.median(np.linalg.norm(v, axis=-1)))
    return origin.astype(np.float32), R, max(w, MIN_WORLD_SHOULDER_M), w < MIN_WORLD_SHOULDER_M


def _basis(vx: np.ndarray, vy: np.ndarray) -> tuple[np.ndarray, bool]:
    """Rows x^, y^, z^ = x^ x y^: x^ along vx, y^ = vy made orthogonal to x^.
    Returns (R, degenerate)."""
    nx = np.linalg.norm(vx)
    xh = vx / max(nx, 1e-9)
    y = vy - (vy @ xh) * xh
    ny = np.linalg.norm(y)
    yh = y / max(ny, 1e-9)
    return np.stack([xh, yh, np.cross(xh, yh)]).astype(np.float32), bool(nx < 1e-6 or ny < 1e-6)


def torso_frame(body_world: np.ndarray, ok: np.ndarray, down: str = "hips",
                ears_world: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
    """canon-v2 rotation (rows) from the pose-found frames of body9: x = median shoulder
    vector (r -> l, i.e. the signer's left on +x as in canon-v1), y = median "down" vector
    made orthogonal to x, z = x cross y. down="hips" (v2a/v2b): shoulder midpoint -> hip
    midpoint; down="nose" (v2c, no lower body): nose -> shoulder midpoint, i.e. minus the
    spec's "up" vector, so y points down as in every other version (the spec's
    (x, up, x cross up) frame is this one rotated 180 degrees about x); down="ears"
    (canon-v3.1, spec section 8): ear midpoint (pose world 7, 8; ears_world (T,2,3)) ->
    shoulder midpoint. The nose sits far in front of the shoulders in MediaPipe world
    coordinates (the "nose" frame is pitched ~65 degrees back); the ears are roughly in the
    shoulders' coronal plane.

    Medians are taken in a reference frame built from the means (which are rotation-
    equivariant), so the result is exactly equivariant to any rotation of the input.
    Returns (R, degenerate); on a degenerate torso the caller falls back to canon-v1."""
    b = body_world[ok]
    v = b[:, 1] - b[:, 2]
    mid = 0.5 * (b[:, 1] + b[:, 2])
    if down == "hips":
        t = 0.5 * (b[:, 7] + b[:, 8]) - mid
    elif down == "ears":
        t = mid - ears_world[ok].mean(1)
    else:
        t = mid - b[:, 0]
    R0, d0 = _basis(v.mean(0), t.mean(0))
    R1, d1 = _basis(np.median(v @ R0.T, axis=0), np.median(t @ R0.T, axis=0))
    return (R1 @ R0).astype(np.float32), d0 or d1


def _sg_matrix(win: int = SG_WIN, order: int = SG_ORDER) -> np.ndarray:
    """(win, win) least-squares polynomial projection: row i gives the smoothed value at
    position i of a window (centre row = the usual Savitzky-Golay kernel; the first / last
    rows are the edge fits, as scipy's mode='interp')."""
    t = np.arange(win) - win // 2
    V = np.vander(t, order + 1, increasing=True).astype(np.float64)
    return V @ np.linalg.pinv(V)


def _runs(valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start / end (exclusive) of the runs of True in a 1-D bool array."""
    d = np.diff(np.concatenate([[0], valid.astype(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def smooth_valid_runs(x: np.ndarray, mask: np.ndarray, win: int = SG_WIN,
                      order: int = SG_ORDER) -> np.ndarray:
    """Savitzky-Golay per joint and coordinate over runs of valid frames only. x (T,J,C),
    mask (T,J). Masked frames and runs shorter than ``win`` are returned unchanged."""
    H = _sg_matrix(win, order)
    h = win // 2
    kern = H[h][::-1]
    out = x.copy()
    T = x.shape[0]
    if T < win:
        return out
    for j in range(x.shape[1]):
        st, en = _runs(mask[:, j] > 0)
        long_ = (en - st) >= win
        st, en = st[long_], en[long_]
        if not len(st):
            continue
        xj = x[:, j].astype(np.float64)                          # (T, C)
        conv = np.stack([np.convolve(xj[:, c], kern, mode="same") for c in range(x.shape[2])], 1)
        inner = np.zeros(T, bool)                                # >= h frames from a run edge
        for a, b in zip(st, en):
            inner[a + h:b - h] = True
        out[inner, j] = conv[inner]
        w = np.arange(win)
        head = xj[st[:, None] + w]                               # (n, win, C)
        tail = xj[en[:, None] - win + w]
        out[st[:, None] + np.arange(h), j] = np.einsum("kw,nwc->nkc", H[:h], head)
        out[en[:, None] - h + np.arange(h), j] = np.einsum("kw,nwc->nkc", H[win - h:], tail)
    return out


def face_own_scale(face_px: np.ndarray, fnose_px: np.ndarray, found: np.ndarray):
    """canon-v3 face: (face26 - mesh nose) / per-clip median inter-ocular distance (mesh
    33-263, isotropic pixels), camera axes. Returns (offsets (T,26,3), iod_px, floored)."""
    i, k = FACE26.index(EYE_L), FACE26.index(EYE_R)
    if found.any():
        iod = float(np.median(np.linalg.norm(face_px[found, i] - face_px[found, k], axis=-1)))
    else:
        iod = 0.0
    return (face_px - fnose_px[:, None]) / max(iod, MIN_IOD_PX), iod, iod < MIN_IOD_PX


def _median_angle(a: np.ndarray, ref: float) -> float:
    """Circular median of angles a around ref (no wrap-around)."""
    return ref + float(np.median(np.angle(np.exp(1j * (a - ref)))))


def yaw_roll_frame(body_world: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """canon-v4 rotation (rows), spec section 9: the canon-v1 yaw about y (median azimuth
    of the l-r shoulder vector, so its median lies in the x-y plane), then a roll about z
    so the median shoulder vector lies on +x. No pitch change: the camera vertical stays
    (MediaPipe's camera-vertical is 5-9 degrees from the torso, head-based axes 47-67)."""
    _, Ry, _, _ = clip_frame(body_world, ok)
    b = body_world[ok]
    v = (b[:, 1] - b[:, 2]) @ Ry.T
    m = v.mean(0)
    r = _median_angle(np.arctan2(v[:, 1], v[:, 0]), float(np.arctan2(m[1], m[0])))
    c, s_ = np.cos(r), np.sin(r)
    Rz = np.array([[c, s_, 0], [-s_, c, 0], [0, 0, 1]], np.float32)   # angle r -> +x
    return (Rz @ Ry).astype(np.float32)


def resample_len(T: int, fps_src: float, fps_out: float = FPS_OUT) -> int:
    if T <= 0:
        return 0
    return max(1, int(np.floor(T * fps_out / fps_src + 0.5)))


def resample(x: np.ndarray, mask: np.ndarray, fps_src: float, fps_out: float = FPS_OUT):
    """x (T,J,C), mask (T,J) -> resampled to fps_out. Linear where both neighbouring
    source frames are valid for a joint, nearest otherwise; mask by nearest frame."""
    T = x.shape[0]
    n = resample_len(T, fps_src, fps_out)
    if n == 0:
        return x[:0].astype(np.float32), mask[:0].astype(np.uint8)
    u = np.minimum(np.arange(n) * (fps_src / fps_out), T - 1)
    i0 = np.floor(u).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    f = (u - i0).astype(np.float32)[:, None, None]
    near = np.floor(u + 0.5).astype(int).clip(0, T - 1)
    lin = (1 - f) * x[i0] + f * x[i1]
    both = (mask[i0] > 0) & (mask[i1] > 0)
    out = np.where(both[..., None], lin, x[near]).astype(np.float32)
    m = mask[near].astype(np.uint8)
    out[m == 0] = 0
    return out, m


def build_mask(flags: np.ndarray, face_found: np.ndarray) -> np.ndarray:
    pose = flags[:, 0] > 0
    m = np.zeros((flags.shape[0], 77), np.uint8)
    m[:, PARTS["body"]] = pose[:, None]
    m[:, PARTS["lhand"]] = (pose & (flags[:, 1] > 0))[:, None]
    m[:, PARTS["rhand"]] = (pose & (flags[:, 2] > 0))[:, None]
    m[:, PARTS["face"]] = (pose & face_found)[:, None]
    return m


def mask_rates(mask: np.ndarray) -> dict:
    if mask.shape[0] == 0:
        return {k: 0.0 for k in PARTS}
    return {k: float(mask[:, sl].mean()) for k, sl in PARTS.items()}


# the raw kp columns canon-v1 uses: pose33 | face26 + face nose | lhand21 | rhand21
SEL = (list(range(33)) + [FACE_OFF + i for i in FACE26 + [FACE_NOSE]]
       + list(range(LHAND_OFF, LHAND_OFF + 21)) + list(range(RHAND_OFF, RHAND_OFF + 21)))
S_POSE, S_FACE, S_FNOSE = slice(0, 33), slice(33, 59), 59
S_LH, S_RH = slice(60, 81), slice(81, 102)
CHUNK = 16384


def select_kp(kp, chunk: int = CHUNK) -> tuple[np.ndarray, np.ndarray]:
    """kp (T,553,4) array or h5py dataset -> (kp[:, SEL, :3] float32, face_found (T,)),
    read in time chunks so a multi-hour clip never needs the full float32 kp in memory."""
    T = kp.shape[0]
    sel = np.zeros((T, len(SEL), 3), np.float32)
    face = np.zeros(T, bool)
    for a in range(0, T, chunk):
        c = np.asarray(kp[a:a + chunk])
        sel[a:a + chunk] = np.nan_to_num(c[:, SEL, :3].astype(np.float32))
        face[a:a + chunk] = (np.nan_to_num(c[:, FACE_OFF:FACE_OFF + 478, :3]) != 0).any((1, 2))
    return sel, face


def canon_clip(kp, world, flags, width, height, fps, version: str = CANON_VERSION):
    """One clip -> (x (T',77,3) float32, mask (T',77) uint8, info dict). kp may be an
    h5py dataset (read in chunks). version: canon-v1 (yaw-only frame, clip origin),
    canon-v2a (full torso frame), canon-v2b (v2a + per-frame shoulder centring, hips
    masked out), canon-v2c (no lower body: frame from shoulders + nose, per-frame
    centring, hips masked out and never read), canon-v3 (v2c + face in its own scale +
    Savitzky-Golay smoothing after the 25 fps resample), canon-v3.1 (v3 with the up axis
    from the shoulder midpoint to the ear midpoint), canon-v4 (v3 pipeline with the v1
    orientation plus roll: yaw then roll, no pitch change)."""
    if version not in VERSIONS:
        raise ValueError(version)
    k, face_found = select_kp(kp)
    world = np.nan_to_num(np.asarray(world, np.float32))
    flags = np.asarray(flags)
    T = k.shape[0]
    fps = float(fps) if fps and fps > 0 else FPS_OUT
    n_out = resample_len(T, fps)
    info = dict(n_in=T, n_out=n_out, fps_src=fps, s=0.0, scale=0.0, yaw=0.0,
                no_pose=False, shoulder_floor=False, torso_degenerate=False, version=version)
    ok = flags[:, 0] > 0 if T else np.zeros(0, bool)
    if not ok.any():
        info["no_pose"] = True
        info["mask_rate"] = mask_rates(np.zeros((n_out, 77), np.uint8))
        return np.zeros((n_out, 77, 3), np.float32), np.zeros((n_out, 77), np.uint8), info

    body = world[:, BODY9, :3]
    px = isotropic_px(k, width, height)
    del k
    s, f1 = metres_per_pixel(world[ok][:, [11, 12], :3], px[ok][:, [11, 12]])
    lh, rh = px[:, S_LH], px[:, S_RH]
    x = np.zeros((T, 77, 3), np.float32)
    x[:, PARTS["body"]] = body
    x[:, PARTS["lhand"]] = attach(body[:, J_LWRIST], lh, lh[:, 0], s)
    x[:, PARTS["rhand"]] = attach(body[:, J_RWRIST], rh, rh[:, 0], s)
    x[:, PARTS["face"]] = attach(body[:, J_NOSE], px[:, S_FACE], px[:, S_FNOSE], s)
    if version in V3_LIKE:
        face_off, iod, iod_floor = face_own_scale(px[:, S_FACE], px[:, S_FNOSE], face_found & ok)
    del px

    origin, R, unit, f2 = clip_frame(body, ok)
    degenerate = False
    if version == "canon-v4":
        R = yaw_roll_frame(body, ok)
    elif version != "canon-v1":
        down = {"canon-v2a": "hips", "canon-v2b": "hips", "canon-v3.1": "ears"}.get(version,
                                                                                  "nose")
        Rt, degenerate = torso_frame(body, ok, down, world[:, list(POSE_EARS), :3])
        if not degenerate:
            R = Rt
            mid = 0.5 * (body[ok, 1] + body[ok, 2])
            origin = np.median(mid @ R.T, axis=0).astype(np.float32)
    x = x @ R.T
    if version in CENTRED:               # per-frame shoulder centring
        x = x - 0.5 * (x[:, J_LSH] + x[:, J_RSH])[:, None, :]
    else:
        x = x - origin
    x = x / unit
    if version in V3_LIKE:
        # face in its own scale, in the clip frame (shoulder widths): body nose (already
        # rotated, centred and divided by the unit) + 0.35 * R (p - p_nose) / IOD
        x[:, PARTS["face"]] = x[:, J_NOSE:J_NOSE + 1] + FACE_SIZE * (face_off @ R.T)
    mask = build_mask(flags, face_found)
    if version in CENTRED:               # hips: not a model input
        mask[:, J_HIPS] = 0
    x[mask == 0] = 0
    x, mask = resample(x, mask, fps)
    if version in V3_LIKE:               # after the 25 fps resample, valid runs only
        x = smooth_valid_runs(x, mask).astype(np.float32)
        info.update(iod_px=iod, iod_floor=bool(iod_floor))
    info.update(s=float(s), scale=float(unit), yaw=float(np.arctan2(R[0, 2], R[0, 0])),
                shoulder_floor=bool(f1 or f2), origin=origin.tolist(),
                torso_degenerate=degenerate, version=version, R=R.tolist(),
                mask_rate=mask_rates(mask))
    return x, mask, info


def raw77_2d(kp, flags):
    """Raw MP-L-V4V0 image (x, y) mapped to the 77 joints (no canonicalisation, native
    fps). Returns (xy (T,77,2) float32, mask (T,77) uint8)."""
    k, face_found = select_kp(kp)
    xy = np.concatenate([k[:, BODY9, :2], k[:, S_LH, :2], k[:, S_RH, :2], k[:, S_FACE, :2]],
                        axis=1)
    m = build_mask(np.asarray(flags), face_found)
    xy[m == 0] = 0
    return xy, m
