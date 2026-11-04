"""Extractor wrappers producing the common (T, K, 4) output (see common.py).

MP-L   legacy mediapipe.solutions.holistic (model_complexity=2, refine_face_landmarks,
       smooth_landmarks, video mode).  kp layout mp553; world = pose_world 33.
       Variants MP-L-<V> (MPL_VARIANTS): thresholds, bicubic upscale, hand-crop
       fallback with mediapipe Hands; extra['hand_src'] (T,2) 0 none/1 holistic/2 crop.
MP-T   mediapipe Tasks HolisticLandmarker, VIDEO running mode. kp layout mp553;
       world = pose_world 33 | left hand world 21 | right hand world 21.
RTMW3D RTMW3D-x (rtmlib ONNX, cocktail14 384x288) + YOLOX-m HumanArt detector.
       kp layout coco133 with x,y image-normalised from the 2D head and z = the model's
       root-relative metric depth; world = raw 3D output (x,y in crop-input pixels,
       z metres root-relative) + per-frame bbox in extra['bbox'].

Scores: MP pose = visibility; MP hands/face = 1 if detected else 0;
        RTMW3D = SimCC confidence per keypoint.
"""
from __future__ import annotations

import os
import time

import numpy as np

from tools.extract.common import ClipResult, config_hash, iter_frames

MP_K = 553
MPT_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
                 "holistic_landmarker/float16/latest/holistic_landmarker.task")


def _fill(arr, t, off, lms, n, with_vis=False, world=False):
    """Write a landmark list into arr[t, off:off+n]."""
    if lms is None:
        return False
    pts = lms.landmark if hasattr(lms, "landmark") else lms
    if len(pts) == 0:
        return False
    m = min(n, len(pts))
    for i in range(m):
        p = pts[i]
        vis = getattr(p, "visibility", None)
        s = float(vis) if (with_vis and vis is not None) else 1.0
        arr[t, off + i] = (p.x, p.y, p.z, s)
    return True


# MP-L hand-improvement variants (docs/mp_hand_improvement.md). All share
# model_complexity=2, refine_face_landmarks, video mode.
MPL_VARIANTS = {
    "V0": dict(det=0.5, track=0.5, upscale=0, fallback=False, smooth=True),
    "V1": dict(det=0.3, track=0.3, upscale=0, fallback=False, smooth=True),
    "V2": dict(det=0.5, track=0.5, upscale=720, fallback=False, smooth=True),
    "V3": dict(det=0.3, track=0.3, upscale=720, fallback=False, smooth=True),
    "V5": dict(det=0.5, track=0.5, upscale=0, fallback=False, smooth=False),
}
# V4 = hand-crop fallback on top of the best of V0-V3: "V4" + base, e.g. "V4V1".
for _b in ("V0", "V1", "V2", "V3"):
    MPL_VARIANTS["V4" + _b] = dict(MPL_VARIANTS[_b], fallback=True)
# V6 (post hoc, added after V4V0 results): V4V0 without the pose-visibility gate.
MPL_VARIANTS["V6"] = dict(MPL_VARIANTS["V0"], fallback=True, fb_vis=0.0)

# Hand-crop fallback geometry (fixed before results; see docs/mp_hand_improvement.md).
FB_VIS = 0.5          # pose wrist and elbow visibility needed to attempt a crop
FB_CENTER = 0.3       # crop centre = wrist + FB_CENTER * (wrist - elbow)
FB_SIDE = 2.0         # crop side = FB_SIDE * forearm length (px)
FB_MIN_FOREARM = 0.6  # forearm length floor, x shoulder width (foreshortened arms)
FB_GATE = 1.0         # accept the nearest hand only if its wrist <= FB_GATE * forearm away
FB_DUP = 0.25         # reject if within FB_DUP * forearm of the other hand's wrist
FB_INPUT = 256        # crop is resized to FB_INPUT^2 before mediapipe Hands
HAND_SRC_NONE, HAND_SRC_HOLISTIC, HAND_SRC_CROP = 0, 1, 2


def upscale_frame(img, min_side):
    """Bicubic upscale so the short side is >= min_side (no-op if already that large)."""
    import cv2
    H, W = img.shape[:2]
    if not min_side or min(H, W) >= min_side:
        return img
    s = min_side / min(H, W)
    return cv2.resize(img, (int(round(W * s)), int(round(H * s))), interpolation=cv2.INTER_CUBIC)


def hand_crop_box(wr, el, sh_w):
    """Square crop (x0, y0, side) in px around the pose-predicted hand, plus the
    effective forearm length L. wr, el: (2,) px; sh_w: shoulder width px."""
    f = np.asarray(wr, np.float64) - np.asarray(el, np.float64)
    L = max(float(np.linalg.norm(f)), FB_MIN_FOREARM * float(sh_w), 8.0)
    c = np.asarray(wr, np.float64) + FB_CENTER * f
    side = FB_SIDE * L
    return c[0] - side / 2, c[1] - side / 2, side, L


def crop_square(img, x0, y0, side, out):
    """Crop [x0, x0+side) x [y0, y0+side) with zero padding, resized to out x out.
    Returns the crop and its integer origin/side in px."""
    import cv2
    H, W = img.shape[:2]
    xi, yi, si = int(np.floor(x0)), int(np.floor(y0)), max(int(np.ceil(side)), 2)
    pad = max(0, -xi, -yi, xi + si - W, yi + si - H)
    src = img if pad == 0 else cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
    c = src[yi + pad:yi + pad + si, xi + pad:xi + pad + si]
    interp = cv2.INTER_CUBIC if si < out else cv2.INTER_AREA
    return cv2.resize(c, (out, out), interpolation=interp), xi, yi, si


def map_crop_landmarks(pts, xi, yi, si, W, H):
    """Crop-normalised (n,3) hand landmarks -> full-frame normalised x, y, with z in
    the holistic convention (scaled like x, i.e. by image width)."""
    pts = np.asarray(pts, np.float64)
    return np.stack([(xi + pts[:, 0] * si) / W, (yi + pts[:, 1] * si) / H,
                     pts[:, 2] * si / W], 1)


def choose_crop_hand(cands_px, wr, other_wr, other_hand_wr, L):
    """Pick the crop hand for one side. cands_px: list of (21,2) px arrays.
    Returns an index or None. Gates: nearest wrist to this side's pose wrist, within
    FB_GATE*L, not closer to the other pose wrist, not a duplicate of the other hand."""
    if not cands_px:
        return None
    d = [float(np.linalg.norm(c[0] - wr)) for c in cands_px]
    i = int(np.argmin(d))
    w0 = cands_px[i][0]
    if d[i] > FB_GATE * L:
        return None
    if other_wr is not None and np.linalg.norm(w0 - other_wr) < d[i]:
        return None
    if other_hand_wr is not None and np.linalg.norm(w0 - other_hand_wr) < FB_DUP * L:
        return None
    return i


class MPLegacy:
    name = "MP-L"
    layout = "mp553"
    world_layout = "pose_world33"
    CFG = dict(model_complexity=2, refine_face_landmarks=True, static_image_mode=False,
               smooth_landmarks=True, enable_segmentation=False,
               min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def __init__(self, variant: str = "V0"):
        import mediapipe as mp
        self.version = mp.__version__
        self._mp = mp
        v = MPL_VARIANTS[variant]
        self.variant = variant
        self.upscale, self.crop = v["upscale"], v["fallback"]
        self.fb_vis = v.get("fb_vis", FB_VIS)
        self.holistic_cfg = dict(MPLegacy.CFG, min_detection_confidence=v["det"],
                                 min_tracking_confidence=v["track"], smooth_landmarks=v["smooth"])
        self.CFG = dict(self.holistic_cfg)
        if variant != "V0":  # V0 keeps the bake-off name and config hash
            self.name = f"MP-L-{variant}"
            self.CFG.update(upscale_min_side=self.upscale, hand_crop=self.crop)
            if self.crop:
                self.CFG.update(crop=dict(vis=self.fb_vis, center=FB_CENTER, side=FB_SIDE,
                                          min_forearm=FB_MIN_FOREARM, gate=FB_GATE, dup=FB_DUP,
                                          input=FB_INPUT,
                                          hands="static,complexity1,max2,det0.5"))
        self._hands = None
        if self.crop:
            self._hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2,
                                                   model_complexity=1,
                                                   min_detection_confidence=0.5)

    def _crop_hands(self, img, kp, src):
        """Fill hands that holistic missed in one frame (kp (553,4) normalised, in
        place) by running mediapipe Hands on a crop around the pose-predicted hand."""
        H, W = img.shape[:2]
        P = kp[:33, :2] * (W, H)
        vis = kp[:33, 3]
        sh_w = float(np.linalg.norm(P[11] - P[12]))
        # side: (pose wrist, pose elbow, hand offset, other pose wrist, other hand offset, k)
        for iw, ie, off, iow, ooff, k in ((15, 13, 511, 16, 532, 0), (16, 14, 532, 15, 511, 1)):
            if src[k] != HAND_SRC_NONE or vis[iw] < self.fb_vis or vis[ie] < self.fb_vis:
                continue
            if not (0 <= kp[iw, 0] <= 1 and 0 <= kp[iw, 1] <= 1):
                continue
            x0, y0, side, L = hand_crop_box(P[iw], P[ie], sh_w)
            crop, xi, yi, si = crop_square(img, x0, y0, side, FB_INPUT)
            r = self._hands.process(crop[:, :, ::-1].copy())
            if not r.multi_hand_landmarks:
                continue
            full = [map_crop_landmarks([(p.x, p.y, p.z) for p in h.landmark], xi, yi, si, W, H)
                    for h in r.multi_hand_landmarks]
            other_wr = P[iow] if vis[iow] >= FB_VIS else None
            other_hand = kp[ooff, :2] * (W, H) if kp[ooff, 3] > 0 else None
            i = choose_crop_hand([f[:, :2] * (W, H) for f in full], P[iw], other_wr,
                                 other_hand, L)
            if i is None:
                continue
            kp[off:off + 21, :3] = full[i]
            kp[off:off + 21, 3] = 1.0
            src[k] = HAND_SRC_CROP

    def run(self, path: str, fps: float, max_frames: int | None = None) -> ClipResult:
        h = self._mp.solutions.holistic.Holistic(**self.holistic_cfg)
        kps, wos, val, srcs = [], [], [], []
        t0 = time.time()
        W = H = PW = PH = 0
        try:
            for img in iter_frames(path, max_frames):
                H, W = img.shape[:2]
                img = upscale_frame(img, self.upscale)
                PH, PW = img.shape[:2]
                r = h.process(img[:, :, ::-1].copy())
                kp = np.zeros((1, MP_K, 4), np.float32)
                wo = np.zeros((1, 33, 4), np.float32)
                okp = _fill(kp, 0, 0, r.pose_landmarks, 33, with_vis=True)
                _fill(kp, 0, 33, r.face_landmarks, 478)
                src = np.array([_fill(kp, 0, 511, r.left_hand_landmarks, 21),
                                _fill(kp, 0, 532, r.right_hand_landmarks, 21)], np.uint8)
                _fill(wo, 0, 0, r.pose_world_landmarks, 33, with_vis=True)
                if self.crop and okp and (src == HAND_SRC_NONE).any():
                    self._crop_hands(img, kp[0], src)
                kps.append(kp[0]); wos.append(wo[0]); val.append(1 if okp else 0); srcs.append(src)
        finally:
            h.close()
        el = time.time() - t0
        T = len(kps)
        attrs = dict(fps=fps, width=W, height=H, n_frames=T, extractor=self.name,
                     extractor_version=f"mediapipe=={self.version}",
                     config_hash=config_hash(self.CFG), layout=self.layout,
                     world_layout=self.world_layout, elapsed_s=el)
        extra = {}
        if self.variant != "V0":
            attrs.update(variant=self.variant, proc_width=PW, proc_height=PH)
            extra["hand_src"] = np.stack(srcs) if T else np.zeros((0, 2), np.uint8)
        return ClipResult(np.stack(kps) if T else np.zeros((0, MP_K, 4), np.float32),
                          np.stack(wos) if T else np.zeros((0, 33, 4), np.float32),
                          np.asarray(val, np.uint8), attrs, extra)


class MPTasks:
    name = "MP-T"
    layout = "mp553"
    world_layout = "pose_world33+lhand_world21+rhand_world21"
    CFG = dict(running_mode="VIDEO", min_face_detection_confidence=0.5,
               min_face_suppression_threshold=0.3, min_face_landmarks_confidence=0.5,
               min_pose_detection_confidence=0.5, min_pose_suppression_threshold=0.3,
               min_pose_landmarks_confidence=0.5, min_hand_landmarks_confidence=0.5,
               output_face_blendshapes=False, output_segmentation_mask=False,
               model="holistic_landmarker.task (float16/latest)")

    def __init__(self, model_path: str):
        import mediapipe as mp
        self.version = mp.__version__
        self._mp = mp
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} (download {MPT_MODEL_URL})")
        self.model_path = model_path
        with open(model_path, "rb") as f:
            import hashlib
            self.model_sha = hashlib.sha256(f.read()).hexdigest()[:12]

    def _make(self):
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
        c = {k: v for k, v in self.CFG.items() if k not in ("running_mode", "model")}
        opts = vision.HolisticLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=self.model_path),
            running_mode=vision.RunningMode.VIDEO, **c)
        return vision.HolisticLandmarker.create_from_options(opts)

    @staticmethod
    def _detect(lm, image, ts_ms):
        """detect_for_video without the mediapipe<=0.10.21 wrapper bugs: the stock
        result builder aborts the process (``Check failed: holder_ != nullptr``) when a
        hand stream is empty, and drops pose/hands whenever the face is missing.
        Same graph and packets; every stream is read only if non-empty."""
        from mediapipe.framework.formats import landmark_pb2
        from mediapipe.python import packet_creator, packet_getter
        from mediapipe.tasks.python.vision import holistic_landmarker as H
        out = lm._process_video_data({H._IMAGE_IN_STREAM_NAME: packet_creator.create_image(
            image).at(ts_ms * H._MICRO_SECONDS_PER_MILLISECOND)})

        def get(stream, world):
            p = out.get(stream)
            if p is None or p.is_empty():
                return []
            msg = landmark_pb2.LandmarkList() if world else landmark_pb2.NormalizedLandmarkList()
            msg.MergeFrom(packet_getter.get_proto(p))
            return list(msg.landmark)

        class R:
            pass
        r = R()
        r.face_landmarks = get(H._FACE_LANDMARKS_STREAM_NAME, False)
        r.pose_landmarks = get(H._POSE_LANDMARKS_STREAM_NAME, False)
        r.pose_world_landmarks = get(H._POSE_WORLD_LANDMARKS_STREAM_NAME, True)
        r.left_hand_landmarks = get(H._LEFT_HAND_LANDMARKS_STREAM_NAME, False)
        r.left_hand_world_landmarks = get(H._LEFT_HAND_WORLD_LANDMARKS_STREAM_NAME, True)
        r.right_hand_landmarks = get(H._RIGHT_HAND_LANDMARKS_STREAM_NAME, False)
        r.right_hand_world_landmarks = get(H._RIGHT_HAND_WORLD_LANDMARKS_STREAM_NAME, True)
        return r

    def run(self, path: str, fps: float, max_frames: int | None = None) -> ClipResult:
        mp = self._mp
        lm = self._make()
        kps, wos, val = [], [], []
        t0 = time.time()
        W = H = 0
        try:
            for i, img in enumerate(iter_frames(path, max_frames)):
                H, W = img.shape[:2]
                im = mp.Image(image_format=mp.ImageFormat.SRGB, data=img[:, :, ::-1].copy())
                r = self._detect(lm, im, int(round(i * 1000.0 / fps)))
                kp = np.zeros((1, MP_K, 4), np.float32)
                wo = np.zeros((1, 75, 4), np.float32)
                okp = _fill(kp, 0, 0, r.pose_landmarks, 33, with_vis=True)
                _fill(kp, 0, 33, r.face_landmarks, 478)
                _fill(kp, 0, 511, r.left_hand_landmarks, 21)
                _fill(kp, 0, 532, r.right_hand_landmarks, 21)
                _fill(wo, 0, 0, r.pose_world_landmarks, 33, with_vis=True)
                _fill(wo, 0, 33, getattr(r, "left_hand_world_landmarks", None), 21)
                _fill(wo, 0, 54, getattr(r, "right_hand_world_landmarks", None), 21)
                kps.append(kp[0]); wos.append(wo[0]); val.append(1 if okp else 0)
        finally:
            lm.close()
        el = time.time() - t0
        T = len(kps)
        return ClipResult(np.stack(kps) if T else np.zeros((0, MP_K, 4), np.float32),
                          np.stack(wos) if T else np.zeros((0, 75, 4), np.float32),
                          np.asarray(val, np.uint8),
                          dict(fps=fps, width=W, height=H, n_frames=T, extractor=self.name,
                               extractor_version=f"mediapipe=={self.version};task_sha={self.model_sha}",
                               config_hash=config_hash(self.CFG), layout=self.layout,
                               world_layout=self.world_layout, elapsed_s=el))


class RTMW3D:
    name = "RTMW3D"
    layout = "coco133"
    world_layout = "coco133_raw3d(x,y crop-input px; z m root-relative)"
    CFG = dict(pose="rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx",
               det="yolox_m_8xb8-300e_humanart-c2c7a14a", det_every=1,
               person="largest det box", backend="onnxruntime-cuda")

    def __init__(self, device: str = "cuda", det_every: int = 1):
        self.det_every = det_every
        self.CFG = dict(self.CFG, det_every=det_every)
        if det_every != 1:
            self.name = f"RTMW3D-d{det_every}"
        import onnxruntime as ort
        if device == "cuda" and hasattr(ort, "preload_dlls"):
            ort.preload_dlls()  # pip nvidia-cudnn/cublas wheels are not on LD_LIBRARY_PATH
        import rtmlib
        from rtmlib import Wholebody3d
        self.version = getattr(rtmlib, "__version__", "0.0.16")
        self.m = Wholebody3d(backend="onnxruntime", device=device)
        if device == "cuda":  # never fall back silently to CPU (shared machine)
            for sub in (self.m.det_model, self.m.pose_model):
                prov = sub.session.get_providers()
                if prov[0] != "CUDAExecutionProvider":
                    raise RuntimeError(f"CUDA provider not active: {prov}")

    def run(self, path: str, fps: float, max_frames: int | None = None) -> ClipResult:
        kps, wos, val, boxes = [], [], [], []
        t0 = time.time()
        W = H = 0
        box, found = None, False
        for t, img in enumerate(iter_frames(path, max_frames)):
            H, W = img.shape[:2]
            if t % self.det_every == 0 or not found:  # reuse the last box in between
                b = self.m.det_model(img)
                found = len(b) > 0
                if found:  # signer = largest person box
                    b = np.asarray(b, np.float32).reshape(-1, 4)
                    box = b[np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))]
                else:
                    box = np.asarray([0, 0, W, H], np.float32)
            k3, sc, _, k2 = self.m.pose_model(img, bboxes=[box])
            k3, sc, k2 = k3[0], sc[0], k2[0]
            kp = np.zeros((133, 4), np.float32)
            kp[:, 0] = k2[:, 0] / W
            kp[:, 1] = k2[:, 1] / H
            kp[:, 2] = k3[:, 2]
            kp[:, 3] = sc
            wo = np.concatenate([k3, sc[:, None]], 1).astype(np.float32)
            kps.append(kp); wos.append(wo); val.append(1 if found else 0); boxes.append(box)
        el = time.time() - t0
        T = len(kps)
        return ClipResult(np.stack(kps) if T else np.zeros((0, 133, 4), np.float32),
                          np.stack(wos) if T else np.zeros((0, 133, 4), np.float32),
                          np.asarray(val, np.uint8),
                          dict(fps=fps, width=W, height=H, n_frames=T, extractor=self.name,
                               extractor_version=f"rtmlib=={self.version}",
                               config_hash=config_hash(self.CFG), layout=self.layout,
                               world_layout=self.world_layout, elapsed_s=el),
                          extra=dict(bbox=np.stack(boxes) if T else np.zeros((0, 4), np.float32)))


def make(name: str, **kw):
    if name == "MP-L":
        return MPLegacy()
    if name.startswith("MP-L-"):  # hand-improvement variants, e.g. MP-L-V1, MP-L-V4V3
        return MPLegacy(name[len("MP-L-"):])
    if name == "MP-T":
        return MPTasks(kw.get("model_path", "holistic_landmarker.task"))
    if name == "RTMW3D":
        return RTMW3D(kw.get("device", "cuda"))
    if name.startswith("RTMW3D-d"):  # detector run every N frames, box reused between
        return RTMW3D(kw.get("device", "cuda"), det_every=int(name[len("RTMW3D-d"):]))
    raise ValueError(name)
