"""Shared I/O for the pose extractors: frame reading, output schema, layouts.

Output per clip (one HDF5 file, written atomically):
  kp     (T, K, 4) float16  x, y image-normalised ([0,1] inside the frame), z, score
  world  (T, Kw, 4) float16 extractor-native 3D (see layout attrs), score
  valid  (T,)      uint8    1 = frame decoded and a person was found
  attrs: fps, width, height, n_frames, extractor, extractor_version, config_hash,
         layout, world_layout, elapsed_s, clip_id

T always equals the number of decoded frames (frames are never dropped);
frames without a detection have zeros + score 0 + valid 0.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")

# ---------------------------------------------------------------- layouts
# Semantic indices used by the metrics, per keypoint layout.
LAYOUTS = {
    # MediaPipe holistic: pose33 | face478 | left hand21 | right hand21
    "mp553": dict(K=553, l_sh=11, r_sh=12, l_el=13, r_el=14, l_wr=15, r_wr=16,
                  face=(33, 511), lhand=(511, 532), rhand=(532, 553), body=(0, 33)),
    # COCO-WholeBody 133: body17 | feet6 | face68 | left hand21 | right hand21
    "coco133": dict(K=133, l_sh=5, r_sh=6, l_el=7, r_el=8, l_wr=9, r_wr=10,
                    face=(23, 91), lhand=(91, 112), rhand=(112, 133), body=(0, 17)),
}

# 21-point hand topology (identical for MediaPipe and COCO-WholeBody hands).
HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
              (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
# finger segments only (no palm wrist->MCP bones except thumb CMC->MCP)
FINGER_BONES = [(1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11),
                (11, 12), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20)]
FINGERTIPS = [4, 8, 12, 16, 20]


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


# ---------------------------------------------------------------- frames
def _is_frame_dir(path: str) -> bool:
    return os.path.isdir(path)


def _frame_files(path: str) -> list[str]:
    return sorted(os.path.join(path, f) for f in os.listdir(path)
                  if f.lower().endswith(IMG_EXT))


def iter_frames(path: str, max_frames: int | None = None) -> Iterator[np.ndarray]:
    """Yield BGR uint8 frames from a video file or a folder of images."""
    import cv2
    n = 0
    if _is_frame_dir(path):
        for f in _frame_files(path):
            if max_frames is not None and n >= max_frames:
                return
            img = cv2.imread(f, cv2.IMREAD_COLOR)
            if img is None:
                raise IOError(f"unreadable frame {f}")
            n += 1
            yield img
        return
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    try:
        while max_frames is None or n < max_frames:
            ok, img = cap.read()
            if not ok:
                break
            n += 1
            yield img
    finally:
        cap.release()


@dataclass
class VideoInfo:
    fps: float
    width: int
    height: int
    frames: int  # decoded count (authoritative)


def probe(path: str, default_fps: float = 25.0) -> VideoInfo:
    """Return fps/size and the *decoded* frame count."""
    import cv2
    if _is_frame_dir(path):
        files = _frame_files(path)
        img = cv2.imread(files[0])
        return VideoInfo(default_fps, img.shape[1], img.shape[0], len(files))
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or default_fps
    cap.release()
    n, w, h = 0, 0, 0
    for img in iter_frames(path):
        if n == 0:
            h, w = img.shape[:2]
        n += 1
    return VideoInfo(float(fps), w, h, n)


# ---------------------------------------------------------------- output
@dataclass
class ClipResult:
    kp: np.ndarray            # (T, K, 4) float32
    world: np.ndarray         # (T, Kw, 4) float32
    valid: np.ndarray         # (T,) uint8
    attrs: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)  # optional extra arrays


def write_clip(path: str, res: ClipResult) -> None:
    import h5py
    T = res.kp.shape[0]
    assert res.kp.ndim == 3 and res.kp.shape[2] == 4, res.kp.shape
    assert res.world.shape[0] == T and res.world.shape[2] == 4, res.world.shape
    assert res.valid.shape == (T,), res.valid.shape
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with h5py.File(tmp, "w") as f:
        f.create_dataset("kp", data=np.nan_to_num(res.kp).astype(np.float16), compression="gzip")
        f.create_dataset("world", data=np.nan_to_num(res.world).astype(np.float16), compression="gzip")
        f.create_dataset("valid", data=res.valid.astype(np.uint8))
        for k, v in res.extra.items():
            f.create_dataset(k, data=v)
        for k, v in res.attrs.items():
            f.attrs[k] = v
    os.replace(tmp, path)


def read_clip(path: str) -> ClipResult:
    import h5py
    with h5py.File(path, "r") as f:
        extra = {k: f[k][()] for k in f.keys() if k not in ("kp", "world", "valid")}
        return ClipResult(f["kp"][()].astype(np.float32), f["world"][()].astype(np.float32),
                          f["valid"][()], dict(f.attrs), extra)
