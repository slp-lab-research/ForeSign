"""Pretraining-only 3D view augmentation for C77-3D windows (plan addendum 2026-11-02).

One rotation per sampled window about the canonical origin (shoulder midpoint):
R = R_x(pitch) @ R_y(yaw), yaw ~ U(-yaw_deg, yaw_deg) about the vertical y axis,
pitch ~ U(-pitch_deg, pitch_deg) about x. Absent joints are 0 and stay 0. The rotated window is the
single model input, so context and EMA target encoders see the same view.
"""
from __future__ import annotations

import math

import numpy as np


def random_rotation(rng, yaw_deg: float, pitch_deg: float):
    """-> (R (3,3) float32 or None when off, yaw, pitch in degrees). Draws nothing when off,
    so a disabled augmentation leaves the sampling stream unchanged."""
    if not yaw_deg and not pitch_deg:
        return None, 0.0, 0.0
    yaw = rng.uniform(-yaw_deg, yaw_deg) if yaw_deg else 0.0
    pitch = rng.uniform(-pitch_deg, pitch_deg) if pitch_deg else 0.0
    a, b = math.radians(yaw), math.radians(pitch)
    ry = np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0],
                   [-math.sin(a), 0.0, math.cos(a)]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(b), -math.sin(b)],
                   [0.0, math.sin(b), math.cos(b)]])
    return (rx @ ry).astype(np.float32), yaw, pitch


def rotate(x: np.ndarray, R: np.ndarray) -> np.ndarray:
    """x (T, J, 3) -> x @ R^T as float32 (zeros stay zeros)."""
    return x.astype(np.float32) @ R.T


# ---- fine-tuning augmentation for the islr_std recipe (plan addendum 2026-11-03) ------------

def _mirror_perm() -> np.ndarray:
    """Joint permutation for a left-right mirror of the C77 layout (canon order: body9, lhand21,
    rhand21, mouth8 = face mesh 61,40,0,270,291,321,17,91, face18 = 70,105,107,336,334,300,33,
    133,362,263,1,168,234,454,172,397,152,10)."""
    p = np.arange(77)
    pairs = [(1, 2), (3, 4), (5, 6), (7, 8)]                       # shoulders, elbows, wrists, hips
    pairs += [(9 + i, 30 + i) for i in range(21)]                  # hands
    pairs += [(51, 55), (52, 54), (56, 58)]                        # mouth: 61-291, 40-270, 321-91
    pairs += [(59, 64), (60, 63), (61, 62), (65, 68), (66, 67),    # brows, eyes
              (71, 72), (73, 74)]                                  # contour 234-454, 172-397
    for a, b in pairs:
        p[a], p[b] = b, a
    return p


MIRROR_PERM = _mirror_perm()


def mirror(x: np.ndarray, mask: np.ndarray):
    """Left-right mirror: swap left/right joints and negate x. x (T,77,3), mask (T,77)."""
    y = x[:, MIRROR_PERM].astype(np.float32).copy()
    y[..., 0] *= -1.0
    return y, mask[:, MIRROR_PERM]


def islr_augment(x: np.ndarray, mask: np.ndarray, rng, p_mirror=0.5, rot_deg=15.0,
                 scale=(0.8, 1.2), shift=0.1):
    """One draw per clip: mirror (p), in-plane rotation about z, isotropic scale, x/y shift.
    rng: numpy Generator. Absent joints are zeroed later by pack_c77 (mask unchanged except for
    the mirror's left/right swap)."""
    x = x.astype(np.float32)
    if rng.random() < p_mirror:
        x, mask = mirror(x, mask)
    a = math.radians(rng.uniform(-rot_deg, rot_deg))
    s = rng.uniform(*scale)
    t = rng.uniform(-shift, shift, size=2)
    rz = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0],
                   [0.0, 0.0, 1.0]], np.float32)
    y = (x @ rz.T) * s
    y[..., 0] += t[0]
    y[..., 1] += t[1]
    return y, mask
