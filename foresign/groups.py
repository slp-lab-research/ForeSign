"""RESONA-77 → ForeSign 18 spatial groups (design §5.1).

77-joint layout (from resona/models/graph.py):
  body9    0-8   : 0 nose, 1 l_sh, 2 r_sh, 3 l_el, 4 r_el, 5 l_wr, 6 r_wr, 7 l_hip, 8 r_hip
  lhand21  9-29  : 9 wrist, 10-13 thumb, 14-17 index, 18-21 middle, 22-25 ring, 26-29 pinky
  rhand21  30-50 : 30 wrist, 31-34 thumb, 35-38 index, 39-42 middle, 43-46 ring, 47-50 pinky
  mouth8   51-58
  face18   59-76

18 groups — wrist lives in the ARM group, NOT the hand (removes the M1a wrist
leakage R1 flagged: masking a hand must not also reveal its position/movement).
  trunk(1): nose + hips
  L-arm/R-arm(2): {shoulder, elbow, body-wrist}
  lhand/rhand(12): palm(hand-wrist) + 5 finger tokens each
  face(3): mouth / eyes+brows / contour
"""
from __future__ import annotations

GROUPS: dict[str, list[int]] = {
    # trunk / head
    "trunk":   [0, 7, 8],
    # arms (shoulder, elbow, body-wrist) — body-wrist 5/6 stays visible when hand masked
    "l_arm":   [1, 3, 5],
    "r_arm":   [2, 4, 6],
    # left hand: palm(hand-wrist 9) + 5 fingers
    "l_palm":  [9],
    "l_thumb": [10, 11, 12, 13],
    "l_index": [14, 15, 16, 17],
    "l_mid":   [18, 19, 20, 21],
    "l_ring":  [22, 23, 24, 25],
    "l_pinky": [26, 27, 28, 29],
    # right hand
    "r_palm":  [30],
    "r_thumb": [31, 32, 33, 34],
    "r_index": [35, 36, 37, 38],
    "r_mid":   [39, 40, 41, 42],
    "r_ring":  [43, 44, 45, 46],
    "r_pinky": [47, 48, 49, 50],
    # face
    "mouth":   list(range(51, 59)),
    "eyes":    list(range(59, 68)),       # face18 first half (eyes/brows, approx)
    "contour": list(range(68, 77)),       # face18 second half (contour, approx)
}

GROUP_NAMES = list(GROUPS.keys())
N_GROUPS = len(GROUP_NAMES)               # 18
assert N_GROUPS == 18, N_GROUPS

# semantic supersets for part-structured masking (M1) and reporting
HAND_GROUPS = ["l_palm", "l_thumb", "l_index", "l_mid", "l_ring", "l_pinky",
               "r_palm", "r_thumb", "r_index", "r_mid", "r_ring", "r_pinky"]
LHAND_GROUPS = [g for g in HAND_GROUPS if g.startswith("l_")]
RHAND_GROUPS = [g for g in HAND_GROUPS if g.startswith("r_")]
ARM_GROUPS = ["l_arm", "r_arm"]
FACE_GROUPS = ["mouth", "eyes", "contour"]
GROUP_INDEX = {g: i for i, g in enumerate(GROUP_NAMES)}

# sanity: every joint 0..76 covered exactly once
_seen = sorted(j for js in GROUPS.values() for j in js)
assert _seen == list(range(77)), f"coverage broken: {len(_seen)} joints"


def shuffled_groups(seed: int = 0) -> list[list[int]]:
    """Group-free control: the same joints and group sizes as GROUPS, but joints
    assigned to groups by a fixed random permutation (no anatomical meaning)."""
    import random
    sizes = [len(GROUPS[g]) for g in GROUP_NAMES]
    joints = sorted(j for g in GROUP_NAMES for j in GROUPS[g])
    random.Random(seed).shuffle(joints)
    out, i = [], 0
    for s in sizes:
        out.append(sorted(joints[i:i + s]))
        i += s
    return out
