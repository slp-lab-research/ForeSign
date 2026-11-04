"""Freeze the production extractor configuration (MP-L-V4V0) to configs/extract/mp_l_v4v0.json.

  python -m tools.extract.freeze_config [--git-sha SHA]     # on the extraction host

Records every holistic / hand-crop parameter, the mediapipe / opencv / numpy versions,
the code git SHA, and the sha256 of the extractor source files. extract_dataset refuses
to run if the live extractor's config hash, the mediapipe version, or the source hashes
differ from this file. The extractor's ``config_hash`` (and the sha256 of this json) is
stored in every output clip.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time

CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(CODE_DIR, "configs", "extract", "mp_l_v4v0.json")
EXTRACTOR = "MP-L-V4V0"
EXPECTED_CONFIG_HASH = "30d5b11c99ac"   # docs/mp_hand_improvement.md
EXPECTED_MEDIAPIPE = "0.10.21"
SOURCE_FILES = ["tools/extract/extractors.py", "tools/extract/common.py"]


def file_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def source_hashes() -> dict:
    """sha256 of the extractor sources with line endings normalised to LF (the repo is
    checked out with CRLF on Windows and LF on the extraction hosts)."""
    out = {}
    for p in SOURCE_FILES:
        with open(os.path.join(CODE_DIR, p), "rb") as f:
            out[p] = hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()
    return out


def git_sha() -> str:
    p = os.path.join(CODE_DIR, "GIT_SHA")  # written by the deploy step (no .git on hosts)
    if os.path.exists(p):
        return open(p).read().strip()
    try:
        return subprocess.check_output(["git", "-C", CODE_DIR, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build_config(sha: str | None = None) -> dict:
    import cv2
    import numpy as np

    from tools.extract import extractors as E
    from tools.extract.common import config_hash
    ex = E.make(EXTRACTOR)
    cfg = dict(
        name=EXTRACTOR,
        extractor=ex.name,
        variant=ex.variant,
        config_hash=config_hash(ex.CFG),
        extractor_cfg=ex.CFG,
        holistic_cfg=ex.holistic_cfg,
        hand_crop=dict(enabled=ex.crop, vis=ex.fb_vis, center=E.FB_CENTER, side=E.FB_SIDE,
                       min_forearm=E.FB_MIN_FOREARM, gate=E.FB_GATE, dup=E.FB_DUP,
                       input=E.FB_INPUT,
                       hands=dict(static_image_mode=True, max_num_hands=2, model_complexity=1,
                                  min_detection_confidence=0.5)),
        upscale_min_side=ex.upscale,
        mediapipe_version=ex.version,
        opencv_version=cv2.__version__,
        numpy_version=np.__version__,
        python_version=platform.python_version(),
        code_git_sha=sha or git_sha(),
        code_files_sha256=source_hashes(),
        output=dict(
            kp="(T,553,4) float16 x,y image-normalised, z, score; pose33|face478|L21|R21",
            world="(T,33,4) float16 pose_world_landmarks x,y,z (m, hip-centred), visibility",
            flags="(T,3) uint8 [pose found, left hand src, right hand src]; "
                  "src 0 none / 1 holistic / 2 crop step",
            scores="pose = visibility; face/hands = 1 if detected else 0"),
        frozen_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    if cfg["config_hash"] != EXPECTED_CONFIG_HASH:
        raise RuntimeError(f"config hash {cfg['config_hash']} != {EXPECTED_CONFIG_HASH}")
    if cfg["mediapipe_version"] != EXPECTED_MEDIAPIPE:
        raise RuntimeError(f"mediapipe {cfg['mediapipe_version']} != {EXPECTED_MEDIAPIPE}")
    return cfg


def load_frozen(path: str = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_file_sha256"] = file_sha256(path)
    return cfg


def check_frozen(cfg: dict, ex=None) -> list[str]:
    """Differences between the frozen config and the live code/extractor (empty = OK)."""
    errs = []
    live = source_hashes()
    for p, h in cfg["code_files_sha256"].items():
        if live.get(p) != h:
            errs.append(f"source {p} changed since freeze")
    if ex is not None:
        from tools.extract.common import config_hash
        if config_hash(ex.CFG) != cfg["config_hash"]:
            errs.append(f"extractor config hash {config_hash(ex.CFG)} != {cfg['config_hash']}")
        if ex.version != cfg["mediapipe_version"]:
            errs.append(f"mediapipe {ex.version} != {cfg['mediapipe_version']}")
    return errs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--git-sha")
    ap.add_argument("--out", default=CONFIG_PATH)
    a = ap.parse_args(argv)
    cfg = build_config(a.git_sha)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"wrote {a.out} config_hash={cfg['config_hash']} sha256={file_sha256(a.out)[:12]}")


if __name__ == "__main__":
    main()
