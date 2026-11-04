"""I/O format and frame-count invariants for tools.extract (synthetic videos via cv2)."""
import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("h5py")

from tools.extract.common import (LAYOUTS, ClipResult, iter_frames, probe, read_clip,
                                  write_clip)


def _video(path, n=23, w=160, h=120, fps=25.0):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert vw.isOpened()
    for i in range(n):
        img = np.full((h, w, 3), 40, np.uint8)
        cv2.circle(img, (10 + 5 * i % (w - 20), h // 2), 8, (255, 255, 255), -1)
        vw.write(img)
    vw.release()
    return path


def _frames(d, n=7, w=52, h=64):
    os.makedirs(d)
    for i in range(n):
        cv2.imwrite(os.path.join(d, f"images{i + 1:04d}.png"), np.full((h, w, 3), i, np.uint8))
    return d


def test_probe_counts_decoded_frames(tmp_path):
    v = _video(str(tmp_path / "a.mp4"), n=23)
    info = probe(v)
    assert info.frames == 23 == sum(1 for _ in iter_frames(v))
    assert (info.width, info.height) == (160, 120)
    assert abs(info.fps - 25.0) < 0.01
    assert sum(1 for _ in iter_frames(v, max_frames=5)) == 5


def test_frame_folder(tmp_path):
    d = _frames(str(tmp_path / "seq"), n=7)
    info = probe(d, default_fps=25.0)
    assert (info.frames, info.width, info.height, info.fps) == (7, 52, 64, 25.0)
    fr = list(iter_frames(d))
    assert [int(f[0, 0, 0]) for f in fr] == list(range(7))  # sorted order


@pytest.mark.parametrize("layout,Kw", [("mp553", 33), ("coco133", 133)])
def test_write_read_roundtrip(tmp_path, layout, Kw):
    T, K = 11, LAYOUTS[layout]["K"]
    kp = np.random.default_rng(0).random((T, K, 4)).astype(np.float32)
    kp[3] = np.nan  # NaNs must never reach disk
    res = ClipResult(kp, np.zeros((T, Kw, 4), np.float32), np.ones(T, np.uint8),
                     dict(fps=25.0, layout=layout, n_frames=T))
    p = str(tmp_path / "c.h5")
    write_clip(p, res)
    import h5py
    with h5py.File(p) as f:
        assert f["kp"].dtype == np.float16 and f["kp"].shape == (T, K, 4)
        assert f["world"].dtype == np.float16 and f["valid"].shape == (T,)
    back = read_clip(p)
    assert not np.isnan(back.kp).any()
    assert back.attrs["layout"] == layout and back.attrs["n_frames"] == T
    np.testing.assert_allclose(back.kp[0], kp[0], atol=1e-3)


def test_write_rejects_inconsistent_T(tmp_path):
    res = ClipResult(np.zeros((5, 553, 4)), np.zeros((4, 33, 4)), np.ones(5, np.uint8))
    with pytest.raises(AssertionError):
        write_clip(str(tmp_path / "x.h5"), res)


@pytest.mark.parametrize("name", ["MP-L", "MP-T"])
def test_mediapipe_frame_count_invariant(tmp_path, name):
    mpipe = pytest.importorskip("mediapipe")
    if not hasattr(mpipe, "solutions"):
        pytest.skip("mediapipe build without legacy solutions (pin mediapipe==0.10.21)")
    from tools.extract.extractors import make
    kw = {}
    if name == "MP-T":
        mp_task = os.environ.get("MPT_MODEL", "holistic_landmarker.task")
        if not os.path.exists(mp_task):
            pytest.skip("holistic_landmarker.task not available")
        kw["model_path"] = mp_task
    v = _video(str(tmp_path / "s.mp4"), n=9)
    res = make(name, **kw).run(v, 25.0)
    assert res.kp.shape == (9, 553, 4)
    assert res.world.shape[0] == 9 and res.valid.shape == (9,)
    assert res.attrs["n_frames"] == 9 and res.attrs["layout"] == "mp553"
    # no person in a synthetic blob video: all invalid, all-zero scores
    assert res.valid.sum() == 0 and float(np.abs(res.kp[..., 3]).sum()) == 0.0
    d = _frames(str(tmp_path / "seq"), n=4, w=96, h=96)
    assert make(name, **kw).run(d, 25.0).kp.shape[0] == 4
