import json
import os

import h5py
import numpy as np
import pytest
import torch


def write_canon(path, clips):
    """clips: {clip_id: (T, split)} -> canon-layout HDF5 (x (T,77,3) f16, mask (T,77) u8)."""
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.attrs["canon_version"] = "canon-test"
        for cid, (T, split, *rest) in clips.items():
            g = f.create_group(cid)
            x = rng.normal(size=(T, 77, 3)).astype(np.float16)
            m = np.ones((T, 77), np.uint8)
            m[:, 7:9] = 0
            x[:, 7:9] = 0
            g.create_dataset("x", data=x)
            g.create_dataset("mask", data=m)
            g.attrs["split"] = split
            g.attrs["fps"] = 25.0
            g.attrs["no_pose"] = bool(rest[0]) if rest else False


@pytest.fixture
def canon_root(tmp_path):
    root = tmp_path / "canon"
    root.mkdir()
    write_canon(root / "autsl.h5", {"autsl__train__a": (40, "train"),
                                    "autsl__test__b": (40, "test"),
                                    "autsl__train__short": (10, "train"),
                                    "autsl__train__nopose": (40, "train", True)})
    write_canon(root / "youtube_asl_b00001.h5", {"youtube_asl__v1": (600, "")})
    write_canon(root / "youtube_asl_b00002.h5", {"youtube_asl__v2": (300, "")})
    write_canon(root / "phoenix14t.h5", {"phoenix14t__train__p": (70, "train"),
                                         "phoenix14t__train__q": (70, "train"),
                                         "phoenix14t__dev__r": (70, "val")})
    ex = tmp_path / "ex.json"
    ex.write_text(json.dumps({"exclude": {"phoenix14t": ["phoenix14t__train__q"]}}),
                  encoding="utf-8")
    return str(root), str(ex)


def test_index_scans_files_and_batches(canon_root, tmp_path):
    from foresign.canon_data import load_or_build_canon_index
    root, _ = canon_root
    idx = load_or_build_canon_index(root, ["autsl", "youtube_asl", "phoenix14t"], min_frames=24,
                                    cache_dir=str(tmp_path / "cache"))
    assert idx["files"]["youtube_asl"] == ["youtube_asl_b00001.h5", "youtube_asl_b00002.h5"]
    ids = {ds: sorted(r[1] for r in rows) for ds, rows in idx["clips"].items()}
    assert ids["autsl"] == ["autsl__test__b", "autsl__train__a"]       # short + no_pose dropped
    assert ids["youtube_asl"] == ["youtube_asl__v1", "youtube_asl__v2"]
    yt = {r[1]: r for r in idx["clips"]["youtube_asl"]}
    assert yt["youtube_asl__v2"][0] == 1 and yt["youtube_asl__v2"][2] == 300
    # cached: a second call does not rescan unchanged files
    import foresign.canon_data as cd
    calls = []
    orig = cd.scan_file
    cd.scan_file = lambda *a, **k: calls.append(a) or orig(*a, **k)
    try:
        load_or_build_canon_index(root, ["autsl", "youtube_asl"], min_frames=24,
                                  cache_dir=str(tmp_path / "cache"))
    finally:
        cd.scan_file = orig
    assert calls == []


def test_missing_dataset_file_raises(canon_root, tmp_path):
    from foresign.canon_data import load_or_build_canon_index
    root, _ = canon_root
    with pytest.raises(FileNotFoundError):
        load_or_build_canon_index(root, ["autsl", "wlasl"], cache_dir=str(tmp_path / "c"))


def _stream(canon_root, tmp_path, **kw):
    from foresign.canon_data import CanonClips
    root, ex = canon_root
    return CanonClips(datasets=["autsl", "youtube_asl", "phoenix14t"], clip_len=32, root=root,
                      exclude_path=ex, cache_dir=str(tmp_path / "cache"), seed=0, **kw)


def test_stream_is_inductive_and_shapes(canon_root, tmp_path, monkeypatch):
    monkeypatch.delenv("FS_INDUCTIVE", raising=False)
    ds = _stream(canon_root, tmp_path)
    pool = {d: sorted(r[1] for r in rows) for d, rows in ds.index.items()}
    assert pool["autsl"] == ["autsl__train__a"]
    assert pool["phoenix14t"] == ["phoenix14t__train__p"]
    it = iter(ds)
    seen = set()
    for _ in range(60):
        pose, nv, lang, dsid = next(it)
        assert pose.shape == (32, 77 * 4) and pose.dtype == torch.float32
        v = pose.reshape(32, 77, 4)
        assert torch.all(v[:, 7:9] == 0)                      # hips absent
        assert torch.all(v[nv:] == 0)                         # padding: coords 0, mask 0
        assert torch.all(v[:nv, 0, 3] == 1)
        seen.add(ds.names[dsid])
    assert seen == {"autsl", "youtube_asl", "phoenix14t"}


def test_transductive_flag_keeps_eval_clips(canon_root, tmp_path, monkeypatch):
    monkeypatch.setenv("FS_INDUCTIVE", "0")
    ds = _stream(canon_root, tmp_path)
    assert "autsl__test__b" in {r[1] for r in ds.index["autsl"]}


def test_window_weighting(canon_root, tmp_path):
    ds = _stream(canon_root, tmp_path, sample_unit="windows", temperature=1.0)
    w = dict(zip(ds.names, ds.weights))
    # youtube: ceil(600/32)+ceil(300/32) = 19+10 = 29 windows; autsl 2; phoenix 3
    assert w["youtube_asl"] == pytest.approx(29 / 34)
    ds2 = _stream(canon_root, tmp_path, sample_unit="clips", temperature=1.0)
    w2 = dict(zip(ds2.names, ds2.weights))
    assert w2["youtube_asl"] == pytest.approx(2 / 4)


def test_in_dim_2(canon_root, tmp_path):
    ds = _stream(canon_root, tmp_path, in_dim=2)
    pose, nv, _, _ = next(iter(ds))
    assert pose.shape == (32, 77 * 3)
