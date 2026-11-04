"""build_canon: raw shards -> canon-v1 HDF5 (idempotent, resumable)."""
import csv
import json
import os

import h5py
import numpy as np

from tests.test_canon import synth
from tools.canon import build_canon as B


def make_unit(root, ds="toy", batch=None, n_shards=2, per=3):
    out = os.path.join(root, "out", ds, *( [batch] if batch else []))
    os.makedirs(out)
    os.makedirs(os.path.join(root, "manifest"), exist_ok=True)
    rows = []
    for s in range(n_shards):
        with h5py.File(os.path.join(out, f"shard_{s:05d}.h5"), "w") as f:
            for c in range(per):
                cid = f"{ds}__c{s}_{c}"
                kp, world, flags, W, H = synth(T=30 + c, seed=c)
                if c == 2:
                    flags[:, 0] = 0              # one no-pose clip per shard
                g = f.create_group(cid)
                g["kp"] = kp.astype(np.float16)
                g["world"] = world.astype(np.float16)
                g["flags"] = flags
                g.attrs.update(dict(width=W, height=H, fps=30.0, split="", source="src/" + cid,
                                    dataset=ds))
                rows.append(dict(clip_id=cid, split="train" if c else "test"))
    stem = f"{ds}_{batch}" if batch else ds
    with open(os.path.join(root, "manifest", stem + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip_id", "split"])
        w.writeheader()
        w.writerows(rows)
    return rows


def test_build_and_summary(tmp_path):
    root = str(tmp_path / "src")
    rows = make_unit(root)
    out = str(tmp_path / "canon" / "toy.h5")
    os.makedirs(os.path.dirname(out))
    summ = B.run(root, "toy", out, workers=1, log=lambda s: None)
    assert summ["clips"] == 6 and summ["no_pose_clips"] == 2
    assert summ["frames_in"] == 2 * (30 + 31 + 32)
    assert summ["frames_out"] == 2 * (25 + 26 + 27)
    with h5py.File(out, "r") as f:
        assert f.attrs["canon_version"] == "canon-v1"
        assert sorted(f.keys()) == sorted(r["clip_id"] for r in rows)
        g = f["toy__c0_1"]
        assert g["x"].shape == (26, 77, 3) and g["x"].dtype == np.float16
        assert g["mask"].shape == (26, 77)
        assert g.attrs["split"] == "train" and f["toy__c1_0"].attrs["split"] == "test"
        assert g.attrs["fps"] == 25 and g.attrs["fps_src"] == 30
        assert g.attrs["canon_version"] == "canon-v1" and g.attrs["code_git_sha"]
        assert bool(f["toy__c0_2"].attrs["no_pose"])
    assert not os.path.exists(out + ".parts")
    with open(str(tmp_path / "canon" / "toy_summary.json")) as f:
        assert json.load(f)["clips"] == 6


def test_resume_skips_done_parts(tmp_path):
    root = str(tmp_path / "src")
    make_unit(root, ds="yt", batch="b00001")
    out = str(tmp_path / "yt_b00001.h5")
    shard_dir, man = B.unit_dirs(root, "yt", "b00001")
    pdir = out + ".parts"
    os.makedirs(pdir)
    # pretend shard 0 was converted before a crash; a stale .tmp is left over
    B.convert_shard(os.path.join(shard_dir, "shard_00000.h5"),
                    os.path.join(pdir, "shard_00000.h5"), B.read_splits(man), "x")
    open(os.path.join(pdir, "shard_00001.h5.tmp"), "w").close()
    logs = []
    summ = B.run(root, "yt", out, batch="b00001", workers=2, log=logs.append)
    assert "1 to convert" in logs[0]
    assert summ["clips"] == 6
    with h5py.File(out, "r") as f:
        assert len(f.keys()) == 6


def test_summarise_rates():
    st = [dict(n_in=10, n_out=10, no_pose=False, floor=False,
               rate=dict(body=1.0, lhand=0.5, rhand=0.0, face=1.0)),
          dict(n_in=30, n_out=30, no_pose=True, floor=False,
               rate=dict(body=0.0, lhand=0.0, rhand=0.0, face=0.0))]
    s = B.summarise(st)
    assert s["no_pose_pct"] == 50.0
    assert s["mask_rate_frames"]["lhand"] == 0.5 * 10 / 40


def test_build_v2b(tmp_path):
    root = str(tmp_path / "src")
    make_unit(root)
    out = str(tmp_path / "toy_v2b.h5")
    summ = B.run(root, "toy", out, workers=1, log=lambda s: None, version="canon-v2b")
    assert summ["canon_version"] == "canon-v2b" and summ["torso_degenerate_clips"] == 0
    with h5py.File(out, "r") as f:
        assert f.attrs["canon_version"] == "canon-v2b"
        g = f["toy__c0_1"]
        assert g.attrs["canon_version"] == "canon-v2b"
        assert g["mask"][:, 7:9].sum() == 0
