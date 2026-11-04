"""Full re-extraction pipeline: manifest stability, resume logic, verification failures,
cleanup guards. Synthetic small videos / frame folders; a fake extractor stands in for
MediaPipe so the tests run anywhere."""
import csv
import os
import shutil
import zipfile

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
h5py = pytest.importorskip("h5py")

from tools.extract import build_manifest as BM
from tools.extract import cleanup as CL
from tools.extract import datasets as D
from tools.extract import extract_dataset as EX
from tools.extract import verify_dataset as V
from tools.extract.common import ClipResult, iter_frames

CFG_HASH = "fakehash0000"


def _video(path, n, w=96, h=72, fps=25.0, seed=0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert vw.isOpened()
    for i in range(n):
        img = np.full((h, w, 3), (seed * 37) % 255, np.uint8)
        cv2.circle(img, (8 + (3 * i) % (w - 16), h // 2), 6, (255, 255, 255), -1)
        vw.write(img)
    vw.release()
    return path


def _frames(d, n, seed=0):
    os.makedirs(d)
    for i in range(n):
        cv2.imwrite(os.path.join(d, f"images{i + 1:04d}.png"),
                    np.full((40, 32, 3), (seed * 10 + i) % 255, np.uint8))
    return d


class FakeEx:
    """Deterministic stand-in: pose found on every frame, wrists visible, left hand from
    holistic on even frames, right hand from the crop step on every third frame."""
    calls: list = []

    def __init__(self, fail_on=()):
        self.fail_on = set(fail_on)

    def run(self, path, fps, max_frames=None):
        FakeEx.calls.append(os.path.basename(path))
        if any(s in path for s in self.fail_on):
            raise IOError("synthetic failure")
        T = sum(1 for _ in iter_frames(path))
        kp = np.zeros((T, 553, 4), np.float32)
        kp[:, :33, :2] = 0.5
        kp[:, :33, 3] = 0.9
        kp[:, 33:511, 3] = 1.0
        src = np.zeros((T, 2), np.uint8)
        src[::2, 0] = 1
        src[::3, 1] = 2
        return ClipResult(kp, np.zeros((T, 33, 4), np.float32), np.ones(T, np.uint8),
                          dict(fps=fps, width=96, height=72, n_frames=T, extractor="FAKE",
                               config_hash=CFG_HASH, elapsed_s=0.01), dict(hand_src=src))


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """A workspace with a synthetic video dataset 'synv', a zip dataset 'synz' and a
    frame-folder dataset 'synf' (two roots = splits)."""
    monkeypatch.setenv("REEXTRACT_ROOT", str(tmp_path / "work"))
    src = tmp_path / "src"
    vids = src / "v"
    for i, n in enumerate([5, 7, 9, 6]):
        _video(str(vids / f"clip{i}.mp4"), n, seed=i)
    _video(str(vids / "sub" / "clip0.mp4"), 4, seed=9)  # same basename, other dir
    # exact duplicate of clip1
    with open(vids / "clip1.mp4", "rb") as f, open(vids / "clip1_copy.mp4", "wb") as g:
        g.write(f.read())
    zpath = src / "arch.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.write(vids / "clip2.mp4", "pre/a b/clip2.mp4")
        z.write(vids / "clip3.mp4", "pre/c/clip3.mp4")
    for s, names in (("train", ["s1", "s2"]), ("test", ["s3"])):
        for j, nm in enumerate(names):
            _frames(str(src / "f" / s / nm), 3 + j, seed=j)
    monkeypatch.setitem(D.DATASETS, "synv", dict(kind="files_rec", split=None,
                                                 roots=[(str(vids), "")]))
    monkeypatch.setitem(D.DATASETS, "synz", dict(kind="zip", split=None, strip_prefix="pre/",
                                                 roots=[(str(zpath), "")]))
    monkeypatch.setitem(D.DATASETS, "synf", dict(kind="framedirs", split="root", fps=25.0,
                                                 roots=[(str(src / "f" / "train"), "train"),
                                                        (str(src / "f" / "test"), "test")]))
    FakeEx.calls = []
    return tmp_path


def _manifest_text(ds):
    with open(D.wpath("manifest", f"{ds}.csv"), encoding="utf-8") as f:
        return f.read()


def _rows(ds):
    return BM.read_manifest(D.wpath("manifest", f"{ds}.csv"))


# ---------------------------------------------------------------- manifest
def test_manifest_stable_across_runs_and_workers(ws):
    BM.build("synv", workers=1, log=lambda m: None)
    a = _manifest_text("synv")
    BM.build("synv", workers=2, log=lambda m: None)
    assert _manifest_text("synv") == a
    rows = _rows("synv")
    ids = [r["clip_id"] for r in rows]
    assert ids == sorted(ids) and len(set(ids)) == len(ids) == 6
    assert "synv__sub__clip0" in ids and "synv__clip0" in ids
    by = {r["clip_id"]: r for r in rows}
    assert by["synv__clip2"]["frames"] == 9 and by["synv__clip2"]["sha256_mode"] == "full"
    assert by["synv__clip1_copy"]["dup_of"] == "synv__clip1" and by["synv__clip1"]["dup_of"] == ""
    assert all(r["downloaded_by_us"] == 0 for r in rows)


def test_manifest_ids_unchanged_when_sources_added(ws):
    BM.build("synv", workers=1, log=lambda m: None)
    before = {r["clip_id"]: r["sha256"] for r in _rows("synv")}
    _video(str(ws / "src" / "v" / "aaa_new.mp4"), 3, seed=5)
    BM.build("synv", workers=1, log=lambda m: None)
    after = {r["clip_id"]: r["sha256"] for r in _rows("synv")}
    assert set(after) - set(before) == {"synv__aaa_new"}
    assert all(after[k] == v for k, v in before.items())


def test_manifest_zip_unpacks_to_scratch_and_frames(ws):
    BM.build("synz", workers=1, log=lambda m: None)
    rows = _rows("synz")
    assert [r["clip_id"] for r in rows] == ["synz__a_b__clip2", "synz__c__clip3"]
    for r in rows:
        assert r["unpacked_to_scratch"] == 1 and "::" in r["source"]
        assert os.path.exists(r["path"]) and r["path"].startswith(D.wpath("scratch", "synz"))
    assert [r["frames"] for r in rows] == [9, 6]
    BM.build("synf", workers=1, log=lambda m: None)
    rows = _rows("synf")
    assert [(r["clip_id"], r["split"], r["frames"]) for r in rows] == [
        ("synf__test__s3", "test", 3), ("synf__train__s1", "train", 3),
        ("synf__train__s2", "train", 4)]
    assert all(r["sha256_mode"] == "frames_md5" and r["fps"] == 25.0 for r in rows)
    # s1 and s3 have identical frames -> later one in clip_id order is the duplicate
    by = {r["clip_id"]: r for r in rows}
    assert by["synf__train__s1"]["dup_of"] == "synf__test__s3"


def test_fast_hash_mode(tmp_path):
    p = tmp_path / "big.bin"
    p.write_bytes(os.urandom(9 << 20))
    sha, mode = BM.file_sha256(str(p))
    assert mode == "fast4mb" and len(sha) == 64
    b = bytearray(p.read_bytes())
    b[(9 << 20) // 2] ^= 1  # change in the unhashed middle is not seen (documented)
    p.write_bytes(bytes(b))
    assert BM.file_sha256(str(p))[0] == sha


# ---------------------------------------------------------------- extraction + resume
def _extract(ds, **kw):
    kw.setdefault("shard_clips", 2)
    return EX.run(ds, workers=1, extractor="FAKE", frozen=None,
                  factory=kw.pop("factory", FakeEx), log=lambda m: None, **kw)


def _verify(ds, **kw):
    return V.verify(ds, cfg_hash=kw.pop("cfg_hash", CFG_HASH), **kw)


def test_extract_and_verify_pass(ws):
    BM.build("synv", workers=1, log=lambda m: None)
    s = _extract("synv")
    assert s["failures"] == 0 and s["dups_copied"] == 1
    assert "clip1_copy.mp4" not in FakeEx.calls  # duplicate copied, not re-extracted
    rep = _verify("synv")
    assert rep["status"] == "pass", rep["checks"]
    assert rep["output_clips"] == 6 and rep["shards"] == 3
    with h5py.File(os.path.join(D.wpath("out", "synv"), "shard_00000.h5")) as f:
        g = f[sorted(f.keys())[0]]
        assert g["kp"].dtype == np.float16 and g["kp"].shape[1:] == (553, 4)
        assert g["world"].shape[1:] == (33, 4) and g["flags"].shape[1] == 3
        assert g.attrs["config_hash"] == CFG_HASH
    assert rep["rates"]["pose"] == 100.0 and 0 < rep["rates"]["crop_share"] < 100


def test_resume_skips_done_and_retries_failures(ws):
    BM.build("synv", workers=1, log=lambda m: None)
    s1 = _extract("synv", factory=lambda: FakeEx(fail_on=["clip3"]))
    assert s1["failures"] == 1
    with open(os.path.join(D.wpath("out", "synv"), "failures.csv")) as f:
        assert "synv__clip3" in f.read()
    assert _verify("synv")["checks"]["missing"] == 1
    FakeEx.calls = []
    s2 = _extract("synv")
    assert FakeEx.calls == ["clip3.mp4"] and s2["failures"] == 0
    assert _verify("synv")["status"] == "pass"
    FakeEx.calls = []
    _extract("synv")
    assert FakeEx.calls == []  # everything done


def test_resume_after_crash(ws):
    BM.build("synv", workers=1, log=lambda m: None)
    _extract("synv", shard_clips=100)
    out = D.wpath("out", "synv")
    idx = EX.shard_index(out)
    packed = sorted(idx)
    # simulate a crash: a half-written shard, a staged clip that was already packed,
    # and a staged clip not yet packed (removed from its shard copy).
    open(os.path.join(out, "shard_00007.h5.tmp"), "wb").write(b"garbage")
    with h5py.File(idx[packed[0]], "r") as f:
        g = f[packed[0]]
        EX.write_staged(os.path.join(out, "_staging", packed[0] + ".h5"), g["kp"][()],
                        g["world"][()], g["flags"][()], dict(g.attrs))
    with h5py.File(idx[packed[1]], "a") as f:
        g = f[packed[1]]
        EX.write_staged(os.path.join(out, "_staging", packed[1] + ".h5"), g["kp"][()],
                        g["world"][()], g["flags"][()], dict(g.attrs))
        del f[packed[1]]
    FakeEx.calls = []
    _extract("synv", shard_clips=100)
    assert FakeEx.calls == []
    assert not os.path.exists(os.path.join(out, "shard_00007.h5.tmp"))
    assert os.listdir(os.path.join(out, "_staging")) == []
    rep = _verify("synv")
    assert rep["status"] == "pass", rep["checks"]


# ---------------------------------------------------------------- verification failures
@pytest.fixture
def done(ws):
    BM.build("synv", workers=1, log=lambda m: None)
    _extract("synv", shard_clips=100)
    assert _verify("synv")["status"] == "pass"
    out = D.wpath("out", "synv")
    return out, os.path.join(out, "shard_00000.h5")


def _rewrite_manifest(ds, fn):
    rows = _rows(ds)
    fn(rows)
    BM.write_manifest(rows, D.wpath("manifest", f"{ds}.csv"))


def test_verify_missing_and_extra(done):
    out, shard = done
    with h5py.File(shard, "a") as f:
        del f["synv__clip0"]
    rep = _verify("synv")
    assert rep["status"] == "fail" and rep["missing"] == ["synv__clip0"]
    _rewrite_manifest("synv", lambda rows: rows.pop())
    rep = _verify("synv")
    assert rep["checks"]["extra"] == 1


def test_verify_frame_mismatch(done):
    def bump(rows):
        rows[0]["frames"] += 1
    _rewrite_manifest("synv", bump)
    rep = _verify("synv")
    assert rep["status"] == "fail" and rep["checks"]["frame_mismatch"] == 1


def test_verify_nan(done):
    out, shard = done
    with h5py.File(shard, "a") as f:
        kp = f["synv__clip2/kp"][()]
        kp[1, 5, 0] = np.nan
        f["synv__clip2/kp"][...] = kp
    rep = _verify("synv")
    assert rep["status"] == "fail" and rep["bad"][0][0] == "synv__clip2"
    assert "nan/inf" in rep["bad"][0][1]


def test_verify_config_hash_and_staging(done):
    out, shard = done
    assert _verify("synv", cfg_hash="otherhash000")["checks"]["bad"] == 6
    open(os.path.join(out, "_staging", "synv__zzz.h5"), "wb").close()
    assert _verify("synv")["checks"]["staging_leftover"] == 1


def test_verify_duplicate_group(done):
    out, shard = done
    with h5py.File(shard, "r") as fi, h5py.File(os.path.join(out, "shard_00001.h5"), "w") as fo:
        fi.copy(fi["synv__clip0"], fo, name="synv__clip0")
    assert _verify("synv")["checks"]["duplicated"] == 1


def test_verify_detection_rate_out_of_range(done):
    ref = {"synv": dict(pose=100.0, face=100.0, hand_given_wrist=95.0)}
    D.BAKEOFF_REF["synv"] = "synv"
    try:
        rep = _verify("synv", ref=ref)  # fake hand_given_wrist is ~42 %
        assert rep["status"] == "fail" and rep["checks"]["rate_out_of_range"] == ["hand_given_wrist"]
        ref["synv"]["hand_given_wrist"] = rep["rates"]["hand_given_wrist"] + 8
        rep = _verify("synv", ref=ref)
        assert rep["status"] == "pass" and rep["warnings"]
    finally:
        del D.BAKEOFF_REF["synv"]


# ---------------------------------------------------------------- cleanup guards
def test_cleanup_requires_verification_and_spares_originals(ws):
    BM.build("synz", workers=1, log=lambda m: None)
    with pytest.raises(RuntimeError):
        CL.clean_scratch("synz", log=lambda m: None)
    _extract("synz")
    assert _verify("synz")["status"] == "pass"
    assert CL.clean_scratch("synz", log=lambda m: None) == 2
    assert not os.path.exists(D.wpath("scratch", "synz"))
    assert os.path.exists(ws / "src" / "arch.zip")


def test_delete_videos_refuses_non_downloaded(ws):
    BM.build("synv", workers=1, log=lambda m: None)
    _extract("synv")
    assert _verify("synv")["status"] == "pass"
    assert CL.delete_videos("synv", log=lambda m: None) == 0
    assert all(os.path.exists(r["source"]) for r in _rows("synv"))
    # a row claiming downloaded_by_us but outside the download staging dir is refused too
    _rewrite_manifest("synv", lambda rows: rows[0].update(downloaded_by_us=1))
    assert CL.delete_videos("synv", log=lambda m: None) == 0
    assert all(os.path.exists(r["source"]) for r in _rows("synv"))


def test_files_exclude_and_id_sub(tmp_path, monkeypatch):
    """OpenASL-style source dir: audio-only streams excluded, format code dropped."""
    root = tmp_path / "oasl"
    root.mkdir()
    for n in ("abcdefghijk.f605.mp4", "abcdefghijk.f251.webm", "zyxwvutsrqp.f243.webm",
              "zyxwvutsrqp.f140.m4a", "_download.log"):
        (root / n).write_bytes(b"x")
    (root / "_av1_orig").mkdir()
    monkeypatch.setitem(D.DATASETS, "syno", dict(kind="files", split=None,
                                                 id_sub=(r"\.f\d+$", ""),
                                                 exclude=r"\.f251\.webm$",
                                                 roots=[(str(root), "")]))
    srcs = D.enumerate_sources("syno")
    assert [s.rel for s in srcs] == ["abcdefghijk.f605.mp4", "zyxwvutsrqp.f243.webm"]
    assert [D.make_clip_id("syno", s, False) for s in srcs] == ["syno__abcdefghijk",
                                                                "syno__zyxwvutsrqp"]


def test_queue_dataset_downloaded_by_us_and_delete(ws, monkeypatch):
    """YouTube-ASL-style: only finished queue entries are sources, rows are marked
    downloaded_by_us, and verified videos under download_root may be deleted."""
    vids = ws / "dl" / "videos"
    a = _video(str(vids / "-abc_DEF123.mp4"), 5, seed=1)
    b = _video(str(vids / "zzzzzzzzzzz.mp4"), 6, seed=2)
    _video(str(vids / "partial0000.mp4"), 3, seed=3)        # not in the queue yet
    (vids / "growing0000.mp4.part").write_bytes(b"x")
    q = ws / "dl" / "completed.tsv"
    q.write_text(f"-abc_DEF123\t-abc_DEF123.mp4\t{os.path.getsize(a)}\t2026-10-25T00:00:00Z\n"
                 f"zzzzzzzzzzz\tzzzzzzzzzzz.mp4\t{os.path.getsize(b)}\t2026-10-25T00:01:00Z\n"
                 f"gone0000000\tgone0000000.mp4\t123\t2026-10-25T00:02:00Z\n")
    monkeypatch.setitem(D.DATASETS, "synq", dict(kind="queue", split=None, downloaded_by_us=1,
                                                 queue=str(q), download_root=str(vids),
                                                 roots=[(str(vids), "")]))
    BM.build("synq", workers=1, log=lambda m: None)
    rows = _rows("synq")
    assert [r["clip_id"] for r in rows] == ["synq__-abc_DEF123", "synq__zzzzzzzzzzz"]
    assert all(r["downloaded_by_us"] == 1 for r in rows)
    skip = ws / "skip.txt"
    skip.write_text("synq__-abc_DEF123\n")
    monkeypatch.setenv("REEXTRACT_QUEUE_SKIP", str(skip))
    assert [s.rel for s in D.enumerate_sources("synq")] == ["zzzzzzzzzzz.mp4"]
    monkeypatch.delenv("REEXTRACT_QUEUE_SKIP")
    _extract("synq")
    assert _verify("synq")["status"] == "pass"
    assert CL.delete_videos("synq", log=lambda m: None) == 2
    assert not os.path.exists(a) and not os.path.exists(b)
    assert os.path.exists(vids / "partial0000.mp4")


# ---------------------------------------------------------------- tier 2 / batches
def test_exclude_csv_skips_extraction_and_verify_passes(ws):
    """A source that cannot be decoded is listed in manifest/<ds>_exclude.csv: extract
    skips it, verify passes and lists it, cleanup never deletes it."""
    BM.build("synv", workers=1, log=lambda m: None)
    s1 = _extract("synv", factory=lambda: FakeEx(fail_on=["clip3"]))
    assert s1["failures"] == 1 and _verify("synv")["status"] == "fail"
    with open(D.wpath("manifest", "synv_exclude.csv"), "w", newline="") as f:
        csv.writer(f).writerows([("clip_id", "reason"), ("synv__clip3", "corrupt source")])
    FakeEx.calls = []
    s2 = _extract("synv", factory=lambda: FakeEx(fail_on=["clip3"]))
    assert FakeEx.calls == [] and s2["failures"] == 0 and s2["excluded"] == 1
    rep = _verify("synv")
    assert rep["status"] == "pass", rep["checks"]
    assert rep["excluded_clips"] == 1 and rep["excluded"][0]["clip_id"] == "synv__clip3"
    assert rep["excluded"][0]["reason"] == "corrupt source"
    assert rep["manifest_rows"] == rep["manifest_clips"] + 1
    _rewrite_manifest("synv", lambda rows: [r.update(downloaded_by_us=1) for r in rows])
    CL.delete_videos("synv", download_root=str(ws / "src"), log=lambda m: None)
    assert os.path.exists(ws / "src" / "v" / "clip3.mp4")
    assert not os.path.exists(ws / "src" / "v" / "clip0.mp4")


def test_time_slice_resumes(ws):
    BM.build("synv", workers=1, log=lambda m: None)
    s1 = _extract("synv", max_wall_s=1e-9)  # slice used before the first clip
    assert s1["sliced"] and s1["clips"] == 0
    s2 = _extract("synv")
    assert not s2["sliced"] and s2["clips"] == 6
    assert _verify("synv")["status"] == "pass"


def test_guarded_map_weight_cap_and_stop():
    from multiprocessing.pool import ThreadPool
    import threading
    import time as _t
    lock, live, peak = threading.Lock(), [0], [0]

    def fn(x):
        with lock:
            live[0] += x
            peak[0] = max(peak[0], live[0])
        _t.sleep(0.02)
        with lock:
            live[0] -= x
        return x
    g = EX.MemGuard(None, rss_fn=lambda: 0, log=lambda m: None)
    out = []
    EX.guarded_map(lambda: ThreadPool(8), fn, [5, 5, 5, 20, 1, 1], 8, g, out.append,
                   poll=0.01, weight=lambda j: j, max_weight=10)
    assert sorted(out) == [1, 1, 5, 5, 5, 20] and peak[0] <= 20  # a heavy job runs alone
    out = []
    EX.guarded_map(lambda: ThreadPool(2), fn, range(1, 50), 2, g, out.append,
                   poll=0.01, stop=lambda: len(out) >= 3)
    assert 3 <= len(out) < 49


def test_batch_paths_and_queue_batches(ws, monkeypatch):
    vids = ws / "dl" / "videos"
    q = ws / "dl" / "completed.tsv"
    lines = []
    for i in range(3):
        p = _video(str(vids / f"vid{i:08d}.mp4"), 4 + i, seed=i)
        lines.append(f"vid{i:08d}\tvid{i:08d}.mp4\t{os.path.getsize(p)}\tt\n")
    q.write_text("".join(lines))
    monkeypatch.setitem(D.DATASETS, "synq", dict(kind="queue", split=None, downloaded_by_us=1,
                                                 queue=str(q), download_root=str(vids),
                                                 roots=[(str(vids), "")]))
    up = D.unit_paths("synq", "b00001")
    assert up["manifest"].endswith(os.path.join("manifest", "synq_b00001.csv"))
    assert up["out_dir"].endswith(os.path.join("out", "synq", "b00001"))
    BM.build("synq", workers=1, limit=2, stem="synq_b00001", log=lambda m: None)
    rows = BM.read_manifest(up["manifest"])
    assert [r["clip_id"] for r in rows] == ["synq__vid00000000", "synq__vid00000001"]
    EX.run("synq", workers=1, extractor="FAKE", frozen=None, factory=FakeEx,
           manifest=up["manifest"], out_dir=up["out_dir"], log=lambda m: None)
    rep = V.verify("synq", manifest=up["manifest"], out_dir=up["out_dir"],
                   report=up["report"], cfg_hash=CFG_HASH)
    assert rep["status"] == "pass" and os.path.exists(up["report"])
    assert CL.delete_videos("synq", manifest=up["manifest"], report=up["report"],
                            log=lambda m: None) == 2
    # next batch: deleted videos and already-batched ids are not sources again
    skip = ws / "skip.txt"
    skip.write_text("\n".join(r["clip_id"] for r in rows) + "\n")
    monkeypatch.setenv("REEXTRACT_QUEUE_SKIP", str(skip))
    assert [s.rel for s in D.enumerate_sources("synq")] == ["vid00000002.mp4"]


def test_tar_frames_unpack_split_and_scratch_cleanup(ws, monkeypatch):
    import tarfile
    src = ws / "src" / "fsw"
    for d, n in (("Plus/BatchA/seq_0", 3), ("Plus/BatchA/seq_1", 2)):
        _frames(str(src / d), n)
    tgz = ws / "src" / "frames.tgz"
    with tarfile.open(tgz, "w:gz") as t:
        t.add(src / "Plus", arcname="Plus")
    monkeypatch.setitem(D.DATASETS, "synt", dict(kind="tar_frames", split=None, fps=30.0,
                                                 strip_prefix="Plus/",
                                                 roots=[(str(tgz), "plus")]))
    BM.build("synt", workers=1, log=lambda m: None)
    rows = _rows("synt")
    assert [r["clip_id"] for r in rows] == ["synt__BatchA__seq_0", "synt__BatchA__seq_1"]
    assert [r["frames"] for r in rows] == [3, 2]
    assert all(r["unpacked_to_scratch"] == 1 and r["kind"] == "frames" for r in rows)
    assert all(os.path.isdir(r["path"]) for r in rows) and rows[0]["fps"] == 30.0
    # cached: a second enumeration does not need to re-read the archive
    assert len(D.enumerate_sources("synt")) == 2
    _extract("synt")
    assert _verify("synt")["status"] == "pass"
    assert CL.clean_scratch("synt", log=lambda m: None) == 2
    assert not os.path.exists(D.wpath("scratch", "synt"))
    assert os.path.exists(tgz)


def test_tier2_registry_and_splits():
    for ds in D.TIER2_ORDER:
        assert ds in D.DATASETS
    src = D.Source("train/0001-0070/0001/03_01_0001_(x)_c.mp4", "", "01", "video")
    assert D.split_of("karsl", src) == "train"
    assert D.split_of("slovo", D.Source("a.mp4", "", "test", "video")) == "test"
    assert D.DATASETS["lsa_t"].get("origin_archive")
    assert D.DATASETS["chicagofswild"]["kind"] == "tar_frames"


# ---------------------------------------------------------------- unreadable sources / NFS
def _add_unreadable(ws):
    v = ws / "src" / "v"
    open(v / "zz_empty.mp4", "wb").close()                       # zero-byte
    with open(v / "zz_garbage.mp4", "wb") as f:                  # not a video
        f.write(b"\x00garbage" * 64)


def test_unreadable_sources_excluded_and_manifest_written(ws, monkeypatch):
    """Unreadable sources get a manifest row (sha256_mode=unreadable) and an entry in
    manifest/<ds>_exclude.csv with the reason; hand-written exclusions are kept; extract
    skips them and verify passes. A rebuild after the source is fixed drops the entry."""
    monkeypatch.setattr(BM, "MAX_UNREADABLE_FRAC", 0.5)
    _add_unreadable(ws)
    excl = D.wpath("manifest", "synv_exclude.csv")
    os.makedirs(os.path.dirname(excl), exist_ok=True)
    with open(excl, "w", newline="") as f:
        csv.writer(f).writerows([("clip_id", "reason", "date"),
                                 ("synv__clip3", "corrupt PNG (hand-written)", "2026-10-26")])
    s = BM.build("synv", workers=2, log=lambda m: None)
    assert s["status"] == "ok" and s["errors"] == 2 and s["excluded_unreadable"] == 2
    by = {r["clip_id"]: r for r in _rows("synv")}
    assert len(by) == 8
    for cid in ("synv__zz_empty", "synv__zz_garbage"):
        assert by[cid]["sha256_mode"] == "unreadable" and by[cid]["frames"] == 0
        assert by[cid]["dup_of"] == ""
    ex = D.load_excludes(D.wpath("manifest", "synv.csv"))
    assert ex["synv__clip3"] == "corrupt PNG (hand-written)"
    assert ex["synv__zz_empty"].startswith(BM.AUTO_REASON)
    assert ex["synv__zz_garbage"].startswith(BM.AUTO_REASON) and len(ex) == 3
    assert os.path.exists(D.wpath("manifest", "synv_manifest_errors.csv"))
    st = _extract("synv")
    assert st["failures"] == 0 and st["excluded"] == 3
    assert not any(c.startswith("zz_") for c in FakeEx.calls)
    rep = _verify("synv")
    assert rep["status"] == "pass", rep["checks"]
    assert rep["excluded_clips"] == 3
    # the sources are fixed / removed: a rebuild drops the automatic entries only
    os.remove(ws / "src" / "v" / "zz_empty.mp4")
    os.remove(ws / "src" / "v" / "zz_garbage.mp4")
    s = BM.build("synv", workers=1, log=lambda m: None)
    assert s["errors"] == 0
    assert D.load_excludes(D.wpath("manifest", "synv.csv")) == {
        "synv__clip3": "corrupt PNG (hand-written)"}
    assert not os.path.exists(D.wpath("manifest", "synv_manifest_errors.csv"))


def test_too_many_unreadable_sources_is_hard_failure(ws):
    """More than MAX_UNREADABLE_FRAC (2%) unreadable: no manifest, exit code 1."""
    _add_unreadable(ws)   # 2 of 8 sources
    with pytest.raises(SystemExit) as e:
        BM.main(["synv", "--workers", "1"])
    assert e.value.code == 1
    assert not os.path.exists(D.wpath("manifest", "synv.csv"))
    assert not os.path.exists(D.wpath("manifest", "synv_exclude.csv"))
    with open(D.wpath("manifest", "synv_manifest_errors.csv")) as f:
        assert len(f.read().strip().splitlines()) == 3
    # within the limit the CLI succeeds
    os.remove(ws / "src" / "v" / "zz_garbage.mp4")
    for i in range(50):   # 1 unreadable of 56 sources < 2%
        _video(str(ws / "src" / "v" / "pad" / f"p{i}.mp4"), 2, seed=i)
    with pytest.raises(SystemExit) as e:
        BM.main(["synv", "--workers", "1"])
    assert e.value.code == 0
    assert list(D.load_excludes(D.wpath("manifest", "synv.csv"))) == ["synv__zz_empty"]


def test_repeated_directory_entries_are_deduplicated(ws, monkeypatch):
    """Some NFS mounts repeat readdir entries of large directories; enumeration
    and the manifest must not see a file twice (was: 'clip_id collision')."""
    real_listdir, real_scandir = os.listdir, os.scandir

    class _Dup:
        def __init__(self, p):
            self.it = real_scandir(p)

        def __enter__(self):
            ents = list(self.it)
            return iter(ents + ents + ents)

        def __exit__(self, *a):
            self.it.close()
    monkeypatch.setattr(os, "listdir", lambda p=".": real_listdir(p) * 3)
    monkeypatch.setattr(os, "scandir", _Dup)
    srcs = D.enumerate_sources("synv")
    assert len(srcs) == len({s.source for s in srcs}) == 6
    assert len(D.enumerate_sources("synf")) == 3
    assert BM.frame_files(str(ws / "src" / "f" / "train" / "s2")) == [
        "images0001.png", "images0002.png", "images0003.png", "images0004.png"]
    monkeypatch.setattr(os, "listdir", real_listdir)
    monkeypatch.setattr(os, "scandir", real_scandir)
    # repeated Source records (same file) are dropped by build; a real collision still fails
    monkeypatch.setattr(BM, "enumerate_sources", lambda ds, roots: srcs + srcs[:2])
    s = BM.build("synv", workers=1, log=lambda m: None)
    assert s["clips"] == 6 and s["repeated_listings"] == 2
    other = D.Source(srcs[0].rel, srcs[0].source + ".other", "", "video")
    monkeypatch.setattr(BM, "enumerate_sources", lambda ds, roots: srcs + [other])
    with pytest.raises(RuntimeError, match="clip_id collision"):
        BM.build("synv", workers=1, log=lambda m: None)


def test_enumeration_from_upload_file_list(ws, monkeypatch):
    """With REEXTRACT_FILE_LIST (the uploader's verified "<size>\t<relpath>" list) the
    sources come from the list, not from readdir; ids are the same as from a walk."""
    walked = {D.make_clip_id("synv", s, False) for s in D.enumerate_sources("synv")}
    src = ws / "src"
    rels = sorted(os.path.relpath(os.path.join(dp, f), src).replace(os.sep, "/")
                  for dp, _, fs in os.walk(src) for f in fs)
    lst = ws / "files.tsv"
    lst.write_text("".join(f"1\t{r}\n" for r in rels + ["v/zz_missing.mp4", "other/x.mp4"]),
                   encoding="utf-8")
    monkeypatch.setattr(D, "LAP", str(src))
    monkeypatch.setenv("REEXTRACT_FILE_LIST", str(lst))
    D._load_file_list.cache_clear()

    def no_readdir(*a, **k):
        raise AssertionError("readdir used")
    monkeypatch.setattr(os, "scandir", no_readdir)
    srcs = D.enumerate_sources("synv")
    ids = {D.make_clip_id("synv", s, False) for s in srcs}
    assert ids == walked | {"synv__zz_missing"}
    assert all(os.path.normpath(s.source).startswith(os.path.normpath(str(src / "v")))
               for s in srcs)
    monkeypatch.undo()
    D._load_file_list.cache_clear()


def test_defer_list_skips_clips_until_their_staged_files_arrive(ws, monkeypatch, tmp_path):
    BM.build("synv", workers=1, log=lambda m: None)
    lst = tmp_path / "defer.txt"
    lst.write_text("synv__clip3" + chr(10))
    monkeypatch.setenv("REEXTRACT_DEFER_LIST", str(lst))
    FakeEx.calls = []
    _extract("synv")
    assert "clip3.mp4" not in FakeEx.calls
    assert _verify("synv")["checks"]["missing"] == 1
    # the other host extracts clip3 into its own out dir; its staged file is copied over
    other = tmp_path / "other"
    monkeypatch.delenv("REEXTRACT_DEFER_LIST")
    row = [r for r in _rows("synv") if r["clip_id"] == "synv__clip3"][0]
    EX._EX, EX._CTX["frozen"] = FakeEx(), None
    staged = str(other / "synv__clip3.h5")
    os.makedirs(other, exist_ok=True)
    EX.extract_one(EX._EX, row, staged, None)
    shutil.copy(staged, os.path.join(D.wpath("out", "synv"), "_staging", "synv__clip3.h5"))
    FakeEx.calls = []
    _extract("synv")
    assert FakeEx.calls == []  # packed, not re-extracted
    assert _verify("synv")["status"] == "pass"


def test_pack_via_local_tmp_gives_same_shards(ws, monkeypatch, tmp_path):
    BM.build("synv", workers=1, log=lambda m: None)
    loc = tmp_path / "packtmp"
    monkeypatch.setenv("REEXTRACT_PACK_TMP", str(loc))
    _extract("synv")
    assert _verify("synv")["status"] == "pass"
    assert os.listdir(loc) == []  # build dir removed
