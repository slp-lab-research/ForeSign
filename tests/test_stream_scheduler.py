"""Streaming extraction driver (tools.extract.stream_extract): the clip scheduler on mock
clips (longest first, overlap of units, class sharing, frames-in-flight cap), a
discrete-event comparison with the old one-pool-per-unit scheme, and the driver loop end
to end on a thread pool with the fake extractor."""
import heapq
import os
import random
import threading
from multiprocessing.pool import ThreadPool

import pytest

from tools.extract import stream_extract as S


def clips(name, frames):
    return [dict(clip_id=f"{name}__{i:03d}", frames=f) for i, f in enumerate(frames)]


# ---------------------------------------------------------------- scheduler
def test_longest_first_within_a_unit():
    s = S.Scheduler(workers=8)
    s.add("b1", "yt", clips("b1", [5, 50, 20, 50, 1]))
    got = []
    while (p := s.dispatch()) is not None:
        got.append(p[1]["frames"])
    assert got == [50, 50, 20, 5, 1]
    assert s.inflight == 5 and s.queued() == 0


def test_fifo_between_units_of_a_class_and_drained():
    s = S.Scheduler(workers=2)
    s.add("b1", "yt", clips("b1", [3, 9]))
    s.add("b2", "yt", clips("b2", [100]))
    u, r = s.dispatch()
    assert (u.name, r["frames"]) == ("b1", 9)  # b1 first although b2 has a longer clip
    u2, r2 = s.dispatch()
    assert (u2.name, r2["frames"]) == ("b1", 3)
    assert s.dispatch() is None  # all workers busy
    assert s.done("b1", r) is False  # one b1 clip still in flight
    assert s.dispatch()[0].name == "b2"
    assert s.done("b1", r2) is True  # b1 drained while b2 runs
    s.remove("b1")
    with pytest.raises(AssertionError):
        s.remove("b2")  # still in flight


def test_wants_next_unit_when_fewer_queued_than_workers():
    s = S.Scheduler(workers=4)
    assert s.wants("yt") and s.wants("t2")
    s.add("b1", "yt", clips("b1", [1] * 6))
    assert not s.wants("yt") and s.wants("t2")
    s.dispatch(), s.dispatch()
    assert s.queued("yt") == 4 and not s.wants("yt")
    s.dispatch()
    assert s.wants("yt")  # 3 queued < 4 workers: open the next batch now


def test_redo_rows_are_merged_longest_first():
    s = S.Scheduler(workers=1)
    s.add("d", "t2", clips("d", [4, 2]))
    s.add("d", "t2", [dict(clip_id="d__x", frames=3)])
    got = []
    while (p := s.dispatch()) is not None:
        got.append(p[1]["frames"])
        s.done("d", p[1])
    assert got == [4, 3, 2]


def test_classes_share_dispatched_frames():
    s = S.Scheduler(workers=1000)
    s.add("b1", "yt", clips("b1", [10] * 50))
    s.add("t", "t2", clips("t", [5] * 100))
    for _ in range(60):
        s.dispatch()
    # equal frames: 30 yt clips = 300 frames vs 60 t2 clips... interleaved by frames served
    assert abs(s.served["yt"] - s.served["t2"]) <= 10


def test_yt_first_when_staging_is_high():
    s = S.Scheduler(workers=1000)
    s.add("t", "t2", clips("t", [5] * 10))
    s.add("b1", "yt", clips("b1", [10] * 10))
    got = [s.dispatch(yt_first=True)[0].cls for _ in range(12)]
    assert got == ["yt"] * 10 + ["t2"] * 2


def test_idle_class_is_caught_up_and_cannot_claim_a_backlog():
    s = S.Scheduler(workers=1000)
    s.add("t", "t2", clips("t", [10] * 100))
    for _ in range(50):
        s.dispatch()  # only t2 has work: 500 frames served
    s.add("b1", "yt", clips("b1", [10] * 100))
    got = [s.dispatch()[0].cls for _ in range(20)]
    assert got.count("yt") in (10, 11)  # alternates instead of 50 yt clips in a row


def test_inflight_frames_cap():
    s = S.Scheduler(workers=10, max_inflight_frames=100)
    s.add("b1", "yt", clips("b1", [150, 60, 50, 30]))
    u, big = s.dispatch()
    assert big["frames"] == 150  # nothing in flight: a heavier clip runs alone
    assert s.dispatch() is None
    s.done("b1", big)
    u, r60 = s.dispatch()
    assert r60["frames"] == 60
    assert s.dispatch() is None  # 60 + 50 > 100: the head waits (no reordering in a unit)
    s.done("b1", r60)
    assert [s.dispatch()[1]["frames"], s.dispatch()[1]["frames"]] == [50, 30]
    assert s.inflight_frames == 80


def test_inflight_cap_lets_the_other_class_through():
    s = S.Scheduler(workers=10, max_inflight_frames=100)
    s.add("b1", "yt", clips("b1", [90, 90]))
    s.add("t", "t2", clips("t", [5, 5]))
    s.served["t2"] = 10**6  # yt would go first
    picks = []
    while (p := s.dispatch()) is not None:
        picks.append((p[0].cls, p[1]["frames"]))
    assert picks == [("yt", 90), ("t2", 5), ("t2", 5)]


# ---------------------------------------------------------------- simulation
def simulate_stream(units, workers, max_frames=0):
    """Event simulation of the streaming driver: unit i+1 is added when the queue of
    the class drops below `workers` (wants()); clip time = frames. Returns makespan,
    utilisation."""
    s = S.Scheduler(workers, max_frames)
    pending = list(units)
    t, busy, ev = 0.0, 0.0, []
    seq = 0
    while True:
        while pending and s.wants("yt"):
            name, fr = pending.pop(0)
            s.add(name, "yt", clips(name, fr))
        while (p := s.dispatch()) is not None:
            seq += 1
            heapq.heappush(ev, (t + p[1]["frames"], seq, p[0].name, p[1]))
            busy += p[1]["frames"]
        if not ev:
            break
        t, _, name, row = heapq.heappop(ev)
        if s.done(name, row):
            s.remove(name)
    return t, busy / (workers * t)


def simulate_per_unit(units, workers):
    """Old scheme: one pool per unit, the next unit starts after the last clip."""
    t, busy = 0.0, 0.0
    for name, fr in units:
        s = S.Scheduler(workers)
        s.add(name, "yt", clips(name, fr))
        ev, seq = [], 0
        while True:
            while (p := s.dispatch()) is not None:
                seq += 1
                heapq.heappush(ev, (t + p[1]["frames"], seq, p[1]))
                busy += p[1]["frames"]
            if not ev:
                break
            t, _, row = heapq.heappop(ev)
            s.done(name, row)
    return t, busy / (workers * t)


def _yt_like_batches(n=6, size=200, seed=0):
    rng = random.Random(seed)
    out = []
    for b in range(n):
        fr = [int(rng.lognormvariate(9.0, 0.9)) for _ in range(size)]
        if b == 1:
            fr[7] = 215595  # the b00002 straggler (one ~7 h video)
        out.append((f"b{b:05d}", fr))
    return out


def test_streaming_beats_one_pool_per_unit_on_long_tails():
    units = _yt_like_batches(n=12)  # ~1.6 h of 48-worker work per batch, one 215k-frame video
    t_old, u_old = simulate_per_unit(units, 48)
    t_new, u_new = simulate_stream(units, 48)
    ideal = sum(sum(fr) for _, fr in units) / 48
    assert u_old < 0.6  # each batch waits for its longest video
    assert u_new > 0.95 and t_new < 1.05 * ideal  # only the final tail idles
    assert t_new < 0.6 * t_old


def test_no_clip_is_lost_or_duplicated():
    units = _yt_like_batches(n=3, size=50, seed=3)
    s = S.Scheduler(5, 0)
    for name, fr in units:
        s.add(name, "yt", clips(name, fr))
    seen = []
    while (p := s.dispatch()) is not None:
        seen.append(p[1]["clip_id"])
        s.done(p[0].name, p[1])
    assert sorted(seen) == sorted(c["clip_id"] for n, fr in units for c in clips(n, fr))


# ---------------------------------------------------------------- driver end to end
h5py = pytest.importorskip("h5py")
pytest.importorskip("cv2")
from tests.test_reextract_pipeline import CFG_HASH, FakeEx, _video, ws  # noqa: E402,F401
from tools.extract import build_manifest as BM  # noqa: E402
from tools.extract import datasets as D  # noqa: E402
from tools.extract import extract_dataset as EX  # noqa: E402
from tools.extract import verify_dataset as V  # noqa: E402


class FakeSource:
    """Three yt 'batches' (datasets syn0/syn1 as class yt, syn2 as t2); post = verify."""

    def __init__(self, units):
        self.units = list(units)  # (ds, cls)
        self.opened, self.posted, self.marks = [], [], {}
        self.lock = threading.Lock()

    def next_unit(self, cls, active):
        for ds, c in self.units:
            if c == cls and ds not in self.opened:
                self.opened.append(ds)
                return (ds, None), False
        return None, True

    def verified(self, stem):
        return False

    def attempts(self, stem, bump=False):
        if bump:
            self.marks[stem] = self.marks.get(stem, 0) + 1
        return self.marks.get(stem, 0)

    def post(self, ds, batch, skip_verify=False):
        rep = V.verify(ds, cfg_hash=CFG_HASH)
        with self.lock:
            self.posted.append((ds, rep["status"]))


def _dataset(ws, monkeypatch, name, lengths):
    d = ws / "src" / name
    for i, n in enumerate(lengths):
        _video(str(d / f"c{i}.mp4"), n, seed=i)
    monkeypatch.setitem(D.DATASETS, name, dict(kind="files", split=None, roots=[(str(d), "")]))
    BM.build(name, workers=1, log=lambda m: None)


def test_driver_runs_units_on_one_pool_and_posts_each(ws, monkeypatch):
    _dataset(ws, monkeypatch, "syn0", [12, 3, 4, 5])
    _dataset(ws, monkeypatch, "syn1", [6, 2, 3])
    _dataset(ws, monkeypatch, "syn2", [4, 4])
    EX._EX = FakeEx(fail_on=["syn1" + os.sep + "c2", "syn1/c2"])  # one failure in syn1
    EX._CTX["frozen"] = None
    calls = []

    def fail_once(job):
        r = EX._job(job)
        calls.append(job[0]["clip_id"])
        if job[0]["clip_id"] == "syn1__c2" and r[1] != "ok":
            EX._EX.fail_on = set()  # the retry succeeds
        return r
    src = FakeSource([("syn0", "yt"), ("syn1", "yt"), ("syn2", "t2")])
    g = EX.MemGuard(None, rss_fn=lambda: 0, log=lambda m: None)
    logs = []
    d = S.Driver(src, lambda: ThreadPool(3), 3, g, logs.append, shard_clips=2,
                 poll=0.01, check_s=0.0, idle_s=0.0, progress_every=0.05, job_fn=fail_once)
    assert d.run() == "finished"
    assert sorted(src.posted) == [("syn0", "pass"), ("syn1", "pass"), ("syn2", "pass")]
    assert calls.count("syn1__c2") == 2 and src.marks == {"syn1": 1}  # one retry
    assert sorted(set(calls)) == sorted(
        [f"syn0__c{i}" for i in range(4)] + [f"syn1__c{i}" for i in range(3)] + ["syn2__c0", "syn2__c1"])
    # overlap: syn1 was opened (and started) before syn0's long clip finished
    first_syn1 = calls.index("syn1__c0") if "syn1__c0" in calls else len(calls)
    assert any(m.startswith("[tier2] opened syn1") for m in logs)
    assert first_syn1 < len(calls) - 1
    assert any("util" in m for m in logs)


def test_driver_stop_file_packs_and_resumes(ws, monkeypatch):
    _dataset(ws, monkeypatch, "syn0", [3, 3, 3, 3, 3])
    EX._EX = FakeEx()
    EX._CTX["frozen"] = None
    stop = ws / "STOP"
    n = [0]

    def job(j):
        n[0] += 1
        if n[0] == 2:
            stop.write_text("")
        return EX._job(j)
    src = FakeSource([("syn0", "yt")])
    g = EX.MemGuard(None, rss_fn=lambda: 0, log=lambda m: None)
    d = S.Driver(src, lambda: ThreadPool(1), 1, g, lambda m: None, poll=0.01, check_s=0.0,
                 idle_s=0.0, job_fn=job, stop_file=str(stop))
    assert d.run() == "stopped" and src.posted == []
    out = D.wpath("out", "syn0")
    assert 2 <= len(EX.shard_index(out)) < 5  # packed on stop
    os.remove(stop)
    src2 = FakeSource([("syn0", "yt")])
    d2 = S.Driver(src2, lambda: ThreadPool(2), 2, g, lambda m: None, poll=0.01, check_s=0.0,
                  idle_s=0.0, job_fn=EX._job)
    assert d2.run() == "finished" and src2.posted == [("syn0", "pass")]
    assert len(EX.shard_index(out)) == 5


@pytest.mark.skipif(os.name == "nt", reason="os.kill(pid, 0) probes liveness on POSIX only")
def test_source_defers_unit_owned_by_live_process(ws, monkeypatch):
    import subprocess
    import sys
    src = S.Source([], log=lambda m: None, defer={"youtube_asl_b00002": os.getpid()})
    for b in ("b00001", "b00002"):
        p = D.wpath("manifest", f"youtube_asl_{b}.csv")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write("clip_id,frames\nx,1\n")
    src.mark("youtube_asl_b00001", "done")
    monkeypatch.setattr(src, "yt_pending", lambda: 0)
    monkeypatch.setattr(S.Source, "downloading", staticmethod(lambda: False))
    monkeypatch.setattr(src, "refresh_staged", lambda: 0.0)
    assert src.next_unit("yt", set()) == (None, False)  # b00002 owned, not final
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    src.defer["youtube_asl_b00002"] = dead.pid
    assert src.next_unit("yt", set()) == (("youtube_asl", "b00002"), False)  # owner gone


def test_stop_file_older_than_start_is_ignored(tmp_path):
    import time as _t
    stop = tmp_path / "STOP"
    stop.write_text("")
    g = EX.MemGuard(None, rss_fn=lambda: 0, log=lambda m: None)
    d = S.Driver(FakeSource([]), lambda: ThreadPool(1), 1, g, lambda m: None,
                 stop_file=str(stop), stop_since=_t.time() + 5)
    assert not d._stopping()
    os.utime(stop, (_t.time() + 10, _t.time() + 10))  # touched again after start
    assert d._stopping()
