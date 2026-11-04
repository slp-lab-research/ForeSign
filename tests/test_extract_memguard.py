"""Memory guard of the dataset extractor: pause/resume hysteresis on a mocked RSS, the
sampling interval, and the guarded dispatch loop (thread pool stand-in)."""
import os
import time
from multiprocessing.dummy import Pool as ThreadPool

import pytest

from tools.extract.extract_dataset import MemGuard, guarded_map, tree_rss_bytes

GB = 2**30


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _guard(seq, budget=10, interval=0.0, clock=None):
    it = iter(seq)
    return MemGuard(budget * GB, rss_fn=lambda: next(it) * GB, interval=interval,
                    clock=clock or Clock(), log=lambda m: None)


def test_pause_above_budget_resume_below_80pct():
    g = _guard([5, 9.9, 10.5, 9.5, 8.1, 7.9, 9.9, 10.1])
    got = [g.check() for _ in range(8)]
    # 10.5 > 10 pauses; 9.5 and 8.1 stay paused (>= 8); 7.9 resumes; 9.9 not paused
    assert got == [False, False, True, True, True, False, False, True]
    assert g.pauses == 2 and g.peak == 10.5 * GB


def test_interval_rate_limits_sampling():
    clk = Clock()
    calls = []

    def rss():
        calls.append(clk.t)
        return 11 * GB
    g = MemGuard(10 * GB, rss_fn=rss, interval=3.0, clock=clk, log=lambda m: None)
    assert g.check() is True
    clk.t = 1.0
    assert g.check() is True and len(calls) == 1  # cached
    clk.t = 3.5
    g.check()
    assert len(calls) == 2
    g.check(force=True)
    assert len(calls) == 3


def test_no_budget_never_pauses_but_tracks_peak():
    g = MemGuard(None, rss_fn=iter([1 * GB, 50 * GB, 2 * GB]).__next__, interval=0,
                 clock=Clock(), log=lambda m: None)
    assert [g.check() for _ in range(3)] == [False, False, False]
    assert g.peak == 50 * GB and g.pauses == 0


def test_paused_time_accumulates():
    clk = Clock()
    vals = iter([11, 11, 5])
    g = MemGuard(10 * GB, rss_fn=lambda: next(vals) * GB, interval=0, clock=clk,
                 log=lambda m: None)
    g.check()
    clk.t = 4.0
    g.check()
    clk.t = 7.0
    assert g.check() is False and g.paused_s == 7.0


def _has_rss_source():
    import importlib.util
    return importlib.util.find_spec("psutil") is not None or os.path.isdir("/proc")


@pytest.mark.skipif(not _has_rss_source(), reason="needs psutil or /proc (Linux)")
def test_tree_rss_is_positive():
    assert tree_rss_bytes() > 0


class ScriptedGuard:
    """Paused for the first `n` checks."""

    def __init__(self, n):
        self.n, self.calls, self.rss, self.pauses = n, 0, 0, 0

    def check(self, force=False):
        self.calls += 1
        return self.calls <= self.n


def test_guarded_map_runs_all_jobs_and_caps_inflight():
    live, peak = [0], [0]

    def fn(x):
        live[0] += 1
        peak[0] = max(peak[0], live[0])
        time.sleep(0.01)
        live[0] -= 1
        return x * 2
    out = []
    g = MemGuard(None, rss_fn=lambda: 0, log=lambda m: None)
    r = guarded_map(lambda: ThreadPool(4), fn, range(20), 3, g, out.append, poll=0.05)
    assert sorted(out) == [2 * i for i in range(20)] and r == 0
    assert peak[0] <= 3


def test_guarded_map_restarts_pool_when_paused_and_idle():
    made = []

    def make():
        made.append(1)
        return ThreadPool(2)
    out = []
    # paused at the start with nothing in flight -> one restart, then runs normally
    r = guarded_map(make, lambda x: x, range(5), 2, ScriptedGuard(2), out.append,
                    log=lambda m: None, poll=0.01)
    assert sorted(out) == list(range(5)) and r == 1 and len(made) == 2


def test_guarded_map_maps_errors():
    def fn(x):
        if x == 3:
            raise ValueError("boom")
        return ("ok", x)
    out = []
    g = MemGuard(None, rss_fn=lambda: 0, log=lambda m: None)
    guarded_map(lambda: ThreadPool(2), fn, range(5), 2, g, out.append,
                on_error=lambda j, e: ("err", j), poll=0.01)
    assert sorted(out) == [("err", 3)] + [("ok", i) for i in (0, 1, 2, 4)]
