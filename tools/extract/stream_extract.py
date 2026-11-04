"""Tier-2 + YouTube-ASL streaming extraction on ONE long-lived worker pool (one CPU host).

  nice -n 19 python -m tools.extract.stream_extract --workers 48 --mem-budget-gb 60

Why: extract_dataset runs one unit (a dataset, or a YouTube-ASL batch of 200 videos) on
its own pool and waits for its last clip. YouTube-ASL videos can be hours long and one
MediaPipe worker does ~8 fps, so a single long video kept 47 of 48 workers idle for
hours (b00002: 199/200 clips after 2.3 h, the last one ~7 h).

Here every unit's clips go through one pool:
  * longest first within each unit (manifest frames, descending): long videos start
    first, so a unit's tail is short clips;
  * overlap: as soon as fewer than --workers clips of a class are queued, the next unit
    of that class (next YouTube batch / next tier-2 dataset) is opened (manifest built in
    a background thread) and its clips fill the free workers;
  * per-unit completion: when a unit's last clip is back it is finished like before
    (duplicates copied, shards packed, run summary, one retry of failed clips), then
    verify -> cleanup --scratch -> cleanup --delete-videos -> state/tier2/<stem>.done run
    in a background thread while the pool keeps working on other units;
  * a video is never split (MediaPipe video-mode tracking must stay continuous), and
    each clip is extracted exactly as by extract_dataset (same _job / worker init /
    frozen config), so the outputs are identical; only the order changes.

Classes and priority (the staging rule of the earlier shell driver): 'yt' = YouTube-ASL
batches, 't2' = tier-2 datasets. A YouTube batch is opened when YT_BATCH finished
downloads are pending, or any are pending and the downloader stopped or the staging dir
is above --yt-high-gb. While both classes have queued clips they share the dispatched
frames equally (replaces the 45-min tier-2 time slices); while the staging dir is above
--yt-high-gb YouTube clips go first. The summed manifest frames in flight stay below
--max-inflight-frames (long videos hold their arrays in the workers).

Kept from extract_dataset: memory guard over the whole process tree (pause dispatch above
--mem-budget-gb, pool restart if paused and idle), worker recycling, crash-safe staging
and resume (a restart re-opens unfinished units; staged clips are not redone).
Stop: touch ~/reextract/STOP -> no new clips; the clips in flight finish, staged clips
are packed, background steps finish, exit 0 (a restart resumes).
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import queue
import re
import subprocess
import sys
import threading
import time

from tools.extract import extract_dataset as EX
from tools.extract.datasets import DATASETS, unit_paths, unit_stem, wpath

YT = "youtube_asl"
CLASSES = ("yt", "t2")


def unit_class(ds: str) -> str:
    return "yt" if ds == YT else "t2"


# ---------------------------------------------------------------- scheduler (pure)
class SchedUnit:
    __slots__ = ("name", "cls", "queue", "inflight", "inflight_frames")

    def __init__(self, name: str, cls: str):
        self.name, self.cls = name, cls
        self.queue: list[dict] = []  # rows, longest first
        self.inflight = 0
        self.inflight_frames = 0

    @property
    def drained(self) -> bool:
        return not self.queue and self.inflight == 0


class Scheduler:
    """Which clip goes to the next free worker (no I/O; unit-tested).

    * within a unit: longest first (row["frames"] descending, ties by clip_id);
    * within a class: units in the order they were added (FIFO);
    * between classes: the class with fewer frames dispatched so far goes first; a class
      with nothing queued is caught up to the other, so it cannot claim a backlog when
      its next unit arrives. ``yt_first`` -> YouTube clips first;
    * the summed frames in flight stay <= max_inflight_frames (> 0) unless nothing is in
      flight (a heavier clip then runs alone); a head clip that does not fit waits (no
      backfilling with shorter clips of its class: that starves long videos for hours)
      and the other class's head may go instead;
    * ``wants(cls)``: fewer than ``prefetch`` (default ``workers``) clips of the class
      queued -> its next unit should be opened now, so it is ready before workers go
      idle (opening a YouTube batch builds its manifest: minutes)."""

    def __init__(self, workers: int, max_inflight_frames: int = 0, prefetch: int | None = None):
        self.workers, self.max_frames = workers, max_inflight_frames
        self.prefetch = prefetch or workers
        self.units: dict[str, SchedUnit] = {}
        self.served = {c: 0 for c in CLASSES}
        self.inflight = 0
        self.inflight_frames = 0

    def add(self, name: str, cls: str, rows: list[dict]) -> SchedUnit:
        u = self.units.get(name)
        if u is None:
            u = self.units[name] = SchedUnit(name, cls)
        u.queue = sorted(u.queue + list(rows), key=lambda r: (-int(r["frames"]), r["clip_id"]))
        return u

    def remove(self, name: str) -> None:
        u = self.units.pop(name)
        assert u.inflight == 0, f"{name} removed with {u.inflight} clips in flight"

    def queued(self, cls: str | None = None) -> int:
        return sum(len(u.queue) for u in self.units.values() if cls in (None, u.cls))

    def wants(self, cls: str) -> bool:
        return self.queued(cls) < self.prefetch

    def _head(self, cls: str) -> SchedUnit | None:
        for u in self.units.values():  # insertion order = FIFO
            if u.cls == cls and u.queue:
                return u
        return None

    def _fit(self, cls: str, room: int | None) -> tuple[SchedUnit, int] | None:
        """(unit, index) of the class's next clip: the longest queued clip of the first
        unit (FIFO) with queued clips, if it fits the frame budget ``room`` (None: no
        budget)."""
        u = self._head(cls)
        if u is None or (room is not None and int(u.queue[0]["frames"]) > room):
            return None
        return u, 0

    def order(self, yt_first: bool = False) -> list[str]:
        heads = {c: self._head(c) for c in CLASSES}
        top = max(self.served.values())
        for c in CLASSES:
            if heads[c] is None:
                self.served[c] = top
        live = [c for c in CLASSES if heads[c] is not None]
        if yt_first:
            return live
        return sorted(live, key=lambda c: (self.served[c], CLASSES.index(c)))

    def dispatch(self, yt_first: bool = False) -> tuple[SchedUnit, dict] | None:
        """Pop the next clip (and account it as in flight), or None."""
        if self.inflight >= self.workers:
            return None
        room = self.max_frames - self.inflight_frames if self.max_frames and self.inflight else None
        for c in self.order(yt_first):
            pick = self._fit(c, room)
            if pick is None:
                continue
            u, i = pick
            row = u.queue.pop(i)
            f = int(row["frames"])
            u.inflight += 1
            u.inflight_frames += f
            self.inflight += 1
            self.inflight_frames += f
            self.served[c] += f
            return u, row
        return None

    def done(self, name: str, row: dict) -> bool:
        """A clip of unit ``name`` came back; True if the unit is now drained."""
        u = self.units[name]
        f = int(row["frames"])
        u.inflight -= 1
        u.inflight_frames -= f
        self.inflight -= 1
        self.inflight_frames -= f
        return u.drained


class Util:
    """Pool utilisation: time-integral of the clips in flight / (workers x wall)."""

    def __init__(self, workers: int, clock=time.monotonic):
        self.workers, self.clock = workers, clock
        self.t0 = self.t = clock()
        self.busy = 0.0
        self.n = 0
        self.mark = (self.t0, 0.0)

    def set(self, n: int) -> None:
        now = self.clock()
        self.busy += self.n * (now - self.t)
        self.t, self.n = now, n

    def total(self) -> float:
        self.set(self.n)
        return self.busy / max(self.workers * (self.t - self.t0), 1e-9)

    def window(self) -> float:
        """Utilisation since the previous window() call."""
        self.set(self.n)
        t0, b0 = self.mark
        self.mark = (self.t, self.busy)
        return (self.busy - b0) / max(self.workers * (self.t - t0), 1e-9)


# ---------------------------------------------------------------- units
class Source:
    """Which unit to open next, manifests, markers and the after-extraction steps
    (verify / cleanup / delete videos), as the earlier shell driver did. Paths under
    REEXTRACT_ROOT; state markers in state/tier2/."""

    def __init__(self, tier2: list[str], yt_batch: int = 200, yt_high_gb: float = 200,
                 manifest_workers: int = 16, verify_workers: int = 16, log=print,
                 yt_dest: str | None = None, py: str = sys.executable,
                 defer: dict[str, int] | None = None):
        self.tier2, self.yt_batch, self.yt_high_gb = tier2, yt_batch, yt_high_gb
        self.defer = dict(defer or {})  # unit stem -> pid that owns it while alive
        self.manifest_workers, self.verify_workers = manifest_workers, verify_workers
        self.log, self.py = log, py
        from tools.extract.datasets import YTASL_DEST
        self.yt_dest = yt_dest or YTASL_DEST
        self.st = wpath("state", "tier2")
        os.makedirs(self.st, exist_ok=True)
        self.staged_gb = 0.0

    def deferred(self, stem: str) -> bool:
        """The unit is owned by another live process (e.g. the old driver finishing it)."""
        pid = self.defer.get(stem)
        if pid is None:
            return False
        try:
            os.kill(pid, 0)
            return True
        except PermissionError:
            return True
        except OSError:  # ProcessLookupError (or the Windows equivalent)
            return False

    # markers
    def finished(self, stem: str) -> bool:
        return any(os.path.exists(os.path.join(self.st, f"{stem}.{s}")) for s in ("done", "failed"))

    def mark(self, stem: str, what: str) -> None:
        open(os.path.join(self.st, f"{stem}.{what}"), "a").close()

    def attempts(self, stem: str, bump: bool = False) -> int:
        p = os.path.join(self.st, f"{stem}.attempts")
        try:
            a = int(open(p).read().strip() or 0)
        except (OSError, ValueError):
            a = 0
        if bump:
            a += 1
            with open(p, "w") as f:
                f.write(f"{a}\n")
        return a

    @staticmethod
    def verified(stem: str) -> bool:
        p = wpath("reports", f"{stem}_verify.json")
        try:
            return json.load(open(p)).get("status") == "pass"
        except (OSError, ValueError):
            return False

    # host probes
    def sub(self, *args: str, env: dict | None = None) -> int:
        """python -m <args> (inherits nice/CPU affinity/stdout of this process)."""
        e = dict(os.environ, **(env or {}))
        return subprocess.call([self.py, "-m", *args], env=e)

    def refresh_staged(self) -> float:
        try:
            out = subprocess.run(["du", "-sb", os.path.join(self.yt_dest, "videos")],
                                 capture_output=True, text=True, timeout=600).stdout
            self.staged_gb = int(out.split()[0]) / 1e9
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            pass
        return self.staged_gb

    def yt_first(self) -> bool:
        return self.staged_gb >= self.yt_high_gb

    @staticmethod
    def downloading() -> bool:
        """A YouTube-ASL download is running: a process whose command line matches
        $YTASL_DOWNLOADER_PATTERN (default: none -> False; the downloader is not part of this
        release)."""
        pat = os.environ.get("YTASL_DOWNLOADER_PATTERN")
        if not pat:
            return False
        return subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0

    def yt_batches(self) -> list[str]:
        out = []
        for p in glob.glob(wpath("manifest", f"{YT}_b*.csv")):
            m = re.fullmatch(rf"{YT}_(b\d{{5}})\.csv", os.path.basename(p))
            if m:
                out.append(m.group(1))
        return sorted(out)

    def yt_skipfile(self) -> str:
        f = os.path.join(self.st, "yt_skip.txt")
        with open(f + ".tmp", "w", encoding="utf-8") as fo:
            for b in self.yt_batches():
                with open(wpath("manifest", f"{YT}_{b}.csv"), encoding="utf-8") as fi:
                    next(fi, None)
                    for line in fi:
                        if line.strip():
                            fo.write(line.split(",", 1)[0] + "\n")
        os.replace(f + ".tmp", f)
        return f

    def yt_pending(self) -> int:
        env = dict(os.environ, REEXTRACT_QUEUE_SKIP=self.yt_skipfile())
        try:
            out = subprocess.run([self.py, "-c", "from tools.extract.datasets import "
                                  f"enumerate_sources as e; print(len(e('{YT}')))"],
                                 env=env, capture_output=True, text=True, timeout=1800).stdout
            return int(out.strip().splitlines()[-1])
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return 0

    # opening
    def next_unit(self, cls: str, active: set[str]) -> tuple[tuple[str, str | None] | None, bool]:
        """((ds, batch) to open, or None; final) -- final: nothing more will come for
        this class (tier-2 list exhausted / no download pending and downloader stopped)."""
        if cls == "t2":
            for ds in self.tier2:
                if self.finished(ds) or ds in active or self.deferred(ds):
                    continue
                if self._ensure_manifest(ds, None):
                    return (ds, None), False
            return None, not any(not self.finished(d) and d not in active for d in self.tier2)
        owned = False
        for b in self.yt_batches():
            stem = unit_stem(YT, b)
            if self.deferred(stem):
                owned = owned or not self.finished(stem)
                continue
            if not self.finished(stem) and stem not in active:
                return (YT, b), False
        pend = self.yt_pending()
        dl = self.downloading()
        staged = self.refresh_staged()
        due = pend >= self.yt_batch or (pend > 0 and (not dl or staged >= self.yt_high_gb))
        if not due:
            return None, pend == 0 and not dl and not owned
        b = f"b{len(self.yt_batches()) + 1:05d}"
        self.log(f"[tier2] youtube batch {b}: building manifest (<= {self.yt_batch} of {pend} "
                 f"pending downloads, staged {staged:.0f} GB)")
        if not self._ensure_manifest(YT, b):
            return None, False
        return (YT, b), False

    def _ensure_manifest(self, ds: str, batch: str | None) -> bool:
        up = unit_paths(ds, batch)
        stem, man = up["stem"], up["manifest"]
        if os.path.exists(man) and os.path.getsize(man) > 0:
            return True
        self.log(f"[tier2] manifest {stem}")
        args = ["tools.extract.build_manifest", ds, "--workers", str(self.manifest_workers)]
        env = None
        if batch:
            args += ["--batch", batch, "--limit", str(self.yt_batch)]
            env = dict(REEXTRACT_QUEUE_SKIP=self.yt_skipfile())
        rc = self.sub(*args, env=env)
        if rc:
            self.log(f"[tier2] manifest {stem} exit {rc} (unreadable sources in "
                     f"manifest/{stem}_manifest_errors.csv)")
        rows = 0
        if os.path.exists(man):
            with open(man, encoding="utf-8") as f:
                rows = sum(1 for _ in f) - 1
        if batch and rows <= 0:
            self.log(f"[tier2] youtube batch {batch}: empty; removed")
            for p in glob.glob(wpath("manifest", f"{stem}*")):
                os.remove(p)
            return False
        if not os.path.exists(man) or os.path.getsize(man) == 0:
            self.log(f"[tier2] {stem}: no manifest written; marking failed")
            self.mark(stem, "failed")
            return False
        try:
            summ = json.load(open(wpath("manifest", f"{stem}_summary.json")))
            self.log(f"[tier2] manifest {stem}: {json.dumps(summ, separators=(',', ':'))[:400]}")
        except (OSError, ValueError):
            pass
        return True

    # after extraction
    def post(self, ds: str, batch: str | None, skip_verify: bool = False) -> None:
        up = unit_paths(ds, batch)
        stem = up["stem"]
        bargs = ["--batch", batch] if batch else []
        if not skip_verify:
            self.log(f"[tier2] verify {stem}")
            if self.sub("tools.extract.verify_dataset", ds, *bargs,
                        "--workers", str(self.verify_workers)) == 0:
                self.log(f"[tier2] verify {stem} PASS")
                open(os.path.join(up["out_dir"], "_READY_FOR_SYNC"), "a").close()
            else:
                self.log(f"[tier2] verify {stem} FAIL (reports/{stem}_verify.json); videos kept; "
                         f"marking failed")
                self.mark(stem, "failed")
                return
        rc = self.sub("tools.extract.cleanup", ds, *bargs, "--scratch")
        if rc:
            self.log(f"[tier2] cleanup --scratch {stem} exit {rc}")
        if ds == YT or DATASETS[ds].get("downloaded_by_us"):
            rc = self.sub("tools.extract.cleanup", ds, *bargs, "--delete-videos")
            if rc:
                self.log(f"[tier2] cleanup --delete-videos {stem} exit {rc}")
        self.mark(stem, "done")
        self.log(f"[tier2] {stem} done")


# ---------------------------------------------------------------- driver
class Driver:
    """One pool, many units. ``src`` provides next_unit / attempts / verified / post /
    yt_first / refresh_staged (Source; a fake in tests). ``make_pool()`` returns a
    multiprocessing-like pool whose workers run ``job_fn`` (EX._job)."""

    def __init__(self, src, make_pool, workers: int, guard: EX.MemGuard, log=print,
                 max_inflight_frames: int = 0, shard_clips: int = 1000, frozen: dict | None = None,
                 extractor: str | None = None, stop_file: str | None = None, poll: float = 1.0,
                 check_s: float = 60.0, idle_s: float = 300.0, progress_every: float = 60.0,
                 job_fn=EX._job, unit_log_every: float = 300.0, prefetch: int | None = None,
                 refresh_s: float = 300.0, max_tasks_per_child: int | None = None,
                 stop_since: float | None = None):
        self.src, self.make_pool, self.workers, self.guard, self.log = src, make_pool, workers, guard, log
        self.sched = Scheduler(workers, max_inflight_frames, prefetch)
        self.shard_clips, self.frozen, self.extractor = shard_clips, frozen, extractor
        self.stop_file, self.poll, self.check_s, self.idle_s = stop_file, poll, check_s, idle_s
        self.progress_every, self.job_fn, self.unit_log_every = progress_every, job_fn, unit_log_every
        self.runs: dict[str, EX.UnitRun] = {}
        self.meta: dict[str, dict] = {}  # units opened and not yet done (incl. post steps)
        self.lock = threading.Lock()
        self.results: queue.Queue = queue.Queue()
        self.opened: queue.Queue = queue.Queue()
        self.opener: threading.Thread | None = None
        self.posts: list[threading.Thread] = []
        self.retry_at = {c: 0.0 for c in CLASSES}
        self.final = {c: False for c in CLASSES}
        self.util = Util(workers)
        self.clip_s = 0.0
        self.pool_restarts = 0
        self.yt_first = False
        self.order: list[str] = []  # unit stems in open order (log)
        self.refresh_s, self.refresh_at = refresh_s, 0.0
        self.max_tasks_per_child = max_tasks_per_child
        self.stop_since = stop_since  # ignore a stop file older than this (epoch s)

    # -- opening (background thread)
    def _open_task(self, want: list[str], active: set[str]) -> None:
        try:
            if hasattr(self.src, "refresh_staged"):
                self.src.refresh_staged()
            for cls in want:
                spec, final = self.src.next_unit(cls, active)
                if spec is None:
                    self.opened.put(("none", cls, final))
                    continue
                ds, batch = spec
                stem = unit_stem(ds, batch)
                active.add(stem)
                if self.src.verified(stem):
                    self.opened.put(("verified", cls, (ds, batch, stem)))
                    continue
                up = unit_paths(ds, batch)
                u = EX.UnitRun(ds, up["manifest"], up["out_dir"], self.shard_clips, None,
                               self.frozen, self.extractor, self.log, self.unit_log_every,
                               self.guard)
                self.opened.put(("unit", cls, (ds, batch, stem, u)))
        except Exception as e:  # noqa: BLE001 - keep the pool running; retried later
            import traceback
            self.log(f"[tier2] opening failed: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            for cls in want:
                self.opened.put(("none", cls, False))

    def _maybe_open(self, now: float) -> None:
        if self.opener is not None and self.opener.is_alive():
            return
        want = [c for c in CLASSES if self.sched.wants(c) and now >= self.retry_at[c]]
        if not want and now < self.refresh_at:
            return
        self.refresh_at = now + self.refresh_s  # the opener also re-measures the staging dir
        if not self.yt_first:  # the class closer to running dry first
            want.sort(key=lambda c: (self.sched.queued(c), CLASSES.index(c)))
        for c in want:
            self.retry_at[c] = now + self.check_s  # no re-launch until this one reports
        with self.lock:
            active = set(self.meta)
        self.opener = threading.Thread(target=self._open_task, args=(want, active), daemon=True)
        self.opener.start()

    def _take_opened(self, now: float) -> None:
        while True:
            try:
                kind, cls, val = self.opened.get_nowait()
            except queue.Empty:
                return
            if kind == "none":
                self.final[cls] = val
                self.retry_at[cls] = now + self.idle_s
                continue
            self.final[cls] = False
            self.retry_at[cls] = now  # look for the following unit as soon as it is wanted
            if kind == "verified":
                ds, batch, stem = val
                self.log(f"[tier2] {stem}: already verified; cleanup only")
                with self.lock:
                    self.meta[stem] = dict(ds=ds, batch=batch, cls=cls, t0=time.time())
                self._post(stem, skip_verify=True)
                continue
            ds, batch, stem, u = val
            self.runs[stem] = u
            with self.lock:
                self.meta[stem] = dict(ds=ds, batch=batch, cls=cls, t0=time.time(), dups_done=False)
            self.order.append(stem)
            self.sched.add(stem, cls, u.primary)
            self.log(f"[tier2] opened {stem} ({cls}): {len(u.primary)} clips to extract "
                     f"({sum(r['frames'] for r in u.primary)} frames), queued now "
                     f"yt {self.sched.queued('yt')} / t2 {self.sched.queued('t2')}, "
                     f"in flight {self.sched.inflight}/{self.workers}")
            if not u.primary:
                self._drained(stem)

    # -- per-unit completion (main thread)
    def _drained(self, stem: str) -> None:
        u, m = self.runs[stem], self.meta[stem]
        if not m["dups_done"]:
            m["dups_done"] = True
            redo = u.copy_dups()
            if redo:
                self.sched.add(stem, m["cls"], redo)
                return
        s = u.finish(self.workers, mem_budget_gb=self.guard.budget / 2**30 or None,
                     max_tasks_per_child=self.max_tasks_per_child,
                     max_inflight_frames=self.sched.max_frames, stream=True)
        if s["failures"] and self.src.attempts(stem) < 1:
            a = self.src.attempts(stem, bump=True)
            self.log(f"[tier2] extract {stem}: {s['failures']} failures (attempt {a}); retrying them")
            u2 = EX.UnitRun(m["ds"], u.manifest, u.out_dir, self.shard_clips, None, self.frozen,
                            self.extractor, self.log, self.unit_log_every, self.guard)
            m["dups_done"] = False
            self.runs[stem] = u2
            if u2.primary:
                self.sched.add(stem, m["cls"], u2.primary)
            else:
                self._drained(stem)
            return
        if s["failures"]:
            a = self.src.attempts(stem, bump=True)
            self.log(f"[tier2] extract {stem}: {s['failures']} failures (attempt {a}; failures in "
                     f"{u.out_dir}/failures.csv); verifying anyway")
        self.sched.remove(stem)
        del self.runs[stem]
        self._post(stem)

    def _post(self, stem: str, skip_verify: bool = False) -> None:
        m = self.meta[stem]

        def go():
            try:
                self.src.post(m["ds"], m["batch"], skip_verify=skip_verify)
            except Exception as e:  # noqa: BLE001
                self.log(f"[tier2] post-processing {stem} failed: {type(e).__name__}: {e}")
            finally:
                with self.lock:
                    self.meta.pop(stem, None)
        t = threading.Thread(target=go, daemon=True, name=f"post-{stem}")
        self.posts.append(t)
        t.start()

    # -- main loop
    def _stopping(self) -> bool:
        if not self.stop_file:
            return False
        try:
            mt = os.path.getmtime(self.stop_file)
        except OSError:
            return False
        return self.stop_since is None or mt >= self.stop_since

    def _progress(self) -> None:
        g = self.guard
        units = ", ".join(
            f"{s} {r.stats['clips']}/{len(r.primary)} (q {len(self.sched.units[s].queue)}, "
            f"run {self.sched.units[s].inflight})" for s, r in self.runs.items() if s in self.sched.units)
        self.log(f"[tier2] pool {self.sched.inflight}/{self.workers} busy, util {self.util.window():.1%} "
                 f"(since start {self.util.total():.1%}), frames in flight {self.sched.inflight_frames}, "
                 f"RSS {g.rss / 2**30:.1f} GB (peak {g.peak / 2**30:.1f}, pauses {g.pauses}), "
                 f"staged-first {self.yt_first}; units: {units or '-'}; post: "
                 f"{sum(t.is_alive() for t in self.posts)}")

    def run(self) -> str:
        """Returns 'finished' (nothing more to do) or 'stopped' (STOP file)."""
        pool = self.make_pool()
        last_prog = time.monotonic()
        state = "stopped"
        try:
            while True:
                now = time.monotonic()
                stopping = self._stopping()
                self._take_opened(now)
                if hasattr(self.src, "yt_first"):
                    self.yt_first = self.src.yt_first()
                opening = self.opener is not None and self.opener.is_alive()
                if (self.sched.inflight == 0 and not self.sched.units and not opening
                        and self.opened.empty() and all(self.final.values())):
                    state = "finished"
                    break
                if not stopping:
                    self._maybe_open(now)
                paused = self.guard.check()
                while not stopping and not paused:
                    pick = self.sched.dispatch(self.yt_first)
                    if pick is None:
                        break
                    su, row = pick
                    job = self.runs[su.name].job(row)
                    name = su.name
                    pool.apply_async(
                        self.job_fn, (job,),
                        callback=lambda r, n=name, rw=row: self.results.put((n, rw, r)),
                        error_callback=lambda e, n=name, rw=row: self.results.put(
                            (n, rw, (rw["clip_id"], "error", 0, 0.0, f"{type(e).__name__}: {e}"))))
                    self.util.set(self.sched.inflight)
                opening = self.opener is not None and self.opener.is_alive()
                if self.sched.inflight == 0:
                    if stopping and not opening:
                        break
                    if paused and self.guard.check(force=True):
                        self.log(f"[tier2] memguard: paused with no clip in flight (RSS "
                                 f"{self.guard.rss / 2**30:.1f} GB); restarting the pool")
                        pool.close()
                        pool.join()
                        pool = self.make_pool()
                        self.pool_restarts += 1
                        self.guard.check(force=True)
                if now - last_prog >= self.progress_every:
                    last_prog = now
                    self._progress()
                try:
                    name, row, res = self.results.get(timeout=self.poll)
                except queue.Empty:
                    continue
                self.clip_s += res[3]
                drained = self.sched.done(name, row)
                self.util.set(self.sched.inflight)
                self.runs[name].on_result(res)
                if drained:
                    self._drained(name)
        finally:
            pool.close()
            pool.join()
        for stem, u in list(self.runs.items()):  # stopped: pack what is staged
            u.finish(self.workers, sliced=True, stream=True)
        if self.opener is not None:
            self.opener.join()
        for t in self.posts:
            t.join()
        self.log(f"[tier2] driver {state}: util {self.util.total():.1%}, pool restarts "
                 f"{self.pool_restarts}, units opened {len(self.order)}")
        return state


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--tier2", default="ghanaian_sl lsa64 sign2mint lsa_t slovo karsl chicagofswild")
    ap.add_argument("--mem-budget-gb", type=float, default=60)
    ap.add_argument("--max-tasks-per-child", type=int, default=4)
    ap.add_argument("--mem-check-s", type=float, default=3.0)
    ap.add_argument("--max-inflight-frames", type=int, default=1200000)
    ap.add_argument("--shard-clips", type=int, default=1000)
    ap.add_argument("--yt-batch", type=int, default=200)
    ap.add_argument("--yt-high-gb", type=float, default=200)
    ap.add_argument("--manifest-workers", type=int, default=16)
    ap.add_argument("--verify-workers", type=int, default=16)
    ap.add_argument("--idle-s", type=float, default=300)
    ap.add_argument("--prefetch", type=int, default=0,
                    help="open a class's next unit when fewer clips are queued (0 = 2 x workers)")
    ap.add_argument("--defer", action="append", default=[], metavar="STEM:PID",
                    help="leave unit STEM alone while process PID lives (it is being "
                         "finished by another driver); opened normally afterwards if not done")
    ap.add_argument("--extra-rss-pid", type=int, action="append", default=[],
                    help="also count the RSS of this process tree against --mem-budget-gb")
    ap.add_argument("--ignore-existing-stop", action="store_true",
                    help="a STOP file older than this driver's start does not stop it "
                         "(touching it again does)")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")  # read by glibc at worker start
    from tools.extract.freeze_config import CONFIG_PATH, check_frozen, load_frozen
    frozen = load_frozen(a.config or CONFIG_PATH)
    errs = check_frozen(frozen)
    if errs:
        raise SystemExit("refusing to run: " + "; ".join(errs))

    def log(m):
        print(time.strftime("%F %T"), m, flush=True)
    tier2 = a.tier2.split()
    for ds in tier2:
        assert ds in DATASETS, ds
    defer = {d.rsplit(":", 1)[0]: int(d.rsplit(":", 1)[1]) for d in a.defer}
    src = Source(tier2, a.yt_batch, a.yt_high_gb, a.manifest_workers, a.verify_workers, log,
                 defer=defer)
    extra = list(a.extra_rss_pid)

    def rss():
        return EX.tree_rss_bytes() + sum(EX.tree_rss_bytes(p) for p in extra)
    guard = EX.MemGuard(a.mem_budget_gb * 2**30, rss_fn=rss, interval=a.mem_check_s,
                        log=lambda m: log(f"[tier2] {m}"))
    t_start = time.time()
    ctx = mp.get_context("spawn")
    name = frozen["name"]

    def make_pool():
        return ctx.Pool(a.workers, initializer=EX._init, initargs=(name, frozen),
                        maxtasksperchild=a.max_tasks_per_child or None)
    log(f"[tier2] stream driver pid {os.getpid()}: workers {a.workers}, budget {a.mem_budget_gb} GB, "
        f"inflight frames <= {a.max_inflight_frames}, tier2 {tier2}, yt batch {a.yt_batch}, "
        f"high {a.yt_high_gb} GB, deferred {defer}, extra RSS pids {extra}")
    d = Driver(src, make_pool, a.workers, guard, log, a.max_inflight_frames, a.shard_clips,
               frozen, name, stop_file=wpath("STOP"), idle_s=a.idle_s,
               prefetch=a.prefetch or 2 * a.workers, max_tasks_per_child=a.max_tasks_per_child,
               stop_since=t_start if a.ignore_existing_stop else None)
    state = d.run()
    if state == "finished":
        open(os.path.join(src.st, "FINISHED"), "a").close()
    raise SystemExit(0)


if __name__ == "__main__":
    main()
