"""Extract one dataset with the frozen extractor into sharded HDF5 (spec section 4, step 2).

  nice -n 19 python -m tools.extract.extract_dataset autsl --workers 48

Reads manifest/<ds>.csv, writes out/<ds>/shard_NNNNN.h5. Each shard holds one group per
clip:
  <clip_id>/kp     (T, 553, 4) float16  x, y (image-normalised), z, score
                                         pose33 | face478 | left hand21 | right hand21
  <clip_id>/world  (T, 33, 4)  float16  pose world landmarks (m, hip-centred), visibility
  <clip_id>/flags  (T, 3)      uint8    [pose found, left hand src, right hand src]
                                         hand src: 0 none, 1 holistic, 2 crop step
  attrs: fps, width, height, n_frames, extractor, extractor_version, config_hash,
         frozen_config_sha256, code_git_sha, clip_id, dataset, source, split, ...

Crash-safe and resumable:
  * workers write one atomic per-clip file to out/<ds>/_staging/<clip_id>.h5;
  * the main process packs staged clips into a new shard (written to .tmp, then renamed)
    every --shard-clips clips and at the end, then deletes the packed staging files;
  * on start, clips found in any shard or in staging are skipped, leftover .tmp files
    are removed, and staged clips already packed are dropped.
Failures are appended to out/<ds>/failures.csv and retried on the next run.
Rows with ``dup_of`` (identical content to an already extracted clip, e.g. PHOENIX-2014
sequences byte-identical to PHOENIX-2014T ones) get a copy of that clip's arrays.

Memory: mediapipe leaks ~150-200 MB of RSS per clip inside each worker (Holistic is
created and closed per clip; the leak is in the library), so workers are recycled every
--max-tasks-per-child clips (default 4). In addition a parent-side guard (MemGuard) sums
the RSS of this process and all its descendants every --mem-check-s seconds and stops
dispatching new clips while the total is above --mem-budget-gb, resuming once it falls
below 80 % of it. If dispatch is paused and no clip is in flight (idle workers holding
leaked memory), the pool is restarted. The peak total RSS is logged and recorded.
Most of the growth is glibc heap fragmentation across mediapipe's threads: main() sets
MALLOC_ARENA_MAX=2 (inherited by the spawned workers) and each worker calls malloc_trim
after every clip, which halves the per-worker RSS (measured with 16 workers, 160 AUTSL
clips: peak 26.1 -> 10.6 GB, max worker 1.8 -> 0.7 GB, same fps). Neither changes the
outputs.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import multiprocessing as mp
import os
import re
import shutil
import time
import traceback

import numpy as np

from tools.extract.build_manifest import read_manifest
from tools.extract.datasets import DATASETS, load_excludes, unit_paths, wpath

FLAGS_LAYOUT = "pose_found,left_hand_src,right_hand_src (src 0 none/1 holistic/2 crop)"
SHARD_RE = re.compile(r"shard_(\d{5})\.h5$")

_EX = None
_CTX: dict = {}


# ---------------------------------------------------------------- per-clip I/O
def write_staged(path: str, kp, world, flags, attrs: dict) -> None:
    import h5py
    T = kp.shape[0]
    assert kp.shape[1:] == (553, 4) and world.shape == (T, 33, 4) and flags.shape == (T, 3)
    comp = dict(compression="gzip", compression_opts=4, shuffle=True) if T else {}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with h5py.File(tmp, "w") as f:
        f.create_dataset("kp", data=np.nan_to_num(kp).astype(np.float16), **comp)
        f.create_dataset("world", data=np.nan_to_num(world).astype(np.float16), **comp)
        f.create_dataset("flags", data=flags.astype(np.uint8), **comp)
        for k, v in attrs.items():
            f.attrs[k] = v
    os.replace(tmp, path)


def clip_attrs(res_attrs: dict, row: dict, frozen: dict | None) -> dict:
    a = {k: v for k, v in res_attrs.items() if isinstance(v, (int, float, str, np.integer, np.floating))}
    a.update(clip_id=row["clip_id"], dataset=row["dataset"], source=row["source"],
             split=row.get("split", ""), flags_layout=FLAGS_LAYOUT,
             manifest_frames=int(row["frames"]))
    if row.get("kind") == "frames" or res_attrs.get("fps") is None:
        a["fps"] = float(row["fps"])
    if frozen:
        a.update(frozen_config=frozen["name"], frozen_config_sha256=frozen["_file_sha256"],
                 code_git_sha=frozen["code_git_sha"])
    return a


# ---------------------------------------------------------------- workers
def _init(name: str, frozen: dict | None):
    global _EX
    _CTX["frozen"] = frozen
    from tools.extract.extractors import make
    _EX = make(name)
    if frozen:
        from tools.extract.freeze_config import check_frozen
        errs = check_frozen(frozen, _EX)
        if errs:
            raise RuntimeError("frozen config mismatch: " + "; ".join(errs))


def _local_copy(row: dict) -> str | None:
    """Copy a video source to fast local storage before decoding (REEXTRACT_LOCAL_COPY dir).
    Hour-long sequential reads of long videos over NFS occasionally stop early and the clip
    comes back short (framecount mismatch); one bulk copy, checked against the manifest
    size, avoids that. Same bytes in, so outputs are unchanged."""
    d = os.environ.get("REEXTRACT_LOCAL_COPY")
    if not d or row.get("kind") != "video" or not os.path.isfile(row["path"]):
        return None
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):  # copies left by workers that were killed mid-clip
        pid = f.split("_", 1)[0]
        if pid.isdigit() and not os.path.exists(f"/proc/{pid}"):
            try:
                os.remove(os.path.join(d, f))
            except OSError:
                pass
    dst = os.path.join(d, f"{os.getpid()}_{os.path.basename(row['path'])}")
    shutil.copyfile(row["path"], dst)
    want, got = int(row.get("size_bytes") or 0), os.path.getsize(dst)
    if want and got != want:
        os.remove(dst)
        raise OSError(f"local copy size {got} != manifest {want}: {row['path']}")
    return dst


def extract_one(ex, row: dict, staged: str, frozen: dict | None):
    local = _local_copy(row)
    try:
        res = ex.run(local or row["path"], float(row["fps"]))
    finally:
        if local:
            os.remove(local)
    T = res.kp.shape[0]
    src = res.extra.get("hand_src")
    if src is None:
        src = np.zeros((T, 2), np.uint8)
    flags = np.concatenate([res.valid.reshape(T, 1).astype(np.uint8),
                            np.asarray(src, np.uint8).reshape(T, 2)], 1)
    world = res.world[:, :33]
    write_staged(staged, res.kp, world, flags, clip_attrs(res.attrs, row, frozen))
    return T, float(res.attrs.get("elapsed_s", 0.0))


def load_defer() -> set:
    """Clip ids another host extracts (REEXTRACT_DEFER_LIST: one id per line). They are left
    out of this run's work list; once their staged files are copied into out/<ds>/_staging
    a run without the list packs them and verify sees the complete dataset."""
    p = os.environ.get("REEXTRACT_DEFER_LIST")
    if not p:
        return set()
    with open(p) as f:
        return {ln.strip() for ln in f if ln.strip()}


def _malloc_trim() -> None:
    """Return freed heap pages to the OS (glibc); no effect on the outputs."""
    if os.environ.get("REEXTRACT_MALLOC_TRIM", "1") != "1":
        return
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _job(args):
    row, staged = args
    try:
        try:
            T, el = extract_one(_EX, row, staged, _CTX.get("frozen"))
        finally:
            _malloc_trim()
        if T != row["frames"]:
            os.remove(staged)  # never pack an output whose length disagrees
            return row["clip_id"], f"framecount {T}!={row['frames']}", T, el, ""
        return row["clip_id"], "ok", T, el, ""
    except Exception as e:  # noqa: BLE001
        return row["clip_id"], "error", 0, 0.0, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


# ---------------------------------------------------------------- memory guard
def tree_rss_bytes(pid: int | None = None) -> int:
    """RSS of process ``pid`` (default: this one) plus all its descendants."""
    try:
        import psutil
    except ImportError:  # /proc fallback (Linux)
        return _proc_tree_rss(pid or os.getpid())
    try:
        root = psutil.Process(pid)
        procs = [root] + root.children(recursive=True)
    except psutil.Error:
        return 0
    tot = 0
    for p in procs:
        try:
            tot += p.memory_info().rss
        except psutil.Error:
            pass
    return tot


def _proc_tree_rss(pid: int) -> int:
    kids: dict[int, list[int]] = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
            kids.setdefault(ppid, []).append(int(d))
        except (OSError, ValueError, IndexError):
            pass
    tot, stack = 0, [pid]
    while stack:
        p = stack.pop()
        stack += kids.get(p, [])
        try:
            with open(f"/proc/{p}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        tot += int(line.split()[1]) * 1024
                        break
        except OSError:
            pass
    return tot


class MemGuard:
    """Pause/resume decision from the summed RSS of the pool.

    ``check()`` samples ``rss_fn`` at most every ``interval`` seconds (``force`` samples
    now) and returns True while dispatch should be paused: it pauses when the RSS is
    above ``budget`` bytes and resumes once it is below ``resume_frac * budget``.
    A budget of 0/None disables pausing (the peak is still tracked)."""

    def __init__(self, budget: float | None, rss_fn=tree_rss_bytes, interval: float = 3.0,
                 resume_frac: float = 0.8, clock=time.monotonic, log=print):
        self.budget = budget or 0
        self.resume = resume_frac * self.budget
        self.rss_fn, self.interval, self.clock, self.log = rss_fn, interval, clock, log
        self.paused = False
        self.rss = 0
        self.peak = 0
        self.pauses = 0
        self.paused_s = 0.0
        self._last = None
        self._paused_at = None

    def check(self, force: bool = False) -> bool:
        now = self.clock()
        if not force and self._last is not None and now - self._last < self.interval:
            return self.paused
        self._last = now
        self.rss = int(self.rss_fn())
        self.peak = max(self.peak, self.rss)
        if not self.budget:
            return False
        if not self.paused and self.rss > self.budget:
            self.paused, self._paused_at = True, now
            self.pauses += 1
            self.log(f"memguard: RSS {self.rss / 2**30:.1f} GB > budget "
                     f"{self.budget / 2**30:.1f} GB; pausing dispatch")
        elif self.paused and self.rss < self.resume:
            self.paused = False
            self.paused_s += now - self._paused_at
            self.log(f"memguard: RSS {self.rss / 2**30:.1f} GB < {self.resume / 2**30:.1f} GB; "
                     f"resuming dispatch")
        return self.paused


def guarded_map(make_pool, fn, jobs, cap: int, guard: MemGuard, on_result, log=print,
                on_error=None, poll: float = 1.0, weight=None, max_weight: float = 0,
                stop=None) -> int:
    """Run ``fn(job)`` for every job on pools from ``make_pool()`` with at most ``cap``
    jobs in flight; no new job is submitted while ``guard.check()`` says paused. If
    paused with nothing in flight, the pool is closed and a fresh one started (its idle
    workers may hold leaked memory). ``on_error(job, exc)`` maps a task exception to a
    result. ``weight(job)`` / ``max_weight`` > 0: a job is only submitted while the
    summed weight in flight stays <= max_weight (or nothing is in flight); bounds the
    frames held by the workers when clips are very long. ``stop()`` -> True: submit
    nothing more, finish the jobs in flight and return. Returns the pool restarts."""
    import queue
    done: queue.Queue = queue.Queue()
    it = iter(jobs)
    exhausted, inflight, restarts = False, 0, 0
    wsum, held = 0.0, None
    pool = make_pool()

    def callbacks(j, w):
        return (lambda r: done.put((w, r)),
                lambda e: done.put((w, on_error(j, e) if on_error else e)))
    try:
        while True:
            paused = guard.check()
            if not exhausted and stop is not None and stop():
                exhausted, held = True, None
            while not paused and not exhausted and inflight < cap:
                if held is None:
                    try:
                        held = next(it)
                    except StopIteration:
                        exhausted = True
                        break
                w = float(weight(held)) if weight else 0.0
                if max_weight and inflight and wsum + w > max_weight:
                    break
                ok, err = callbacks(held, w)
                pool.apply_async(fn, (held,), callback=ok, error_callback=err)
                held = None
                inflight += 1
                wsum += w
            if exhausted and inflight == 0:
                break
            if paused and inflight == 0:
                if guard.check(force=True):
                    log(f"memguard: paused with no clip in flight (RSS "
                        f"{guard.rss / 2**30:.1f} GB); restarting the pool")
                    pool.close()
                    pool.join()
                    pool = make_pool()
                    restarts += 1
                    guard.check(force=True)
                continue
            try:
                w, res = done.get(timeout=poll)
            except queue.Empty:
                continue
            inflight -= 1
            wsum -= w
            on_result(res)
    finally:
        pool.close()
        pool.join()
    return restarts


# ---------------------------------------------------------------- shards
def list_shards(out_dir: str) -> list[str]:
    return sorted(p for p in glob.glob(os.path.join(out_dir, "shard_*.h5")) if SHARD_RE.search(p))


def shard_index(out_dir: str) -> dict[str, str]:
    """clip_id -> shard path for every packed clip."""
    import h5py
    idx = {}
    for p in list_shards(out_dir):
        with h5py.File(p, "r") as f:
            for k in f.keys():
                idx[k] = p
    return idx


def recover(out_dir: str, log=print) -> tuple[dict, set, int]:
    """Clean up after a crash. Returns (packed index, staged clip ids, next shard no)."""
    import h5py
    for p in glob.glob(os.path.join(out_dir, "*.tmp")) + \
            glob.glob(os.path.join(out_dir, "_staging", "*.tmp")):
        os.remove(p)
    nxt = 0
    for p in glob.glob(os.path.join(out_dir, "shard_*")):
        m = re.search(r"shard_(\d{5})", p)
        if m:
            nxt = max(nxt, int(m.group(1)) + 1)
    idx = {}
    for p in list_shards(out_dir):
        try:
            with h5py.File(p, "r") as f:
                for k in f.keys():
                    idx[k] = p
        except Exception as e:  # noqa: BLE001 - shards are renamed atomically, so unexpected
            log(f"unreadable shard {p}: {e}; moved to .corrupt, its clips will be redone")
            os.replace(p, p + ".corrupt")
    staged = set()
    for p in glob.glob(os.path.join(out_dir, "_staging", "*.h5")):
        cid = os.path.basename(p)[:-3]
        if cid in idx:
            os.remove(p)
            continue
        try:
            with h5py.File(p, "r") as f:
                f["kp"].shape  # noqa: B018
            staged.add(cid)
        except Exception:  # noqa: BLE001
            os.remove(p)
    return idx, staged, nxt


def pack(out_dir: str, clip_ids: list[str], shard_no: int) -> str:
    """Pack staged clips into shard_<no>.h5. With REEXTRACT_PACK_TMP (a local dir) each
    staged file is first copied there and the shard is built there, then copied to out_dir
    in one sequential pass: HDF5's small reads and writes run ~40x slower over NFS."""
    import h5py
    dst = os.path.join(out_dir, f"shard_{shard_no:05d}.h5")
    if os.path.exists(dst):
        raise FileExistsError(dst)
    tmp = dst + ".tmp"
    local = os.environ.get("REEXTRACT_PACK_TMP")
    if local:
        local = os.path.join(local, f"{os.getpid()}_{os.path.basename(out_dir.rstrip('/'))}")
        os.makedirs(local, exist_ok=True)
    build = os.path.join(local, os.path.basename(tmp)) if local else tmp
    try:
        with h5py.File(build, "w") as fo:
            fo.attrs["shard"] = shard_no
            for cid in clip_ids:
                src = os.path.join(out_dir, "_staging", cid + ".h5")
                if local:
                    src = shutil.copyfile(src, os.path.join(local, cid + ".h5"))
                with h5py.File(src, "r") as fi:
                    g = fo.create_group(cid)
                    for k in fi.keys():
                        fi.copy(fi[k], g, name=k)
                    for k, v in fi.attrs.items():
                        g.attrs[k] = v
                if local:
                    os.remove(src)
        if local:
            shutil.copyfile(build, tmp)
        with open(tmp, "r+b") as f:
            os.fsync(f.fileno())
    finally:
        if local:
            shutil.rmtree(local, ignore_errors=True)
    os.replace(tmp, dst)
    for cid in clip_ids:
        os.remove(os.path.join(out_dir, "_staging", cid + ".h5"))
    return dst


def copy_dup(row: dict, src_out_dirs: list[str], staged: str, frozen: dict | None,
             indices: dict) -> bool:
    """Copy the arrays of row['dup_of'] (packed or staged) into a staged file."""
    import h5py
    tgt = row["dup_of"]
    for od in src_out_dirs:
        if od not in indices:
            indices[od] = shard_index(od) if os.path.isdir(od) else {}
        p = indices[od].get(tgt)
        sp = os.path.join(od, "_staging", tgt + ".h5")
        if p is None and os.path.exists(sp):
            p, grp = sp, "/"
        elif p is not None:
            grp = tgt
        else:
            continue
        with h5py.File(p, "r") as f:
            g = f[grp]
            kp, world, flags = g["kp"][()], g["world"][()], g["flags"][()]
            a = dict(g.attrs)
        if kp.shape[0] != row["frames"]:
            return False
        a = clip_attrs(a, row, frozen)
        a["dup_of"] = tgt
        write_staged(staged, kp, world, flags, a)
        return True
    return False


# ---------------------------------------------------------------- driver
def _append_failures(path: str, fails: list[tuple]) -> None:
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "clip_id", "status", "frames", "message"])
        for cid, status, T, msg in fails:
            w.writerow([time.strftime("%F %T"), cid, status, T, msg.strip().splitlines()[0] if msg else ""])


class UnitRun:
    """State of one unit (dataset, or dataset + batch) being extracted: the clips still
    to do (primaries longest first, duplicates), staged clips waiting to be packed,
    statistics and failures. Shared by run() (one unit on its own pool) and the
    streaming driver (stream_extract: many units on one long-lived pool). Construction
    recovers the output dir after a crash (see recover())."""

    def __init__(self, ds: str, manifest: str | None = None, out_dir: str | None = None,
                 shard_clips: int = 1000, limit: int | None = None, frozen: dict | None = None,
                 extractor: str | None = None, log=print, progress_every: float = 60.0,
                 guard: "MemGuard | None" = None):
        self.ds, self.frozen, self.extractor = ds, frozen, extractor
        self.manifest = manifest or wpath("manifest", f"{ds}.csv")
        self.out_dir = out_dir or wpath("out", ds)
        self.shard_clips, self.progress_every, self.log = shard_clips, progress_every, log
        self.guard = guard or MemGuard(None, rss_fn=lambda: 0)
        os.makedirs(os.path.join(self.out_dir, "_staging"), exist_ok=True)
        rows = read_manifest(self.manifest)
        if limit:
            rows = rows[:limit]
        self.excl = load_excludes(self.manifest)
        if self.excl:
            n0 = len(rows)
            rows = [r for r in rows if r["clip_id"] not in self.excl]
            log(f"[{ds}] {n0 - len(rows)} clips excluded "
                f"({os.path.basename(self.manifest)[:-4]}_exclude.csv)")
        self.rows = rows
        idx, staged, self.shard_no = recover(self.out_dir, log)
        done = set(idx) | staged
        todo = [r for r in rows if r["clip_id"] not in done]
        defer = load_defer()
        if defer:
            n0 = len(todo)
            todo = [r for r in todo if r["clip_id"] not in defer]
            log(f"[{ds}] {n0 - len(todo)} clips deferred to another host (REEXTRACT_DEFER_LIST); "
                f"their staged files are packed by a later run")
        self.primary = sorted([r for r in todo if not r["dup_of"]], key=lambda r: -r["frames"])
        self.dups = [r for r in todo if r["dup_of"]]
        self.total_frames = sum(r["frames"] for r in rows)
        self.done_frames0 = sum(r["frames"] for r in rows if r["clip_id"] in done)
        log(f"[{ds}] {len(rows)} clips ({self.total_frames} frames): {len(idx)} packed, "
            f"{len(staged)} staged, {len(self.primary)} to extract, {len(self.dups)} "
            f"duplicates to copy")
        self.pending = sorted(staged)
        self.t0 = time.time()
        self.stats = dict(clips=0, frames=0, clip_s=0.0)
        self.fails: list[tuple] = []
        self._last = self.t0
        self.dups_copied = 0

    def staged_path(self, r: dict) -> str:
        return os.path.join(self.out_dir, "_staging", r["clip_id"] + ".h5")

    def job(self, r: dict) -> tuple:
        return r, self.staged_path(r)

    def flush(self, force: bool = False) -> None:
        while self.pending and (force or len(self.pending) >= self.shard_clips):
            chunk, self.pending = self.pending[:self.shard_clips], self.pending[self.shard_clips:]
            p = pack(self.out_dir, chunk, self.shard_no)
            self.shard_no += 1
            self.log(f"[{self.ds}] packed {len(chunk)} clips -> {os.path.basename(p)}")

    def on_result(self, res) -> None:
        cid, status, T, el, msg = res
        ds, stats, guard = self.ds, self.stats, self.guard
        if status == "ok":
            self.pending.append(cid)
            stats["clips"] += 1
            stats["frames"] += T
            stats["clip_s"] += el
        else:
            self.fails.append((cid, status, T, msg))
            self.log(f"[{ds}] FAIL {cid} {status} {msg.strip().splitlines()[0] if msg else ''}")
        self.flush()
        now = time.time()
        if now - self._last >= self.progress_every:
            self._last = now
            wall = now - self.t0
            fps = stats["frames"] / max(wall, 1e-9)
            remain = self.total_frames - self.done_frames0 - stats["frames"]
            eta_h = remain / fps / 3600 if fps > 0 else float("nan")
            self.log(f"[{ds}] progress {stats['clips']}/{len(self.primary)} clips, "
                     f"{stats['frames']} frames, {fps:.1f} fps wall, "
                     f"{stats['frames'] / max(stats['clip_s'], 1e-9):.2f} fps/proc, "
                     f"fails {len(self.fails)}, ETA {eta_h:.2f} h, RSS {guard.rss / 2**30:.1f} GB "
                     f"(peak {guard.peak / 2**30:.1f}, pauses {guard.pauses})")
            with open(os.path.join(self.out_dir, "_progress.json"), "w") as f:
                json.dump(dict(stats, wall_s=wall, fps=fps, remaining_frames=remain,
                               eta_h=eta_h, fails=len(self.fails),
                               rss_gb=round(guard.rss / 2**30, 2),
                               peak_rss_gb=round(guard.peak / 2**30, 2), mem_pauses=guard.pauses,
                               time=time.strftime("%F %T")), f)

    def copy_dups(self) -> list[dict]:
        """Copy each duplicate row from its already extracted twin (same or earlier
        dataset); returns the rows that had no twin (to be extracted)."""
        indices: dict = {}
        redo = []
        for r in self.dups:
            twin_ds = r["dup_of"].split("__")[0]
            od = self.out_dir
            srcs = [od] + ([os.path.join(os.path.dirname(od), twin_ds)] if twin_ds != self.ds else [])
            if copy_dup(r, srcs, self.staged_path(r), self.frozen, indices):
                self.on_result((r["clip_id"], "ok", r["frames"], 0.0, ""))
            else:
                redo.append(r)
        self.dups_copied = len(self.dups) - len(redo)
        if redo:
            self.log(f"[{self.ds}] {len(redo)} duplicates had no extracted twin; extracting them")
        return redo

    def finish(self, workers: int, mem_budget_gb=None, max_tasks_per_child=None,
               pool_restarts: int = 0, sliced: bool = False, max_inflight_frames: int = 0,
               **extra) -> dict:
        """Pack what is staged, append failures.csv, write and log the run summary."""
        self.flush(force=True)
        wall = time.time() - self.t0
        stats, guard = self.stats, self.guard
        if self.fails:
            _append_failures(os.path.join(self.out_dir, "failures.csv"), self.fails)
        summary = dict(dataset=self.ds, extractor=self.extractor, workers=workers,
                       wall_s=round(wall, 1), clips=stats["clips"], frames=stats["frames"],
                       sum_clip_elapsed_s=round(stats["clip_s"], 1),
                       fps_wall=round(stats["frames"] / max(wall, 1e-9), 2),
                       fps_per_proc=round(stats["frames"] / max(stats["clip_s"], 1e-9), 3),
                       failures=len(self.fails), dups_copied=self.dups_copied,
                       mem_budget_gb=mem_budget_gb, max_tasks_per_child=max_tasks_per_child,
                       peak_rss_gb=round(guard.peak / 2**30, 2), mem_pauses=guard.pauses,
                       mem_paused_s=round(guard.paused_s, 1), pool_restarts=pool_restarts,
                       excluded=len(self.excl), sliced=sliced,
                       max_inflight_frames=max_inflight_frames, **extra,
                       time=time.strftime("%F %T"))
        with open(os.path.join(self.out_dir, f"_run_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
            json.dump(summary, f, indent=1)
        self.log(f"[{self.ds}] done: {json.dumps(summary)}")
        return summary


def run(ds: str, workers: int = 1, extractor: str | None = None, frozen: dict | None = None,
        manifest: str | None = None, out_dir: str | None = None, shard_clips: int = 1000,
        limit: int | None = None, factory=None, log=print, progress_every: float = 60.0,
        mem_budget_gb: float | None = None, max_tasks_per_child: int = 4,
        mem_check_s: float = 3.0, max_wall_s: float = 0, max_inflight_frames: int = 0) -> dict:
    """Extract every manifest row of `ds` not yet done. ``factory`` (tests) builds an
    in-process extractor instead of tools.extract.extractors.make(extractor).

    Rows listed in manifest/<stem>_exclude.csv are skipped. ``max_wall_s`` > 0: stop
    dispatching after that many seconds, finish the clips in flight, pack and return
    with ``sliced=True`` (the next run resumes). ``max_inflight_frames`` > 0: bound the
    summed manifest frames of the clips in flight (long YouTube videos)."""
    guard = MemGuard((mem_budget_gb or 0) * 2**30, interval=mem_check_s,
                     log=lambda m: log(f"[{ds}] {m}"))
    u = UnitRun(ds, manifest, out_dir, shard_clips, limit, frozen, extractor, log,
                progress_every, guard)
    pool_restarts = [0]
    sliced = [False]

    def stop():
        if max_wall_s and not sliced[0] and time.time() - u.t0 > max_wall_s:
            log(f"[{ds}] time slice of {max_wall_s:.0f}s used; finishing the clips in flight")
            sliced[0] = True
        return sliced[0]

    def extract_rows(rs):
        if not rs:
            return
        jobs = [u.job(r) for r in rs]
        if factory is not None or workers <= 1:
            global _EX
            if factory is not None:
                _EX = factory()
                _CTX["frozen"] = frozen
            else:
                _init(extractor, frozen)
            for j in jobs:
                if stop():
                    break
                u.on_result(_job(j))
        else:
            ctx = mp.get_context("spawn")

            def make_pool():
                return ctx.Pool(workers, initializer=_init, initargs=(extractor, frozen),
                                maxtasksperchild=max_tasks_per_child or None)

            def on_error(j, e):
                return j[0]["clip_id"], "error", 0, 0.0, f"{type(e).__name__}: {e}"
            pool_restarts[0] += guarded_map(make_pool, _job, jobs, workers, guard, u.on_result,
                                            log=lambda m: log(f"[{ds}] {m}"),
                                            on_error=on_error, stop=stop,
                                            weight=lambda j: j[0]["frames"],
                                            max_weight=max_inflight_frames)

    extract_rows(u.primary)
    if sliced[0]:
        u.dups = []  # copied in the run that finishes the primaries
    # duplicates: copy from the already extracted twin (same or earlier dataset)
    extract_rows(u.copy_dups())
    return u.finish(workers, mem_budget_gb=mem_budget_gb, max_tasks_per_child=max_tasks_per_child,
                    pool_restarts=pool_restarts[0], sliced=sliced[0],
                    max_inflight_frames=max_inflight_frames)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(DATASETS))
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--shard-clips", type=int, default=1000)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--mem-budget-gb", type=float, default=None,
                    help="pause dispatch while the summed RSS of this process tree exceeds "
                         "this (resume below 80%%)")
    ap.add_argument("--max-tasks-per-child", type=int, default=4,
                    help="recycle each worker after this many clips (mediapipe leaks RSS)")
    ap.add_argument("--mem-check-s", type=float, default=3.0)
    ap.add_argument("--batch", help="queue datasets: manifest/<ds>_<batch>.csv -> out/<ds>/<batch>/")
    ap.add_argument("--max-wall-s", type=float, default=0,
                    help="stop dispatching after this many seconds (exit 3; a rerun resumes)")
    ap.add_argument("--max-inflight-frames", type=int, default=0,
                    help="bound the summed frames of the clips in flight (0 = off)")
    ap.add_argument("--config", default=None, help="frozen config json (default configs/extract/mp_l_v4v0.json)")
    a = ap.parse_args(argv)
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")  # read by glibc at worker start
    from tools.extract.freeze_config import CONFIG_PATH, check_frozen, load_frozen
    frozen = load_frozen(a.config or CONFIG_PATH)
    errs = check_frozen(frozen)
    if errs:
        raise SystemExit("refusing to run: " + "; ".join(errs))
    up = unit_paths(a.dataset, a.batch)
    s = run(a.dataset, a.workers, frozen["name"], frozen, manifest=up["manifest"],
            out_dir=up["out_dir"], shard_clips=a.shard_clips,
            limit=a.limit, log=lambda m: print(time.strftime("%F %T"), m, flush=True),
            mem_budget_gb=a.mem_budget_gb, max_tasks_per_child=a.max_tasks_per_child,
            mem_check_s=a.mem_check_s, max_wall_s=a.max_wall_s,
            max_inflight_frames=a.max_inflight_frames)
    # 0 done, 1 some clips failed, 3 time slice used (a rerun continues and retries failures)
    raise SystemExit(3 if s["sliced"] else 1 if s["failures"] else 0)


if __name__ == "__main__":
    main()
