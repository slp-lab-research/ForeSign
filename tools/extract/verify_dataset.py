"""Verify one extracted dataset (spec section 4, step 3) and write reports/<ds>_verify.json.

  python -m tools.extract.verify_dataset autsl --workers 16

Checks (any failure -> status "fail", exit code 1):
  * every manifest row not listed in manifest/<stem>_exclude.csv has exactly one output group (no missing, duplicated, or extra clips;
    nothing left in _staging);
  * kp (T,553,4), world (T,33,4), flags (T,3) present; T == manifest decoded frame count;
  * no NaN / inf in kp or world;
  * config_hash attr == frozen config (when a frozen config is given);
  * detection rates (frame-weighted) within --tol-fail points of the bake-off reference
    for the dataset (configs/extract/bakeoff_reference.json); > --tol-warn is a warning.

Detection rates: pose (pose found), face (any face landmark), left/right hand (any hand),
hand_given_wrist (hand present where its MP pose wrist has visibility >= 0.5 and lies
inside the image; this is closest to the bake-off "hand det %"), crop_share (share of
detected hands that came from the crop step).

Reference: ``--make-reference <bakeoff out dir> --bakeoff-manifest <csv>`` computes the
same rates on the bake-off MP-L-V4V0 per-clip outputs (40 clips per dataset).
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

from tools.extract.build_manifest import read_manifest
from tools.extract.datasets import BAKEOFF_REF, DATASETS, load_excludes, unit_paths, wpath
from tools.extract.extract_dataset import list_shards

CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF_PATH = os.path.join(CODE_DIR, "configs", "extract", "bakeoff_reference.json")
RATE_KEYS = ["pose", "face", "lhand", "rhand", "hand_given_wrist", "crop_share"]
CHECKED_RATES = ["pose", "face", "hand_given_wrist"]


def rate_counts(kp: np.ndarray, flags: np.ndarray) -> dict:
    """Additive counts for the detection rates of one clip."""
    kp = kp.astype(np.float32)
    T = kp.shape[0]
    c = dict(frames=T, pose=int(flags[:, 0].sum()) if T else 0,
             face=int((kp[:, 33:511, 3] > 0).any(1).sum()) if T else 0,
             lhand=int((flags[:, 1] > 0).sum()), rhand=int((flags[:, 2] > 0).sum()),
             hands=int((flags[:, 1:] > 0).sum()), crop=int((flags[:, 1:] == 2).sum()),
             wrist=0, wrist_hand=0)
    for iw, col in ((15, 1), (16, 2)):
        ok = (kp[:, iw, 3] >= 0.5) & (kp[:, iw, 0] >= 0) & (kp[:, iw, 0] <= 1) & \
             (kp[:, iw, 1] >= 0) & (kp[:, iw, 1] <= 1) & (flags[:, 0] > 0)
        c["wrist"] += int(ok.sum())
        c["wrist_hand"] += int((ok & (flags[:, col] > 0)).sum())
    return c


def rates_from_counts(c: dict) -> dict:
    f = max(c["frames"], 1)
    return dict(pose=100 * c["pose"] / f, face=100 * c["face"] / f, lhand=100 * c["lhand"] / f,
                rhand=100 * c["rhand"] / f,
                hand_given_wrist=100 * c["wrist_hand"] / max(c["wrist"], 1),
                crop_share=100 * c["crop"] / max(c["hands"], 1))


def _add(a: dict, b: dict) -> dict:
    for k, v in b.items():
        a[k] = a.get(k, 0) + v
    return a


def check_shard(args) -> dict:
    """Per-clip checks for one shard; returns {clip_id: info}."""
    path, cfg_hash = args
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        for cid in f.keys():
            g = f[cid]
            info = dict(shard=os.path.basename(path), errors=[])
            try:
                kp, world, flags = g["kp"][()], g["world"][()], g["flags"][()]
            except KeyError as e:
                info["errors"].append(f"missing dataset {e}")
                out[cid] = info
                continue
            T = kp.shape[0]
            info["T"] = T
            if kp.shape[1:] != (553, 4):
                info["errors"].append(f"kp shape {kp.shape}")
            if world.shape != (T, 33, 4):
                info["errors"].append(f"world shape {world.shape}")
            if flags.shape != (T, 3):
                info["errors"].append(f"flags shape {flags.shape}")
            if not (np.isfinite(kp).all() and np.isfinite(world).all()):
                info["errors"].append("nan/inf")
            if cfg_hash and g.attrs.get("config_hash") != cfg_hash:
                info["errors"].append(f"config_hash {g.attrs.get('config_hash')}")
            info["fps"] = float(g.attrs.get("fps", 0))
            info["width"], info["height"] = int(g.attrs.get("width", 0)), int(g.attrs.get("height", 0))
            if not info["errors"]:
                info["counts"] = rate_counts(kp, flags)
            out[cid] = info
    return out


def verify(ds: str, manifest: str | None = None, out_dir: str | None = None,
           report: str | None = None, cfg_hash: str | None = None, ref: dict | None = None,
           tol_fail: float = 15.0, tol_warn: float = 5.0, workers: int = 1,
           max_list: int = 200) -> dict:
    manifest = manifest or wpath("manifest", f"{ds}.csv")
    out_dir = out_dir or wpath("out", ds)
    report = report or wpath("reports", f"{ds}_verify.json")
    all_rows = read_manifest(manifest)
    excl = load_excludes(manifest)
    rows = [r for r in all_rows if r["clip_id"] not in excl]
    excluded = [dict(clip_id=r["clip_id"], frames=r["frames"], reason=excl[r["clip_id"]])
                for r in all_rows if r["clip_id"] in excl]
    mrows = {r["clip_id"]: r for r in rows}
    shards = list_shards(out_dir)
    jobs = [(p, cfg_hash) for p in shards]
    if workers > 1 and len(jobs) > 1:
        with mp.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            parts = pool.map(check_shard, jobs)
    else:
        parts = [check_shard(j) for j in jobs]
    seen: dict[str, dict] = {}
    dup_groups = []
    for part in parts:
        for cid, info in part.items():
            if cid in seen:
                dup_groups.append(cid)
            seen[cid] = info
    staging = os.path.join(out_dir, "_staging")
    leftover = sorted(os.listdir(staging)) if os.path.isdir(staging) else []
    missing = sorted(set(mrows) - set(seen))
    extra = sorted(set(seen) - set(mrows) - set(excl))
    bad, mismatch, warnings = [], [], []
    counts: dict = {}
    for cid, info in seen.items():
        if cid not in mrows:
            continue
        if info["errors"]:
            bad.append((cid, info["errors"]))
            continue
        r = mrows[cid]
        if info["T"] != r["frames"]:
            mismatch.append((cid, info["T"], r["frames"]))
            continue
        _add(counts, info["counts"])
    rates = rates_from_counts(counts) if counts else {}
    ref_rates = (ref or {}).get(BAKEOFF_REF.get(ds, ds))
    rate_check = {}
    rate_fail = []
    if ref_rates and rates:
        for k in CHECKED_RATES:
            d = rates[k] - ref_rates[k]
            rate_check[k] = dict(value=round(rates[k], 2), ref=round(ref_rates[k], 2),
                                 diff=round(d, 2))
            if abs(d) > tol_fail:
                rate_fail.append(k)
            elif abs(d) > tol_warn:
                warnings.append(f"{k} {rates[k]:.1f} vs bake-off {ref_rates[k]:.1f}")
    elif not ref_rates:
        warnings.append("no bake-off reference for this dataset")
    merr = os.path.join(os.path.dirname(manifest), f"{ds}_manifest_errors.csv")
    n_merr = 0
    if os.path.exists(merr):
        with open(merr, encoding="utf-8") as f:
            n_merr = max(sum(1 for _ in f) - 1, 0)
        if n_merr:
            warnings.append(f"{n_merr} sources could not be read when building the manifest "
                            f"(not in manifest; see {os.path.basename(merr)})")
    if excluded:
        warnings.append(f"{len(excluded)} manifest clips excluded "
                        f"({os.path.basename(manifest)[:-4]}_exclude.csv; listed under 'excluded')")
    zero = [r["clip_id"] for r in rows if r["frames"] == 0]
    if zero:
        warnings.append(f"{len(zero)} manifest clips decode to 0 frames")
    checks = dict(missing=len(missing), extra=len(extra), duplicated=len(dup_groups),
                  staging_leftover=len(leftover), bad=len(bad), frame_mismatch=len(mismatch),
                  rate_out_of_range=rate_fail)
    ok = not (missing or extra or dup_groups or leftover or bad or mismatch or rate_fail) and rows
    rep = dict(dataset=ds, status="pass" if ok else "fail", time=time.strftime("%F %T"),
               manifest_clips=len(rows), manifest_frames=sum(r["frames"] for r in rows),
               manifest_rows=len(all_rows), excluded_clips=len(excluded),
               excluded_frames=sum(e["frames"] for e in excluded), excluded=excluded,
               output_clips=len(seen), output_frames=counts.get("frames", 0),
               shards=len(shards), config_hash=cfg_hash, checks=checks,
               rates={k: round(v, 2) for k, v in rates.items()}, rate_check=rate_check,
               tol_fail=tol_fail, tol_warn=tol_warn, warnings=warnings,
               missing=missing[:max_list], extra=extra[:max_list],
               duplicated=dup_groups[:max_list], staging_leftover=leftover[:max_list],
               bad=bad[:max_list], frame_mismatch=mismatch[:max_list])
    os.makedirs(os.path.dirname(os.path.abspath(report)), exist_ok=True)
    with open(report, "w") as f:
        json.dump(rep, f, indent=1)
    return rep


def make_reference(bake_out: str, bake_manifest: str, path: str = REF_PATH) -> dict:
    """Detection rates of the bake-off MP-L-V4V0 outputs per dataset (same definitions)."""
    import h5py
    with open(bake_manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    per: dict = {}
    for r in rows:
        p = os.path.join(bake_out, r["clip_id"] + ".h5")
        if not os.path.exists(p):
            continue
        with h5py.File(p, "r") as f:
            kp, valid = f["kp"][()], f["valid"][()]
            src = f["hand_src"][()] if "hand_src" in f else np.zeros((kp.shape[0], 2), np.uint8)
        flags = np.concatenate([valid.reshape(-1, 1), src], 1).astype(np.uint8)
        _add(per.setdefault(r["dataset"], {}), rate_counts(kp, flags))
    ref = {d: {k: round(v, 3) for k, v in rates_from_counts(c).items()} | {"frames": c["frames"]}
           for d, c in per.items()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(ref, f, indent=1, sort_keys=True)
    return ref


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", choices=sorted(DATASETS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tol-fail", type=float, default=15.0)
    ap.add_argument("--tol-warn", type=float, default=5.0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--make-reference", metavar="BAKEOFF_OUT_DIR")
    ap.add_argument("--batch", help="queue datasets: verify manifest/<ds>_<batch>.csv")
    ap.add_argument("--bakeoff-manifest")
    a = ap.parse_args(argv)
    if a.make_reference:
        print(json.dumps(make_reference(a.make_reference, a.bakeoff_manifest), indent=1))
        return
    from tools.extract.freeze_config import CONFIG_PATH, load_frozen
    frozen = load_frozen(a.config or CONFIG_PATH)
    ref = json.load(open(REF_PATH)) if os.path.exists(REF_PATH) else None
    up = unit_paths(a.dataset, a.batch)
    rep = verify(a.dataset, manifest=up["manifest"], out_dir=up["out_dir"], report=up["report"],
                 cfg_hash=frozen["config_hash"], ref=ref, tol_fail=a.tol_fail,
                 tol_warn=a.tol_warn, workers=a.workers)
    print(json.dumps({k: rep[k] for k in ("dataset", "status", "manifest_clips", "output_clips",
                                          "output_frames", "excluded_clips", "checks", "rates", "rate_check",
                                          "warnings")}), flush=True)
    sys.exit(0 if rep["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
