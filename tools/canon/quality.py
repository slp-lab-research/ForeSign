"""Normalisation quality check + calibration probes (additional evidence; the pre-registered
gate decision is not touched).

  python -m tools.canon.quality run --selection ~/canon/fp/selection.json \
      --out-dir ~/canon/quality --workers 16

For the 300 gate clips of each ASL dataset plus 300 clips of phoenix14t / autsl / isign:
  * every clip is canonicalised from the raw shards with canon-v1, -v2c and -v3 (same code as
    build_canon) and also mapped to raw image coordinates (isotropic pixels, z = 0);
  * per-clip geometry metrics (shoulder width, axis angles, limb / hand / face ratios,
    far-joint rate, jitter), aggregated per dataset as median / IQR / % Tukey outliers;
  * figures for 3 clips per dataset (front x-y and side z-y views, several frames overlaid)
    and one GIF per dataset (raw vs canon-v3).
Calibration probes (same P1 / P2 window features and 5-fold logistic regression as the gate):
  floor  - one dataset's clips split randomly into two pseudo-datasets (chance 0.5);
  signer - 3 groups of disjoint signers, 300 clips each (ASL Citizen participants, WLASL
           signer_id, AUTSL signerN), i.e. what signer identity alone gives (chance 1/3).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from multiprocessing import Pool

import numpy as np

from tools.canon import fingerprint as F
from tools.canon.canon import (FACE26, FPS_OUT, canon_clip, raw77_2d, resample)

VERS = ["raw", "canon-v1", "canon-v2c", "canon-v3", "canon-v3.1", "canon-v4"]
EXTRA = ["phoenix14t", "autsl", "isign"]
RAW_DIRS = dict(F.RAW_DIRS, phoenix14t=["~/reextract/out/phoenix14t"],
                autsl=["~/reextract/out/autsl"], isign=["~/reextract/out/isign"])
MANIFEST = {"phoenix14t": "~/reextract/manifest/phoenix14t.csv",
            "autsl": "~/reextract/manifest/autsl.csv",
            "isign": "~/reextract/manifest/isign.csv",
            "asl_citizen": "~/reextract/manifest/asl_citizen.csv",
            "wlasl": "~/reextract/manifest/wlasl.csv"}
# Official split files of the source releases (signer probe); see tools/extract/datasets.py.
_SRC = os.environ.get("REEXTRACT_SRC_ROOT", "videos")
ASLC_SPLITS = os.path.join(_SRC, "asl_citizen", "ASL_Citizen", "splits")
WLASL_JSON = os.path.join(_SRC, "wlasl", "WLASL_v0.3.json")
J_IOD = (51 + FACE26.index(33), 51 + FACE26.index(263))

# 77-joint bones (RESONA-77 order)
HAND = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10),
        (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19),
        (19, 20), (5, 9), (9, 13), (13, 17)]
BONES_L = [(1, 3), (3, 5)] + [(9 + a, 9 + b) for a, b in HAND]
BONES_R = [(2, 4), (4, 6)] + [(30 + a, 30 + b) for a, b in HAND]
BONES_C = [(1, 2), (1, 7), (2, 8), (7, 8)] + [(51 + i, 51 + (i + 1) % 8) for i in range(8)]


# ------------------------------------------------------------------ metrics (pure)
def _nan_median(v):
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else float("nan")


def _angle(a, b):
    """Angle in degrees between vectors a (n,3) and b (3,)."""
    na = np.linalg.norm(a, axis=-1)
    cos = (a @ b) / np.maximum(na * np.linalg.norm(b), 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def clip_metrics(x: np.ndarray, m: np.ndarray) -> dict:
    """Geometry of one clip (T,77,3), mask (T,77) in its own units. Ratios are per frame
    (divided by that frame's shoulder width) and the clip value is the median."""
    v = m > 0
    body = v[:, 1] & v[:, 2]
    out = dict(frames=int(len(x)), pose_rate=float(body.mean()) if len(x) else 0.0)
    if not body.any():
        return out
    sw = np.linalg.norm(x[:, 1] - x[:, 2], axis=-1)
    sh = x[:, 1] - x[:, 2]
    mid = 0.5 * (x[:, 1] + x[:, 2])
    out["shoulder_width"] = _nan_median(sw[body])
    out["shoulder_axis_deg"] = _nan_median(_angle(sh[body], np.array([1.0, 0, 0])))
    up_ok = body & v[:, 0]
    out["up_axis_deg"] = _nan_median(_angle(x[up_ok, 0] - mid[up_ok], np.array([0, -1.0, 0])))
    ratios = []
    for s_, e_, w_ in ((1, 3, 5), (2, 4, 6)):
        ok = v[:, s_] & v[:, e_] & v[:, w_]
        up = np.linalg.norm(x[ok, e_] - x[ok, s_], axis=-1)
        fo = np.linalg.norm(x[ok, w_] - x[ok, e_], axis=-1)
        ratios.append(fo / np.maximum(up, 1e-9))
    out["forearm_upperarm"] = _nan_median(np.concatenate(ratios))
    hs = []
    for w_, mcp in ((9, 18), (30, 39)):                   # hand wrist -> middle MCP
        ok = body & v[:, w_] & v[:, mcp]
        hs.append(np.linalg.norm(x[ok, mcp] - x[ok, w_], axis=-1) / np.maximum(sw[ok], 1e-9))
    out["hand_size"] = _nan_median(np.concatenate(hs))
    ok = body & v[:, J_IOD[0]] & v[:, J_IOD[1]]
    out["iod"] = _nan_median(np.linalg.norm(x[ok, J_IOD[0]] - x[ok, J_IOD[1]], axis=-1)
                             / np.maximum(sw[ok], 1e-9))
    far = (np.linalg.norm(x, axis=-1) > 3.0) & v
    out["far_frame_pct"] = 100.0 * float(far.any(1)[v.any(1)].mean())
    out["jitter"] = jitter(x, m)
    return out


def jitter(x: np.ndarray, m: np.ndarray, joints=tuple(range(0, 51))) -> float:
    """Mean |acceleration| (per frame^2) over valid frame triples of body + hands."""
    J = list(joints)
    xs, v = x[:, J], m[:, J] > 0
    if len(x) < 3:
        return float("nan")
    ok = v[2:] & v[1:-1] & v[:-2]
    acc = np.linalg.norm(xs[2:] - 2 * xs[1:-1] + xs[:-2], axis=-1)
    return float(acc[ok].mean()) if ok.any() else float("nan")


def summarise(values) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], np.float64)
    if not len(v):
        return dict(n=0)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    out = ((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).mean()
    return dict(n=int(len(v)), median=float(med), q1=float(q1), q3=float(q3), iqr=float(iqr),
                outlier_pct=100.0 * float(out))


def signer_groups(signers: dict, n_groups: int = 3, seed: int = 0) -> dict:
    """clip -> signer  =>  clip -> group, with every signer in exactly one group (signer
    order shuffled, then dealt round-robin by clip count so the groups are balanced)."""
    rng = np.random.default_rng(seed)
    sids = sorted(set(signers.values()), key=str)
    rng.shuffle(sids)
    count = {s: 0 for s in sids}
    for s in signers.values():
        count[s] += 1
    load = [0] * n_groups
    gof = {}
    for s in sorted(sids, key=lambda s: -count[s]):
        g = int(np.argmin(load))
        gof[s] = g
        load[g] += count[s]
    return {c: gof[s] for c, s in signers.items()}


# ------------------------------------------------------------------ per-clip worker
_IDX: dict = {}


def _raw_path(ds: str, clip: str) -> str:
    if ds not in _IDX:
        _IDX[ds] = F.raw_index(ds, RAW_DIRS[ds])
    return _IDX[ds][clip]


def frame_axis_metrics(info: dict, world: np.ndarray, flags: np.ndarray, version: str) -> dict:
    """Spec section 8 acceptance metrics from the clip rotation R (rows x, y=down, z):
    torso_up_deg = median over pose-found frames of the angle between the canonical down
    axis and the anatomical torso direction (shoulder midpoint -> hip midpoint, world);
    up_spread_deg = IQR over frames of the angle between the per-frame vector that defines
    the version's up axis (ears / nose / hips) and the canonical down axis."""
    if "R" not in info:
        return {}
    ok = flags[:, 0] > 0
    w = world[ok, :, :3].astype(np.float64)
    R = np.asarray(info["R"], np.float64)
    mid = 0.5 * (w[:, 11] + w[:, 12])
    torso = 0.5 * (w[:, 23] + w[:, 24]) - mid
    out = dict(torso_up_deg=_nan_median(_angle(torso, R[1])))
    vec = {"canon-v3.1": mid - 0.5 * (w[:, 7] + w[:, 8]), "canon-v3": mid - w[:, 0],
           "canon-v2c": mid - w[:, 0]}.get(version)
    if vec is not None and len(vec):
        a = _angle(vec, R[1])
        out["up_spread_deg"] = float(np.subtract(*np.percentile(a, [75, 25])))
    return out


def versions_of(g) -> dict:
    """h5 group of a raw clip -> {version: (x (T,77,3) at 25 fps, mask)}; raw is the
    isotropic-pixel image skeleton (z = 0) resampled to 25 fps, same 77 joints."""
    a = g.attrs
    W, H, fps = float(a["width"]), float(a["height"]), float(a["fps"])
    kp, world, flags = g["kp"], g["world"][()], g["flags"][()]
    out = {"_axes": {}}
    for v in VERS[1:]:
        x, m, info = canon_clip(kp, world, flags, W, H, fps, version=v)
        out[v] = (x, m)
        out["_axes"][v] = frame_axis_metrics(info, world, flags, v)
    xy, m = raw77_2d(kp, flags)
    xyz = np.zeros(xy.shape[:2] + (3,), np.float32)
    xyz[..., 0], xyz[..., 1] = xy[..., 0] * W, xy[..., 1] * H
    xyz[m == 0] = 0
    out["raw"] = resample(xyz, m, fps)
    out["_rawn"] = resample(xy, m, fps)
    return out


def work(job: dict) -> dict:
    import h5py
    ds, clip = job["ds"], job["clip"]
    with h5py.File(_raw_path(ds, clip), "r") as f:
        vers = versions_of(f[clip])
    rawn = vers.pop("_rawn")
    axes = vers.pop("_axes")
    res = dict(ds=ds, clip=clip, tags=job["tags"], metrics={}, p1={}, p2={})
    for v, (x, m) in vers.items():
        res["metrics"][v] = dict(clip_metrics(x, m), **axes.get(v, {}))
        # window features as in the gate: raw = normalised image (x, y), canon = 3D
        xf, mf = rawn if v == "raw" else (x, m)
        if job.get("start_s") is not None and len(xf) >= F.WIN:
            w, wm = F.take_window(xf, mf, FPS_OUT, job["start_s"])
            res["p1"][v] = F.p1_features(w, wm)
            res["p2"][v] = F.p2_features(w, wm)
    if job.get("viz"):
        res["viz"] = {v: F.take_window(x, m, FPS_OUT, job["start_s"] or 0.0)
                      for v, (x, m) in vers.items()}
    return res


# ------------------------------------------------------------------ clip sets
def _manifest(ds):
    from tools.extract.build_manifest import read_manifest
    return read_manifest(os.path.expanduser(MANIFEST[ds]))


def extra_sets(n: int = 300, seed: int = 0) -> list[dict]:
    jobs = []
    for ds in EXTRA:
        idx = F.raw_index(ds, RAW_DIRS[ds])
        rows = [r for r in _manifest(ds) if r["clip_id"] in idx and r["split"] in ("train", "")
                and r["frames"] / max(r["fps"], 1e-6) >= F.WIN_S]
        rows.sort(key=lambda r: r["clip_id"])
        rng = np.random.default_rng(seed)
        for i in sorted(rng.choice(len(rows), min(n, len(rows)), replace=False)):
            r = rows[i]
            jobs.append(dict(ds=ds, clip=r["clip_id"], tags=["quality"],
                             start_s=F.pick_start(rng, r["frames"], r["fps"])))
    return jobs


def _signers(ds: str) -> dict:
    """clip_id -> signer id for the train split, from the dataset's own metadata."""
    rows = {r["clip_id"]: r for r in _manifest(ds) if r["split"] == "train"}
    out = {}
    if ds == "asl_citizen":
        with open(os.path.join(ASLC_SPLITS, "train.csv"), encoding="utf-8") as f:
            for r in csv.DictReader(f):
                cid = "asl_citizen__" + F.sanitize(os.path.splitext(r["Video file"])[0])
                if cid in rows:
                    out[cid] = r["Participant ID"]
    elif ds == "wlasl":
        with open(WLASL_JSON, encoding="utf-8") as f:
            sid = {i["video_id"]: i["signer_id"] for e in json.load(f) for i in e["instances"]}
        for cid in rows:
            vid = F.resona_key("wlasl", cid)
            if vid in sid:
                out[cid] = sid[vid]
    elif ds == "autsl":
        for cid in rows:
            mm = re.search(r"signer(\d+)_", cid)
            if mm:
                out[cid] = int(mm.group(1))
    return {c: s for c, s in out.items()
            if rows[c]["frames"] / max(rows[c]["fps"], 1e-6) >= F.WIN_S}, rows


def signer_sets(ds: str, n: int = 300, n_groups: int = 3, seed: int = 0):
    sig, rows = _signers(ds)
    grp = signer_groups(sig, n_groups, seed)
    rng = np.random.default_rng(seed + 1)
    jobs, info = [], dict(signers=len(set(sig.values())), clips=len(sig), groups=[])
    for g in range(n_groups):
        cl = sorted(c for c in sig if grp[c] == g)
        pick = sorted(rng.choice(len(cl), min(n, len(cl)), replace=False))
        info["groups"].append(dict(signers=len({sig[c] for c in cl}), clips=len(cl),
                                   picked=len(pick)))
        for i in pick:
            r = rows[cl[i]]
            jobs.append(dict(ds=ds, clip=cl[i], tags=[f"signer{g}"],
                             start_s=F.pick_start(rng, r["frames"], r["fps"])))
    return jobs, info


def all_jobs(selection: dict) -> tuple[list[dict], dict]:
    jobs = []
    for ds, d in selection.items():
        for r in d["clips"]:
            jobs.append(dict(ds=ds, clip=r["clip"], tags=["quality"], start_s=r["start_s"]))
    jobs += extra_sets()
    info = {}
    for ds in ("asl_citizen", "wlasl", "autsl"):
        j, info[ds] = signer_sets(ds)
        jobs += j
    # merge duplicates (a clip may be both a gate clip and a signer-group clip)
    merged = {}
    for j in jobs:
        k = (j["ds"], j["clip"])
        if k in merged:
            merged[k]["tags"] += j["tags"]
        else:
            merged[k] = dict(j)
    jobs = list(merged.values())
    seen = {}
    for j in jobs:                                   # 3 figure clips per dataset
        if "quality" in j["tags"] and seen.get(j["ds"], 0) < 3:
            j["viz"] = True
            seen[j["ds"]] = seen.get(j["ds"], 0) + 1
    return jobs, info


# ------------------------------------------------------------------ run / analyse
METRICS = ["torso_up_deg", "up_spread_deg",
           "shoulder_width", "shoulder_axis_deg", "up_axis_deg", "forearm_upperarm",
           "hand_size", "iod", "far_frame_pct", "jitter"]
DS_ORDER = F.ASL5 + EXTRA


def run(selection: str, out_dir: str, workers: int = 16, log=print) -> None:
    import pickle
    with open(os.path.expanduser(selection)) as f:
        sel = json.load(f)
    out_dir = os.path.expanduser(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    jobs, info = all_jobs(sel)
    log(f"{len(jobs)} clips; signer sets {json.dumps(info)}")
    results = []
    with Pool(workers, maxtasksperchild=25) as pool:
        for i, r in enumerate(pool.imap_unordered(work, jobs, chunksize=1), 1):
            results.append(r)
            if i % 200 == 0:
                log(f"  {i}/{len(jobs)}")
    with open(os.path.join(out_dir, "clips.pkl.tmp"), "wb") as f:
        pickle.dump(dict(results=results, signer_info=info), f)
    os.replace(os.path.join(out_dir, "clips.pkl.tmp"), os.path.join(out_dir, "clips.pkl"))


def calibration(results: list[dict], seed: int = 0) -> dict:
    out = {}
    for ds in ("asl_citizen", "wlasl", "autsl"):
        rs = [r for r in results if r["ds"] == ds and any(t.startswith("signer") for t in r["tags"])]
        if not rs:
            continue
        g = np.array([next(int(t[6:]) for t in r["tags"] if t.startswith("signer")) for r in rs])
        rng = np.random.default_rng(seed)
        half = np.zeros(len(rs), int)
        half[rng.permutation(len(rs))[: len(rs) // 2]] = 1
        for v in VERS:
            keep = np.array([v in r["p1"] for r in rs])
            if not keep.any():
                continue
            P1 = np.stack([r["p1"][v] for r, k in zip(rs, keep) if k])
            P2 = np.stack([r["p2"][v] for r, k in zip(rs, keep) if k])
            C = (P1.shape[1] // 18 - 3) // 3
            n1, n2 = F.p1_names(C), F.p2_names(C)
            for name, y, cls in ((f"signer_{ds}", g[keep], ["g0", "g1", "g2"]),
                                 (f"floor_{ds}", half[keep], ["half0", "half1"])):
                out.setdefault(name, {})[v] = {
                    "P1": F.probe(P1, y, n1, cls, top=5), "P2": F.probe(P2, y, n2, cls, top=5)}
    return out


def gate3_probe(results: list[dict]) -> dict:
    """3-way ASL gate probe (asl_citizen / wlasl / how2sign, P1 and P2) on the gate clips and
    windows, from this module's on-the-fly canonicalisation (float32; the gate itself reads
    the float16 canon files). Gives the gate number for versions without canon files."""
    out = {}
    rs = [r for r in results if r["ds"] in F.ASL3 and "quality" in r["tags"]]
    y = np.array([F.ASL3.index(r["ds"]) for r in rs])
    for v in VERS:
        keep = np.array([v in r["p1"] for r in rs])
        P1 = np.stack([r["p1"][v] for r, k in zip(rs, keep) if k])
        P2 = np.stack([r["p2"][v] for r, k in zip(rs, keep) if k])
        C = (P1.shape[1] // 18 - 3) // 3
        out[v] = {"P1": F.probe(P1, y[keep], F.p1_names(C), F.ASL3, top=10),
                  "P2": F.probe(P2, y[keep], F.p2_names(C), F.ASL3, top=5)}
    return out


def aggregate(results: list[dict]) -> dict:
    agg = {}
    for ds in DS_ORDER:
        rs = [r for r in results if r["ds"] == ds and "quality" in r["tags"]]
        for v in VERS:
            for k in METRICS:
                agg.setdefault(ds, {}).setdefault(v, {})[k] = summarise(
                    [r["metrics"][v].get(k) for r in rs])
    return agg


def _draw(ax, x, m, dims, alpha=1.0, lw=0.8):
    a, b = dims
    for bones, col in ((BONES_C, "0.4"), (BONES_L, "tab:blue"), (BONES_R, "tab:red")):
        for i, j in bones:
            if m[i] and m[j]:
                ax.plot([x[i, a], x[j, a]], [x[i, b], x[j, b]], color=col, lw=lw, alpha=alpha)
    f = m[51:77] > 0
    ax.scatter(x[51:77][f, a], x[51:77][f, b], s=1.5, color="tab:green", alpha=alpha)
    if m[0]:
        ax.scatter([x[0, a]], [x[0, b]], s=6, color="k", alpha=alpha)


def figures(results: list[dict], fig_dir: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)
    paths = []
    frames = [0, 16, 33, 49]
    cols = [("raw", (0, 1), "raw image (px)"), ("canon-v1", (0, 1), "v1 front x-y"),
            ("canon-v1", (2, 1), "v1 side z-y"), ("canon-v2c", (0, 1), "v2c front"),
            ("canon-v2c", (2, 1), "v2c side"), ("canon-v3", (0, 1), "v3 front"),
            ("canon-v3", (2, 1), "v3 side"), ("canon-v3.1", (0, 1), "v3.1 front"),
            ("canon-v3.1", (2, 1), "v3.1 side"), ("canon-v4", (0, 1), "v4 front"),
            ("canon-v4", (2, 1), "v4 side")]
    for ds in DS_ORDER:
        viz = [r for r in results if r["ds"] == ds and "viz" in r][:3]
        if not viz:
            continue
        fig, axes = plt.subplots(len(viz), len(cols), figsize=(2.0 * len(cols), 2.2 * len(viz)),
                                 squeeze=False)
        for i, r in enumerate(viz):
            for c, (v, dims, title) in enumerate(cols):
                ax = axes[i, c]
                x, m = r["viz"][v]
                for k, t in enumerate(frames):
                    _draw(ax, x[t], m[t], dims, alpha=0.25 + 0.25 * k)
                ax.set_xticks([])
                ax.set_yticks([])
                if v != "raw":
                    ax.set_xlim(-2.2, 2.2)
                    ax.set_ylim(2.2, -1.6)
                else:
                    ax.invert_yaxis()
                ax.set_aspect("equal", adjustable="datalim")
                if i == 0:
                    ax.set_title(title, fontsize=8)
            axes[i, 0].set_ylabel(r["clip"].split("__")[-1][:22], fontsize=6)
        fig.suptitle(f"{ds}: 4 frames of the 2-s window overlaid (later = darker); "
                     "blue = signer left, red = signer right, green = face", fontsize=8)
        fig.tight_layout()
        p = os.path.join(fig_dir, f"skeleton_{ds}.png")
        fig.savefig(p, dpi=90)
        plt.close(fig)
        paths.append(p)
        paths.append(_gif(viz[0], ds, fig_dir, plt))
    cvers = [("raw", "0.7"), ("canon-v1", "tab:blue"), ("canon-v2c", "tab:orange"),
             ("canon-v3", "tab:green"), ("canon-v3.1", "tab:purple"), ("canon-v4", "tab:red")]
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(10, 2.2 * len(METRICS)))
    for ax, k in zip(axes, METRICS):
        for di, ds in enumerate(DS_ORDER):
            rs = [r for r in results if r["ds"] == ds and "quality" in r["tags"]]
            for vi, (v, col) in enumerate(cvers):
                if v == "raw" and k in ("shoulder_width", "far_frame_pct", "jitter",
                                        "torso_up_deg", "up_spread_deg"):
                    continue
                vals = [r["metrics"][v].get(k) for r in rs]
                vals = [x for x in vals if x is not None and np.isfinite(x)]
                if vals:
                    bp = ax.boxplot(vals, positions=[di + (vi - 2.5) * 0.14], widths=0.12,
                                    patch_artist=True, showfliers=True,
                                    flierprops=dict(markersize=1.5))
                    for b in bp["boxes"]:
                        b.set_facecolor(col)
        ax.set_xticks(range(len(DS_ORDER)))
        ax.set_xticklabels(DS_ORDER, fontsize=7)
        ax.set_title(k, fontsize=8)
        if k == "far_frame_pct":
            ax.set_yscale("symlog")
    axes[0].legend(handles=[mpatches.Patch(color=c, label=v) for v, c in cvers], fontsize=7,
                   ncol=4)
    fig.tight_layout()
    p = os.path.join(fig_dir, "metrics_boxplots.png")
    fig.savefig(p, dpi=80)
    plt.close(fig)
    paths.append(p)
    return paths


def _gif(r, ds, fig_dir, plt):
    from matplotlib.animation import FuncAnimation, PillowWriter
    fig, axes = plt.subplots(1, 2, figsize=(4.4, 2.4))
    xr, mr = r["viz"]["raw"]
    xc, mc = r["viz"]["canon-v4"]
    v = xr[mr > 0]
    lim = (v[:, 0].min() - 20, v[:, 0].max() + 20, v[:, 1].max() + 20, v[:, 1].min() - 20) \
        if len(v) else (0, 1, 1, 0)

    def draw(t):
        for ax in axes:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
        _draw(axes[0], xr[t], mr[t], (0, 1))
        axes[0].set_xlim(lim[0], lim[1])
        axes[0].set_ylim(lim[2], lim[3])
        axes[0].set_aspect("equal")
        axes[0].set_title("raw image", fontsize=7)
        _draw(axes[1], xc[t], mc[t], (0, 1))
        axes[1].set_xlim(-2.2, 2.2)
        axes[1].set_ylim(2.2, -1.6)
        axes[1].set_aspect("equal")
        axes[1].set_title("canon-v4 front", fontsize=7)
    anim = FuncAnimation(fig, draw, frames=range(0, len(xr), 2))
    p = os.path.join(fig_dir, f"anim_{ds}.gif")
    anim.save(p, writer=PillowWriter(fps=12), dpi=60)
    plt.close(fig)
    return p


def analyse(out_dir: str, fig_dir: str, json_out: str) -> dict:
    import pickle
    out_dir = os.path.expanduser(out_dir)
    with open(os.path.join(out_dir, "clips.pkl"), "rb") as f:
        d = pickle.load(f)
    # imap_unordered returns clips in completion order; the CV folds and the random halves
    # depend on the order, so sort for a deterministic result
    results = sorted(d["results"], key=lambda r: (r["ds"], r["clip"]))
    agg = aggregate(results)
    cal = calibration(results)
    g3 = gate3_probe(results)
    paths = figures(results, fig_dir)

    def slim(r):
        return dict(acc=r["acc"], acc_std=r["acc_std"], chance=r["chance"], n=r["n"],
                    confusion=r["confusion"], top=[t["name"] for t in r["top_features"]])
    res = dict(metrics=agg,
               calibration={k: {v: {p: slim(r[p]) for p in ("P1", "P2")} for v, r in vv.items()}
                            for k, vv in cal.items()},
               gate3_from_quality={v: {p: slim(r[p]) for p in ("P1", "P2")} for v, r in g3.items()},
               signer_sets=d["signer_info"], figures=[os.path.basename(p) for p in paths],
               n_clips={ds: sum(1 for r in results if r["ds"] == ds and "quality" in r["tags"])
                        for ds in DS_ORDER})
    with open(json_out, "w") as f:
        json.dump(res, f, indent=1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="canon normalisation quality + calibration")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--selection", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--workers", type=int, default=16)
    a_ = sub.add_parser("analyse")
    a_.add_argument("--out-dir", required=True)
    a_.add_argument("--fig-dir", required=True)
    a_.add_argument("--json", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.selection, a.out_dir, a.workers, log=lambda s: print(s, flush=True))
    else:
        res = analyse(a.out_dir, a.fig_dir, a.json)
        print(json.dumps({k: {v: (round(x["P1"]["acc"], 3), round(x["P2"]["acc"], 3))
                              for v, x in vv.items()} for k, vv in res["calibration"].items()},
                         indent=1))


if __name__ == "__main__":
    main()
