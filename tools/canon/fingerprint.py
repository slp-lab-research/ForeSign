"""Dataset-fingerprint gate (spec 2026-10-29 section 3).

Can a linear probe tell which ASL dataset a 2-second window comes from, using only
content-agnostic statistics? Three steps, runnable on different machines:

  select             choose 300 train clips per dataset (rng seed 0) from the canon-v1
                     files, clips >= 2 s, and one random 2-s window per clip (start in s)
                     -> selection.json
  features           --version canon | raw         -> feats_<version>.npz
                     (--version resona77 reads the earlier 2D pose files; not released)
                     canon also yields canon_noz (z dropped).
  probe     (any)    5-fold stratified logistic regression on P1 / P2, 3-way and 5-way,
                     the pre-registered decision rule -> results json + markdown

Every version's window is the same time span [start_s, start_s + 2 s) of the clip, resampled
to 25 fps (50 frames) with ``canon.resample`` so that velocity / acceleration are per 1/25 s
everywhere (RESONA-77 and raw are stored at the source fps).

P1 (per window, per foresign 18 joint groups): mean / std / range of each coordinate,
mean |velocity|, mean |acceleration|, missing rate. P2: window mean of every joint coordinate.
Invalid joints are excluded from the statistics; a group with no valid joint gets 0.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

from foresign.groups import GROUP_NAMES, GROUPS
from tools.canon.canon import FPS_OUT, raw77_2d, resample

WIN_S = 2.0
WIN = int(WIN_S * FPS_OUT)          # 50
N_PER_DS = 300
ASL3 = ["asl_citizen", "wlasl", "how2sign"]
ASL5 = ASL3 + ["youtube_asl", "openasl"]
# probe tasks: the gate (3way), secondary 5way, and the within-task-type analysis (spec 6)
TASKS = {"3way": ASL3, "5way": ASL5, "isolated": ["asl_citizen", "wlasl"],
         "continuous": ["how2sign", "youtube_asl", "openasl"]}
CANDIDATES = ["canon_v2a", "canon_v2b", "canon_v2c"]   # spec section 6 (+ amendment)
RESONA_FILE = {"asl_citizen": "asl_citizen", "wlasl": "wlasl", "how2sign": "how2sign",
               "youtube_asl": "yt_asl", "openasl": "openasl"}
RAW_DIRS = {"asl_citizen": ["~/reextract/out/asl_citizen"],
            "wlasl": ["~/reextract/out/wlasl"],
            "how2sign": ["~/reextract/out/how2sign"],
            "openasl": ["~/reextract/out/openasl"],
            "youtube_asl": ["~/reextract/out/youtube_asl/b*"]}
COORDS = "xyz"


# ------------------------------------------------------------------ windows
def take_window(x: np.ndarray, mask: np.ndarray, fps: float, start_s: float):
    """Source-fps arrays (T,J,C), (T,J) -> the 2-s window at start_s, at 25 fps (50 frames)."""
    a = max(0, int(np.floor(start_s * fps + 1e-6)))
    b = a + int(np.ceil(WIN_S * fps)) + 2
    xs, ms = x[a:b], mask[a:b]
    if abs(fps - FPS_OUT) > 1e-6:
        xs, ms = resample(xs, ms, fps, FPS_OUT)
    if len(xs) < WIN:                                   # last frames of a clip: pad by edge
        pad = WIN - len(xs)
        xs = np.concatenate([xs, np.repeat(xs[-1:], pad, 0)])
        ms = np.concatenate([ms, np.repeat(ms[-1:], pad, 0)])
    return xs[:WIN].astype(np.float32), ms[:WIN].astype(np.uint8)


def pick_start(rng, n_frames: int, fps: float) -> float | None:
    dur = n_frames / fps
    if dur < WIN_S:
        return None
    return float(rng.uniform(0.0, dur - WIN_S))


# ------------------------------------------------------------------ features
def p1_names(C: int) -> list[str]:
    names = []
    for g in GROUP_NAMES:
        for st in ("mean", "std", "range"):
            names += [f"{g}.{st}_{COORDS[c]}" for c in range(C)]
        names += [f"{g}.vel", f"{g}.acc", f"{g}.missing"]
    return names


def p1_features(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """x (T,77,C), m (T,77) -> 18 groups x (3C + 3)."""
    T, _, C = x.shape
    v = m > 0
    vel_ok = v[1:] & v[:-1]
    vel = np.linalg.norm(x[1:] - x[:-1], axis=-1)
    acc_ok = v[2:] & v[1:-1] & v[:-2]
    acc = np.linalg.norm(x[2:] - 2 * x[1:-1] + x[:-2], axis=-1)
    out = []
    for g in GROUP_NAMES:
        J = GROUPS[g]
        gv = v[:, J]
        vals = x[:, J][gv]                              # (n, C)
        if len(vals):
            out += list(vals.mean(0)) + list(vals.std(0)) + list(vals.max(0) - vals.min(0))
        else:
            out += [0.0] * (3 * C)
        vo, ao = vel_ok[:, J], acc_ok[:, J]
        out.append(float(vel[:, J][vo].mean()) if vo.any() else 0.0)
        out.append(float(acc[:, J][ao].mean()) if ao.any() else 0.0)
        out.append(1.0 - float(gv.mean()))
    return np.asarray(out, np.float32)


def p2_names(C: int) -> list[str]:
    return [f"j{j}.{COORDS[c]}" for j in range(77) for c in range(C)]


def p2_features(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    v = (m > 0)[..., None]
    n = v.sum(0)
    return np.where(n > 0, (x * v).sum(0) / np.maximum(n, 1), 0).reshape(-1).astype(np.float32)


# ------------------------------------------------------------------ RESONA-77 (old data)
def resona_normalize(arr: np.ndarray, conf_thr: float = 0.3, min_px: float = 10.0,
                     fallback: float = 100.0):
    """numpy port of resona/data/normalize.py shoulder_normalize: (T,234) ->
    (xy (T,77,2) float32, mask (T,77) uint8); invalid = conf <= conf_thr, zeroed."""
    j = np.asarray(arr, np.float64)[:, :231].reshape(-1, 77, 3)
    xy, conf = j[..., :2], j[..., 2]
    ok = (conf[:, 1] > conf_thr) & (conf[:, 2] > conf_thr)
    if ok.any():
        center = (0.5 * (xy[ok, 1] + xy[ok, 2])).mean(0)
        scale = max(float(np.linalg.norm(xy[ok, 1] - xy[ok, 2], axis=-1).mean()), min_px)
    else:
        valid = conf > conf_thr
        center = xy[valid].mean(0) if valid.any() else np.zeros(2)
        scale = fallback
    out = (xy - center) / scale
    mask = conf > conf_thr
    out[~mask] = 0
    return out.astype(np.float32), mask.astype(np.uint8)


def resona_fps(ds: str, attrs) -> float:
    if "fps" in attrs and float(attrs["fps"]) > 0:
        return float(attrs["fps"])
    return 25.0 if "phoenix" in ds else 30.0


def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


def resona_key(ds: str, cid: str) -> str:
    """New clip id -> the matching key stem in the old RESONA-77 file (best effort; the
    caller checks it exists). Old keys are compared after ``sanitize``."""
    rest = cid.split("__", 1)[1] if "__" in cid else cid
    parts = rest.split("__")
    if ds == "wlasl":                                    # wlasl__<split>__<vid>-wlasl-<gloss>
        m = re.match(r"(\d+)", parts[-1])
        return m.group(1) if m else parts[-1]
    return parts[-1]


# ------------------------------------------------------------------ probe
def probe(X: np.ndarray, y: np.ndarray, names: list[str], classes: list[str],
          seed: int = 0, top: int = 15) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def model():
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=20000))
    accs, pred = [], np.zeros_like(y)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        mdl = model().fit(X[tr], y[tr])
        pred[te] = mdl.predict(X[te])
        accs.append(float((pred[te] == y[te]).mean()))
    full = model().fit(X, y)
    coef = full[-1].coef_                               # (K, F) on standardised features
    imp = np.abs(coef).max(0)
    order = np.argsort(-imp)[:top]
    feats = [dict(name=names[i], weight=float(imp[i]),
                  class_means={classes[k]: float(X[y == k, i].mean()) for k in range(len(classes))})
             for i in order]
    return dict(acc=float(np.mean(accs)), acc_std=float(np.std(accs)), fold_acc=accs,
                n=int(len(y)), chance=1.0 / len(classes), classes=classes,
                confusion=confusion_matrix(y, pred, labels=list(range(len(classes)))).tolist(),
                top_features=feats)


def decide(res3: dict, version: str = "canon") -> dict:
    """Pre-registered rule (spec section 3): a canonicalisation passes if, on the 3-way ASL
    task, its P1 accuracy <= RESONA-77 P1 accuracy - 0.15 and <= 0.70."""
    r = res3["resona77"]["P1"]["acc"]
    c = res3[version]["P1"]["acc"]
    ok1, ok2 = c <= r - 0.15, c <= 0.70
    return dict(rule="3-way P1 acc <= RESONA-77 3-way P1 acc - 0.15 AND <= 0.70",
                version=version, resona77_p1=r, canon_p1=c, threshold=min(r - 0.15, 0.70),
                drop_ok=bool(ok1), abs_ok=bool(ok2), passed=bool(ok1 and ok2))


# revised gate (spec section 8): gate label -> calibration (quality.py) version name
REVISED = {"canon": "canon-v1", "canon_v2c": "canon-v2c", "canon_v3": "canon-v3",
           "canon_v3_1": "canon-v3.1", "canon_v4": "canon-v4"}
SIGNER_SETS = ("asl_citizen", "autsl", "wlasl")


def revised_gate(res3: dict, calib: dict, margin: float = 0.10, fallback: dict | None = None) -> dict:
    """Spec section 8: pass iff 3-way P1 <= S + 0.10, S = the highest within-dataset
    3-signer-group P1 among ASL Citizen, AUTSL and WLASL for the same canon version.
    Pre-registered for canon-v3.1; reported post hoc for the earlier versions. ``fallback``
    (quality.py's gate3_from_quality, keyed by version name) supplies the 3-way P1 of a
    version that has no canon files (computed on the fly, same clips / windows / probe)."""
    out = {}
    for v, qv in REVISED.items():
        src = "canon files"
        if v in res3:
            p1 = res3[v]["P1"]["acc"]
        elif fallback and qv in fallback:
            p1, src = fallback[qv]["P1"]["acc"], "on-the-fly canonicalisation (quality.py)"
        else:
            continue
        if not all(f"signer_{d}" in calib and qv in calib[f"signer_{d}"]
                                    for d in SIGNER_SETS):
            continue
        sv = {d: calib[f"signer_{d}"][qv]["P1"]["acc"] for d in SIGNER_SETS}
        S = max(sv.values())
        out[v] = dict(p1_source=src, rule="3-way P1 <= S + 0.10, S = max within-dataset 3-signer-group P1 "
                           "(asl_citizen, autsl, wlasl), same version",
                      S=S, S_by_dataset=sv, threshold=S + margin, p1=p1,
                      passed=bool(p1 <= S + margin), post_hoc=v != "canon_v3_1",
                      for_record_only=v == "canon_v4")
    return out


def select_v2(decisions: dict) -> dict:
    """Spec section 6 as amended on 2026-10-30 (before any v2 result): the variants that
    pass the gate are candidates; canon-v2c is preferred if it passes (no extrapolated
    joints), otherwise the lowest 3-way P1."""
    passing = [v for v in CANDIDATES if v in decisions and decisions[v]["passed"]]
    if not passing:
        return dict(adopted=None, passing=[], note="no canon-v2 candidate passes")
    best = "canon_v2c" if "canon_v2c" in passing else \
        min(passing, key=lambda v: decisions[v]["canon_p1"])
    return dict(adopted=best, passing=passing)


# ------------------------------------------------------------------ I/O steps
def _canon_files(canon_dir: str, ds: str) -> list[str]:
    d = os.path.expanduser(canon_dir)
    if ds == "youtube_asl":
        return sorted(glob.glob(os.path.join(d, "youtube_asl_b*.h5")))
    return [os.path.join(d, f"{ds}.h5")]


def select(canon_dir: str, datasets=ASL5, n: int = N_PER_DS, seed: int = 0) -> dict:
    """Eligible: split train ('' = no official split, used for YouTube-ASL / OpenASL),
    pose found, >= 2 s. One rng for all datasets in the fixed order."""
    import h5py
    rng = np.random.default_rng(seed)
    sel = {}
    for ds in datasets:
        elig = []
        for p in _canon_files(canon_dir, ds):
            with h5py.File(p, "r") as f:
                for cid in f.keys():
                    a = f[cid].attrs
                    if str(a["split"]) in ("train", "") and not bool(a["no_pose"]) \
                            and float(a["n_in"]) / float(a["fps_src"]) >= WIN_S \
                            and f[cid]["x"].shape[0] >= WIN:
                        elig.append((cid, os.path.basename(p), int(a["n_in"]),
                                     float(a["fps_src"])))
        elig.sort()
        idx = sorted(rng.choice(len(elig), size=min(n, len(elig)), replace=False))
        rows = []
        for i in idx:
            cid, fn, n_in, fps = elig[i]
            # the start must fit in the shorter of source and 25-fps lengths
            rows.append(dict(clip=cid, file=fn, start_s=pick_start(rng, n_in, fps),
                             n_in=n_in, fps_src=fps))
        sel[ds] = dict(n_eligible=len(elig), clips=rows)
    return sel


def _stack(rows):
    return (np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows]))


def features_canon(sel: dict, canon_dir: str, label: str = "canon") -> dict:
    import h5py
    out = {label: [], label + "_noz": []}
    meta = []
    for ds, d in sel.items():
        files = {}
        for r in d["clips"]:
            p = os.path.join(os.path.expanduser(canon_dir), r["file"])
            f = files.setdefault(p, h5py.File(p, "r"))
            g = f[r["clip"]]
            x, m = take_window(g["x"][()].astype(np.float32), g["mask"][()], FPS_OUT, r["start_s"])
            out[label].append((p1_features(x, m), p2_features(x, m)))
            out[label + "_noz"].append((p1_features(x[..., :2], m), p2_features(x[..., :2], m)))
            meta.append((ds, r["clip"]))
        for f in files.values():
            f.close()
    return {k: (*_stack(v), meta) for k, v in out.items()}


def raw_index(ds: str, dirs=None) -> dict:
    import h5py
    idx = {}
    for pat in (dirs or RAW_DIRS[ds]):
        for d in sorted(glob.glob(os.path.expanduser(pat))):
            for p in sorted(glob.glob(os.path.join(d, "shard_*.h5"))):
                with h5py.File(p, "r") as f:
                    for k in f.keys():
                        idx[k] = p
    return idx


def features_raw(sel: dict, raw_dirs=None) -> dict:
    import h5py
    rows, meta = [], []
    for ds, d in sel.items():
        idx = raw_index(ds, (raw_dirs or {}).get(ds))
        for r in d["clips"]:
            with h5py.File(idx[r["clip"]], "r") as f:
                g = f[r["clip"]]
                fps = float(g.attrs["fps"])
                a = int(np.floor(r["start_s"] * fps))
                b = a + int(np.ceil(WIN_S * fps)) + 2
                xy, m = raw77_2d(g["kp"][a:b], g["flags"][a:b])
            x, m = take_window(xy, m, fps, r["start_s"] - a / fps)
            rows.append((p1_features(x, m), p2_features(x, m)))
            meta.append((ds, r["clip"]))
    return {"raw": (*_stack(rows), meta)}


def clip_containers(f, max_depth: int = 4) -> list[str]:
    """Groups of an old RESONA-77 file whose children are the (T,234) clip datasets
    (e.g. asl_citizen/all, wlasl2000/all, how2sign/train); empty groups are skipped."""
    import h5py
    out, todo = [], [("", f, 0)]
    while todo:
        path, g, d = todo.pop()
        it = iter(g)
        first = next(it, None)
        if first is None:
            continue
        if isinstance(g[first], h5py.Dataset):
            out.append(path)
            continue
        if d < max_depth:
            todo += [(f"{path}/{k}".lstrip("/"), g[k], d + 1) for k in g]
    return sorted(out)


def features_resona(sel: dict, resona_dir: str, seed: int = 1) -> dict:
    """Same clip and window where the old file has the clip (matched by ``resona_key``;
    the fps is then the new data's source fps, e.g. 23.976 for How2Sign whose old file has
    no fps attr), otherwise an independent clip >= 2 s from the old file's train split (or all clips
    when the file has no train group), seed ``seed``."""
    import h5py
    rng = np.random.default_rng(seed)
    rows, meta, match = [], [], {}
    for ds, d in sel.items():
        path = os.path.join(resona_dir, f"skeleton_resona77_{RESONA_FILE[ds]}.h5")
        with h5py.File(path, "r") as f:
            groups = {}                                # sanitize(key) -> dataset paths
            for cont in clip_containers(f):            # list names only (opening each
                for k in f[cont]:                      # object is far slower)
                    groups.setdefault(sanitize(k), []).append(f"{cont}/{k}")
            used = []
            for r in d["clips"]:
                key = sanitize(resona_key(ds, r["clip"]))
                cands = groups.get(key, [])
                cands = sorted(cands, key=lambda n: ("/train/" not in n, n))
                if cands:       # same source video: its true fps comes from the new data
                    used.append((cands[0], r["start_s"], r["fps_src"]))
            n_match = len(used)
            if n_match < len(d["clips"]):
                allp = sorted(n for v in groups.values() for n in v)
                tr = [n for n in allp if "/train/" in n]
                pool = tr or allp
                have = {u[0] for u in used}
                order = rng.permutation(len(pool))
                for i in order:
                    if len(used) >= len(d["clips"]):
                        break
                    n_ = pool[i]
                    if n_ in have:
                        continue
                    ds_ = f[n_]
                    fps_ = resona_fps(ds, ds_.attrs)
                    st = pick_start(rng, ds_.shape[0], fps_)
                    if st is not None:
                        used.append((n_, st, fps_))
                        have.add(n_)
            for name, st, fps in used:
                arr = f[name][()]
                if st + WIN_S > arr.shape[0] / fps:     # matched clip shorter in the old file
                    st = max(0.0, arr.shape[0] / fps - WIN_S)
                xy, m = resona_normalize(arr)
                x, m = take_window(xy, m, fps, st)
                rows.append((p1_features(x, m), p2_features(x, m)))
                meta.append((ds, name))
            match[ds] = dict(matched=n_match, total=len(used), pool_split="train" if
                             any("/train/" in u[0] for u in used) else "all")
    return {"resona77": (*_stack(rows), meta)}, match


def save_feats(path: str, version: str, p1, p2, meta, extra=None) -> None:
    C = (p1.shape[1] // len(GROUP_NAMES) - 3) // 3
    np.savez_compressed(path, version=version, p1=p1, p2=p2,
                        ds=np.array([m[0] for m in meta]), clip=np.array([m[1] for m in meta]),
                        p1_names=np.array(p1_names(C)), p2_names=np.array(p2_names(C)),
                        extra=json.dumps(extra or {}))


def run_probes(feat_files: list[str], calibration: str | None = None) -> dict:
    res = {t: {} for t in TASKS}
    res["meta"] = {}
    for p in feat_files:
        z = np.load(p, allow_pickle=False)
        v = str(z["version"])
        res["meta"][v] = dict(file=os.path.basename(p), extra=json.loads(str(z["extra"])),
                              n_per_ds={d: int((z["ds"] == d).sum()) for d in ASL5})
        for task, dss in TASKS.items():
            keep = np.isin(z["ds"], dss)
            if not all((z["ds"] == d).any() for d in dss):
                continue
            y = np.array([dss.index(d) for d in z["ds"][keep]])
            res[task][v] = {
                "P1": probe(z["p1"][keep], y, list(z["p1_names"]), dss),
                "P2": probe(z["p2"][keep], y, list(z["p2_names"]), dss)}
    if "resona77" in res["3way"]:
        res["decisions"] = {v: decide(res["3way"], v)
                            for v in ["canon"] + CANDIDATES + ["canon_v3", "canon_v3_1", "canon_v4"]
                            if v in res["3way"]}
        if "canon" in res["decisions"]:
            res["decision"] = res["decisions"]["canon"]          # canon-v1 (section 3)
        if any(v in res["decisions"] for v in CANDIDATES):
            res["v2_selection"] = select_v2(res["decisions"])
    if calibration:
        with open(calibration) as f:
            q = json.load(f)
        res["revised_gate"] = revised_gate(res["3way"], q["calibration"],
                                           fallback=q.get("gate3_from_quality"))
    return res


VERSION_LABEL = {"resona77": "RESONA-77 (old, 2D shoulder-norm)",
                 "raw": "raw MP-L-V4V0 (2D image coords)",
                 "canon": "canon-v1 (3D)", "canon_noz": "canon-v1, z dropped (2D)",
                 "canon_v2a": "canon-v2a (3D, torso frame)",
                 "canon_v2a_noz": "canon-v2a, z dropped (2D)",
                 "canon_v2b": "canon-v2b (3D, torso frame + per-frame centring, no hips)",
                 "canon_v2b_noz": "canon-v2b, z dropped (2D)",
                 "canon_v2c": "canon-v2c (3D, no lower body: shoulder+nose frame, "
                              "per-frame centring)",
                 "canon_v2c_noz": "canon-v2c, z dropped (2D)",
                 "canon_v3": "canon-v3 (3D, v2c + smoothing + face own scale)",
                 "canon_v3_noz": "canon-v3, z dropped (2D)",
                 "canon_v3_1": "canon-v3.1 (3D, v3 with ear-midpoint up axis)",
                 "canon_v3_1_noz": "canon-v3.1, z dropped (2D)",
                 "canon_v4": "canon-v4 (3D, v3 pipeline, v1 orientation + roll; final)",
                 "canon_v4_noz": "canon-v4, z dropped (2D)"}
VERSION_ORDER = list(VERSION_LABEL)
TASK_TITLE = {"3way": "3-way ASL: asl_citizen / wlasl / how2sign (gate task, chance 1/3)",
              "5way": "5-way ASL: + youtube_asl / openasl (secondary, chance 1/5)",
              "isolated": "Within task type, isolated: asl_citizen / wlasl (chance 1/2)",
              "continuous": "Within task type, continuous: how2sign / youtube_asl / openasl "
                            "(chance 1/3)"}


def report_md(res: dict) -> str:
    L = ["# Dataset-fingerprint gate (canon-v1, canon-v2a, canon-v2b)", "",
         "Spec: `docs/design/2026-10-29-canonicalisation-fingerprint-gate-design.md` "
         "§3 (gate) and §6 (canon-v2). Generated by `code/tools/canon/fingerprint.py`; raw "
         "numbers in `results/fingerprint_gate.json`.", "",
         "Task: which ASL dataset does a 2-s window come from? 300 train clips per dataset "
         "(rng seed 0), one random 2-s window each, resampled to 25 fps (50 frames); the same "
         "clips and windows for every version. 5-fold stratified, StandardScaler + L2 "
         "logistic regression C=1. P1 = per-group summary statistics (18 foresign groups: "
         "mean/std/range per coordinate, mean |vel|, mean |acc|, missing rate); P2 = "
         "window-mean pose (77 joints).", ""]
    for task in TASKS:
        if not res.get(task):
            continue
        L += [f"## {TASK_TITLE[task]}", "",
              "| version | P1 acc ± std | P2 acc ± std |", "|---|---|---|"]
        for v in VERSION_ORDER:
            if v in res[task]:
                p1, p2 = res[task][v]["P1"], res[task][v]["P2"]
                L.append(f"| {VERSION_LABEL[v]} | {p1['acc']:.3f} ± {p1['acc_std']:.3f} | "
                         f"{p2['acc']:.3f} ± {p2['acc_std']:.3f} |")
        L.append("")
    if res.get("decisions"):
        rule = next(iter(res["decisions"].values()))["rule"]
        L += ["## Decisions (pre-registered rule, not changed after seeing results)", "",
              f"Rule (§3): {rule}. Selection (§6 as amended 2026-10-30, before any v2 result): "
              "among the passing v2 variants canon-v2c is preferred, otherwise the lowest 3-way P1.", "",
              "| version | 3-way P1 | required <= | drop >= 0.15 | <= 0.70 | outcome |",
              "|---|---|---|---|---|---|"]
        for v, d in res["decisions"].items():
            L.append(f"| {VERSION_LABEL[v]} | {d['canon_p1']:.3f} | {d['threshold']:.3f} | "
                     f"{'yes' if d['drop_ok'] else 'no'} | {'yes' if d['abs_ok'] else 'no'} | "
                     f"**{'PASS' if d['passed'] else 'FAIL'}** |")
        L.append("")
        if "v2_selection" in res:
            sel = res["v2_selection"]
            what = f"adopted {sel['adopted']}" if sel["adopted"] else \
                "none adopted (no candidate passes)"
            L += [f"**canon-v2 selection: {what}.**", ""]
        if "canon_v3" in res["decisions"]:
            ok = res["decisions"]["canon_v3"]["passed"]
            L += [f"**canon-v3 (§7): {'PASSES' if ok else 'FAILS'} the gate.**", ""]
        if "canon_v3_1" in res["decisions"]:
            ok = res["decisions"]["canon_v3_1"]["passed"]
            L += [f"canon-v3.1 under the original §3 rule: {'PASS' if ok else 'FAIL'} "
                  "(reported for completeness; §8 replaces the rule).", ""]
    if res.get("revised_gate"):
        L += ["## Revised gate (§8, signer-calibrated)", "",
              "Pass iff 3-way P1 <= S + 0.10, S = the highest within-dataset 3-signer-group P1 "
              "(ASL Citizen, AUTSL, WLASL; 300 clips per group, seed 0; `docs/canon_quality.md`) "
              "for the same version. Pre-registered for canon-v3.1; the other rows are post hoc.",
              "", "| version | S (asl_citizen / autsl / wlasl) | threshold | 3-way P1 | outcome |",
              "|---|---|---|---|---|"]
        for v, d in res["revised_gate"].items():
            sv = " / ".join(f"{d['S_by_dataset'][k]:.3f}" for k in SIGNER_SETS)
            tag = " (post hoc)" if d["post_hoc"] else " (pre-registered)"
            if d.get("for_record_only"):
                tag = " (for the record, §9)"
            if d["p1_source"] != "canon files":
                tag += " ¹"
            L.append(f"| {VERSION_LABEL[v]}{tag} | {d['S']:.3f} ({sv}) | {d['threshold']:.3f} | "
                     f"{d['p1']:.3f} | **{'PASS' if d['passed'] else 'FAIL'}** |")
        if any(d["p1_source"] != "canon files" for d in res["revised_gate"].values()):
            L.append("")
            L.append("¹ 3-way P1 from the on-the-fly canonicalisation of the same gate clips and "
                     "windows (`quality.py`, float32; no canon files exist for this version). "
                     "For v1 / v2c / v3 this path reproduces the canon-file values within 0.004.")
        L.append("")
    L += ["## Confusion matrices (rows true, cols predicted; out-of-fold)", ""]
    for task in TASKS:
        for v in VERSION_ORDER:
            if v in res.get(task, {}):
                for p in ("P1", "P2"):
                    c = res[task][v][p]
                    L.append(f"- {task} {VERSION_LABEL[v]} {p} ({', '.join(c['classes'])}): "
                             f"`{c['confusion']}`")
    L += ["", "## Top separating features (largest |standardised coefficient|, fit on all "
          "windows)", ""]
    for task in ("3way", "5way"):
        for v in VERSION_ORDER:
            if v not in res.get(task, {}):
                continue
            c = res[task][v]["P1"]
            L += [f"**{task} {VERSION_LABEL[v]} P1**", "",
                  "| feature | max abs coef | class means |", "|---|---|---|"]
            for f in c["top_features"][:10]:
                cm = ", ".join(f"{k} {val:.3g}" for k, val in f["class_means"].items())
                L.append(f"| {f['name']} | {f['weight']:.2f} | {cm} |")
            L.append("")
    if res.get("meta"):
        L += ["## Inputs", ""]
        for v, m in res["meta"].items():
            L.append(f"- {v}: {m['file']}, windows per dataset {m['n_per_ds']}"
                     + (f", {json.dumps(m['extra'])}" if m["extra"] else ""))
        L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description="dataset-fingerprint gate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--canon-dir", required=True)
    s.add_argument("--out", required=True)
    f = sub.add_parser("features")
    f.add_argument("--version", choices=["canon", "raw", "resona77"], required=True)
    f.add_argument("--selection", required=True)
    f.add_argument("--canon-dir")
    f.add_argument("--resona-dir")
    f.add_argument("--out-dir", required=True)
    f.add_argument("--label", default="canon", help="name of the canon version's features")
    p = sub.add_parser("probe")
    p.add_argument("feats", nargs="+")
    p.add_argument("--json", required=True)
    p.add_argument("--md", required=True)
    p.add_argument("--calibration", default=None, help="results/canon_quality.json (section 8)")
    a = ap.parse_args(argv)
    if a.cmd == "select":
        sel = select(a.canon_dir)
        with open(a.out, "w") as fo:
            json.dump(sel, fo, indent=1)
        print({k: (v["n_eligible"], len(v["clips"])) for k, v in sel.items()})
    elif a.cmd == "features":
        with open(a.selection) as fi:
            sel = json.load(fi)
        os.makedirs(a.out_dir, exist_ok=True)
        extra = {}
        if a.version == "canon":
            res = features_canon(sel, a.canon_dir, a.label)
        elif a.version == "raw":
            res = features_raw(sel)
        else:
            res, extra = features_resona(sel, a.resona_dir)
        for v, (p1, p2, meta) in res.items():
            save_feats(os.path.join(a.out_dir, f"feats_{v}.npz"), v, p1, p2, meta,
                       {"resona_match": extra} if extra else None)
            print(v, p1.shape, p2.shape, extra or "")
    else:
        res = run_probes(a.feats, a.calibration)
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w") as fo:
            json.dump(res, fo, indent=1)
        with open(a.md, "w", encoding="utf-8") as fo:
            fo.write(report_md(res))
        for t in TASKS:
            for v, r in res[t].items():
                print(t, v, "P1 %.3f P2 %.3f" % (r["P1"]["acc"], r["P2"]["acc"]))
        print(json.dumps(res.get("decisions"), indent=1), res.get("v2_selection"))
        print(json.dumps(res.get("revised_gate"), indent=1))


if __name__ == "__main__":
    main()
