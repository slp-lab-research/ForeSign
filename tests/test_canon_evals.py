import json
import os

import numpy as np
import pytest
import torch

from tests.test_canon_data import write_canon

TINY = dict(d_model=24, depth=2, n_heads=2, pred_dim=12, pred_depth=1, pred_heads=2,
            ema_base=0.99, t_patch=2, clip_len=32, input_format="c77", in_dim=3)


def tiny_ckpt(tmp_path, **over):
    from foresign.build import build_model
    cfg = {**TINY, **over}
    torch.manual_seed(0)
    p = tmp_path / "ck.pt"
    torch.save({"model": build_model(cfg).state_dict(), "step": 0, "cfg": cfg}, p)
    return str(p)


def test_confmode_c77():
    from foresign.confmode import resolve_drop_conf
    assert resolve_drop_conf({"input_format": "c77"}) is False
    os.environ["FS_DROP_CONF"] = "1"
    try:
        with pytest.raises(ValueError):
            resolve_drop_conf({"input_format": "c77"})
    finally:
        del os.environ["FS_DROP_CONF"]


# ---- ISLR labels / splits ----------------------------------------------------------------

def test_islr_labels():
    from evals.islr_canon import clip_label
    autsl = {"train/signer0_sample1": "ugly"}
    assert clip_label("autsl", "autsl__train__signer0_sample1_color", autsl) == "ugly"
    assert clip_label("autsl", "autsl__train__signer0_sample9_color", autsl) is None
    assert clip_label("wlasl", "wlasl__test__17301-wlasl-don_t_want") == "don_t_want"
    assert clip_label("lsa64", "lsa64__012_003_004") == "012"
    with pytest.raises(ValueError):
        clip_label("slovo", "slovo__x")


def test_collect_canon(tmp_path):
    from evals.islr_canon import collect_canon
    root = tmp_path / "c"
    root.mkdir()
    write_canon(root / "lsa64.h5", {f"lsa64__00{c}_00{s}_001": (30, "") for c in (1, 2)
                                    for s in (1, 2, 9)})
    write_canon(root / "wlasl.h5", {"wlasl__train__1-wlasl-a": (30, "train"),
                                    "wlasl__val__2-wlasl-a": (30, "val"),
                                    "wlasl__test__3-wlasl-a": (30, "test"),
                                    "wlasl__test__4-wlasl-zzz": (30, "test")})
    tr, te, n, _ = collect_canon("lsa64", str(root), cache_dir=str(tmp_path / "cache"))
    assert n == 2 and len(tr) == 4 and len(te) == 2
    assert all(cid.split("_")[3] == "009" for _, cid, _ in te)
    tr, te, n, _ = collect_canon("wlasl", str(root), cache_dir=str(tmp_path / "cache"))
    assert n == 1 and [c for _, c, _ in tr] == ["wlasl__train__1-wlasl-a"]
    assert [c for _, c, _ in te] == ["wlasl__test__3-wlasl-a"]          # unseen class dropped


def test_extract_canon_shapes(tmp_path):
    from evals.islr_canon import extract_canon
    from foresign.adapters import ForeSignAdapter
    root = tmp_path / "c"
    root.mkdir()
    write_canon(root / "lsa64.h5", {"lsa64__001_001_001": (20, ""), "lsa64__001_002_001": (50, "")})
    ad = ForeSignAdapter(tiny_ckpt(tmp_path), device="cpu")
    items = [(0, "lsa64__001_001_001", 0), (0, "lsa64__001_002_001", 1)]
    pooled, toks, masks, y = extract_canon(ad, str(root), "lsa64", ["lsa64.h5"], items, bs=1)
    assert pooled.shape == (2, 24) and toks.shape == (2, 16, 24)
    assert masks[0].sum() == 10 and masks[1].sum() == 16 and y.tolist() == [0, 1]


# ---- dataset-ID probe --------------------------------------------------------------------

def _id_root(tmp_path):
    root = tmp_path / "c"
    root.mkdir()
    for ds, n in (("asl_citizen", 12), ("wlasl", 12), ("how2sign", 12)):
        clips = {f"{ds}__train__{i}": (60 + i, "train") for i in range(n)}
        clips[f"{ds}__test__x"] = (60, "test")
        clips[f"{ds}__train__short"] = (20, "train")
        write_canon(root / f"{ds}.h5", clips)
    return root


def test_dsid_sample_is_fixed_and_balanced(tmp_path):
    from evals.dsid_probe import sample_windows
    root = _id_root(tmp_path)
    a, _ = sample_windows(str(root), ["asl_citizen", "wlasl", "how2sign"], n=10, win=50, seed=0,
                       cache_dir=str(tmp_path / "cache"))
    b, _ = sample_windows(str(root), ["asl_citizen", "wlasl", "how2sign"], n=10, win=50, seed=0,
                       cache_dir=str(tmp_path / "cache"))
    assert a == b and len(a) == 30
    assert all("__train__" in r[2] and "short" not in r[2] for r in a)
    assert {r[0] for r in a} == {"asl_citizen", "wlasl", "how2sign"}
    assert all(0 <= r[3] <= 60 + 12 - 50 for r in a)


def test_dsid_probe_end_to_end(tmp_path, monkeypatch):
    from evals import dsid_probe
    root = _id_root(tmp_path)
    monkeypatch.setenv("FS_RESULTS_DIR", str(tmp_path / "res"))
    monkeypatch.setenv("FS_CANON_ROOT", str(root))
    res = dsid_probe.run(tiny_ckpt(tmp_path, clip_len=64), tag="t", n=10, win=50, device="cpu",
                         cache_dir=str(tmp_path / "cache"), folds=2)
    assert res["n_per_dataset"] == {"asl_citizen": 10, "wlasl": 10, "how2sign": 10}
    for k in ("encoder", "input_meanpose"):
        assert 0.0 <= res[k]["acc_mean"] <= 1.0 and len(res[k]["confusion"]) == 3
    assert (tmp_path / "res" / "dsid_t.json").exists()


def test_linear_cv_separable():
    from evals.dsid_probe import linear_cv
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(0, 1, (30, 5)), rng.normal(5, 1, (30, 5))])
    y = np.array([0] * 30 + [1] * 30)
    r = linear_cv(X, y, folds=5)
    assert r["acc_mean"] > 0.95 and np.array(r["confusion"]).sum() == 60


# ---- CSLR on canon PHOENIX ---------------------------------------------------------------

def test_cslr_canon_mapping_and_coverage(tmp_path, monkeypatch):
    from evals import cslr_phoenix as cp
    root = tmp_path / "c"
    root.mkdir()
    write_canon(root / "phoenix14t.h5", {"phoenix14t__train__a": (40, "train"),
                                         "phoenix14t__dev__b": (40, "val")})
    ann = tmp_path / "ann"
    ann.mkdir()
    for split, rows in (("train", [("a", "X Y"), ("missing", "Z")]), ("dev", [("b", "X")]),
                        ("test", [])):
        (ann / f"phoenix14t_{split}.json").write_text(json.dumps(
            [{"id": i, "h5_key": f"ph14t/{split}/{i}", "gloss": g, "text": ""} for i, g in rows]),
            encoding="utf-8")
    monkeypatch.setenv("FS_ANN_DIR", str(ann))
    monkeypatch.setenv("FS_CANON_ROOT", str(root))
    items, vocab = cp.load_split("train", ds="canon_ph14t")
    assert [k for k, _ in items] == ["phoenix14t__train__a", None]
    store = cp.preload(items, vocab, cp.h5_path("canon_ph14t"), in_dim=3)
    assert store[0][0].shape == (40, 77 * 4) and store[1][0] is None
    cov = cp.coverage({"train": items, "dev": cp.load_split("dev", vocab, ds="canon_ph14t")[0]})
    assert cov == {"train": {"ann": 2, "found": 1}, "dev": {"ann": 1, "found": 1}}


def test_cslr_window_encoder_c77(tmp_path):
    from evals.cslr_phoenix import WindowEncoder
    enc = WindowEncoder(tiny_ckpt(tmp_path), freeze=True)
    x = torch.zeros(2, 64, 77 * 4)
    z, tl = enc(x, torch.tensor([64, 40]))
    assert z.shape == (2, 32, 24) and tl.tolist() == [32, 20]


def test_cslr_rejects_format_mismatch():
    from evals.cslr_phoenix import check_compat
    with pytest.raises(ValueError):
        check_compat({}, "canon_ph14t")                 # 2D (resona77) checkpoint
    check_compat({"input_format": "c77"}, "canon_ph14")
