"""fingerprint gate: windows, features, RESONA-77 normaliser, probe, decision."""
import json

import numpy as np
import pytest

from tools.canon import fingerprint as F


def test_take_window_lengths():
    x = np.random.default_rng(0).standard_normal((300, 77, 3)).astype(np.float32)
    m = np.ones((300, 77), np.uint8)
    for fps in (25.0, 30.0, 29.97, 50.0, 23.976):
        w, wm = F.take_window(x, m, fps, 1.0)
        assert w.shape == (50, 77, 3) and wm.shape == (50, 77)
    w, _ = F.take_window(x, m, 25.0, 1.0)
    np.testing.assert_array_equal(w, x[25:75])
    # window at the very end of a clip is padded, never short
    w, _ = F.take_window(x[:60], m[:60], 30.0, 0.0)
    assert w.shape[0] == 50


def test_pick_start():
    rng = np.random.default_rng(0)
    assert F.pick_start(rng, 59, 30.0) is None
    for _ in range(100):
        s = F.pick_start(rng, 90, 30.0)
        assert 0 <= s <= 1.0


def test_p1_p2_shapes_and_values():
    T = 50
    x = np.zeros((T, 77, 3), np.float32)
    x[:, :, 0] = np.arange(T)[:, None]                # x moves 1 per frame, constant velocity
    m = np.ones((T, 77), np.uint8)
    m[:, 51:59] = 0                                    # mouth missing
    f = F.p1_features(x, m)
    names = F.p1_names(3)
    assert f.shape == (18 * 12,) and len(names) == len(f)
    d = dict(zip(names, f))
    assert d["trunk.vel"] == pytest.approx(1.0) and d["trunk.acc"] == pytest.approx(0.0)
    assert d["trunk.range_x"] == pytest.approx(49) and d["trunk.mean_x"] == pytest.approx(24.5)
    assert d["mouth.missing"] == 1.0 and d["mouth.mean_x"] == 0.0 and d["trunk.missing"] == 0
    assert len(F.p1_features(x[..., :2], m)) == 18 * 9 == len(F.p1_names(2))
    p2 = F.p2_features(x, m)
    assert p2.shape == (231,) and len(F.p2_names(3)) == 231
    assert p2[0] == pytest.approx(24.5) and p2[51 * 3] == 0


def test_resona_normalize_matches_reference():
    rng = np.random.default_rng(0)
    T = 20
    j = np.zeros((T, 77, 3))
    j[..., :2] = rng.uniform(0, 1000, (T, 77, 2))
    j[..., 2] = rng.uniform(0, 1, (T, 77))
    j[:, 1, 2] = j[:, 2, 2] = 0.9
    arr = np.concatenate([j.reshape(T, -1), np.zeros((T, 3))], 1)
    xy, m = F.resona_normalize(arr)
    center = (0.5 * (j[:, 1, :2] + j[:, 2, :2])).mean(0)
    scale = np.linalg.norm(j[:, 1, :2] - j[:, 2, :2], axis=-1).mean()
    ref = (j[..., :2] - center) / scale
    ref[j[..., 2] <= 0.3] = 0
    np.testing.assert_allclose(xy, ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(m, j[..., 2] > 0.3)


def test_resona_key():
    assert F.resona_key("asl_citizen", "asl_citizen__00010630150123391857-CANDY_1") == \
        "00010630150123391857-CANDY_1"
    assert F.sanitize("00010630150123391857-CANDY 1") == "00010630150123391857-CANDY_1"
    assert F.resona_key("wlasl", "wlasl__dev__00431-wlasl-above") == "00431"


def test_probe_separable_and_not():
    rng = np.random.default_rng(0)
    n = 90
    y = np.repeat([0, 1, 2], n // 3)
    X = rng.standard_normal((n, 5))
    r = F.probe(X, y, [f"f{i}" for i in range(5)], ["a", "b", "c"])
    assert r["acc"] < 0.6 and np.array(r["confusion"]).sum() == n
    X[:, 3] += 5 * y
    r = F.probe(X, y, [f"f{i}" for i in range(5)], ["a", "b", "c"])
    assert r["acc"] > 0.95 and r["top_features"][0]["name"] == "f3"


def test_decide_rule():
    res = {"resona77": {"P1": {"acc": 0.95}}, "canon": {"P1": {"acc": 0.75}}}
    d = F.decide(res)
    assert d["drop_ok"] and not d["abs_ok"] and not d["passed"]
    res["canon"]["P1"]["acc"] = 0.70
    assert F.decide(res)["passed"]
    res = {"resona77": {"P1": {"acc": 0.80}}, "canon": {"P1": {"acc": 0.66}}}
    assert not F.decide(res)["passed"]


def test_save_and_probe_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    paths = []
    for v, C in (("resona77", 2), ("canon", 3)):
        n = 150
        ds = np.repeat(F.ASL5, n // 5)
        p1 = rng.standard_normal((n, 18 * (3 * C + 3))).astype(np.float32)
        p2 = rng.standard_normal((n, 77 * C)).astype(np.float32)
        if v == "resona77":
            p1[:, :60] += 3 * np.array([F.ASL5.index(d) for d in ds])[:, None]
        meta = [(d, f"c{i}") for i, d in enumerate(ds)]
        p = str(tmp_path / f"feats_{v}.npz")
        F.save_feats(p, v, p1, p2, meta)
        paths.append(p)
    res = F.run_probes(paths)
    assert set(res["3way"]) == {"resona77", "canon"} and set(res["5way"]) == {"resona77", "canon"}
    assert res["3way"]["resona77"]["P1"]["acc"] > 0.9
    assert res["decision"]["passed"]
    md = F.report_md(res)
    assert "**PASS**" in md and "3way" in md and "isolated" in md
    assert set(res["isolated"]) == {"resona77", "canon"} and res["isolated"]["canon"]["P1"]["chance"] == 0.5
    assert res["continuous"]["canon"]["P1"]["classes"] == ["how2sign", "youtube_asl", "openasl"]
    json.dumps(res)


def test_clip_containers_and_resona_features(tmp_path):
    import h5py
    rng = np.random.default_rng(0)
    p = tmp_path / "skeleton_resona77_wlasl.h5"
    with h5py.File(p, "w") as f:
        f.create_group("meta")
        for k in ("00431", "00500", "00600"):
            f.create_dataset(f"wlasl2000/all/{k}", data=rng.uniform(1, 2, (90, 234)))
        f.create_group("wlasl2000/train")
    with h5py.File(p, "r") as f:
        assert F.clip_containers(f) == ["wlasl2000/all"]
    sel = {"wlasl": {"clips": [dict(clip="wlasl__train__00431-wlasl-above", start_s=0.5,
                                    fps_src=25.0),
                               dict(clip="wlasl__train__99999-wlasl-x", start_s=0.1,
                                    fps_src=25.0)]}}
    res, match = F.features_resona(sel, str(tmp_path))
    p1, p2, meta = res["resona77"]
    assert p1.shape == (2, 18 * 9) and match["wlasl"]["matched"] == 1
    assert meta[0] == ("wlasl", "wlasl2000/all/00431")


def test_select_v2_rule():
    d = lambda p, ok: dict(canon_p1=p, passed=ok)  # noqa: E731
    assert F.select_v2({"canon_v2a": d(0.8, False), "canon_v2b": d(0.9, False)})["adopted"] is None
    assert F.select_v2({"canon_v2a": d(0.6, True), "canon_v2b": d(0.9, False)})["adopted"] == "canon_v2a"
    assert F.select_v2({"canon_v2a": d(0.8, False), "canon_v2b": d(0.6, True)})["adopted"] == "canon_v2b"
    assert F.select_v2({"canon_v2a": d(0.65, True), "canon_v2b": d(0.6, True)})["adopted"] == "canon_v2b"


def test_decide_for_v2_label():
    res = {"resona77": {"P1": {"acc": 0.99}}, "canon_v2a": {"P1": {"acc": 0.5}}}
    d = F.decide(res, "canon_v2a")
    assert d["passed"] and d["version"] == "canon_v2a"


def test_select_v2_prefers_v2c():
    d = lambda p, ok: dict(canon_p1=p, passed=ok)  # noqa: E731
    dec = {"canon_v2a": d(0.5, True), "canon_v2b": d(0.55, True), "canon_v2c": d(0.65, True)}
    assert F.select_v2(dec)["adopted"] == "canon_v2c"
    dec["canon_v2c"] = d(0.75, False)
    assert F.select_v2(dec)["adopted"] == "canon_v2a"


def test_revised_gate():
    cal = {f"signer_{d}": {v: {"P1": {"acc": a}} for v in ("canon-v1", "canon-v3.1")}
           for d, a in (("asl_citizen", 0.78), ("autsl", 0.76), ("wlasl", 0.66))}
    res3 = {"canon": {"P1": {"acc": 0.92}}, "canon_v3_1": {"P1": {"acc": 0.85}},
            "canon_v2c": {"P1": {"acc": 0.5}}}
    r = F.revised_gate(res3, cal, fallback={"canon-v2c": {"P1": {"acc": 0.1}}})
    assert set(r) == {"canon", "canon_v3_1"}            # no calibration for v2c here
    assert r["canon_v3_1"]["S"] == 0.78 and r["canon_v3_1"]["passed"]
    assert not r["canon"]["passed"] and r["canon"]["post_hoc"]
    assert not r["canon_v3_1"]["post_hoc"]
