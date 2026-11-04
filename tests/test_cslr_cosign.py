import json
import math

import numpy as np
import pytest
import torch

import evals.cslr_cosign as C
from tests.test_canon_data import write_canon
from tests.test_canon_evals import tiny_ckpt


def test_temporal_scale_range():
    x = np.arange(100)[:, None].astype(np.float32)
    rng = np.random.default_rng(0)
    for _ in range(50):
        y = C.temporal_scale(x, rng)
        assert 80 <= len(y) <= 120 and y[0, 0] == 0 and y[-1, 0] == 99
        assert np.all(np.diff(y[:, 0]) >= 0)


def test_greedy_and_beam():
    lp = np.log(np.array([[0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.9, 0.05, 0.05],
                          [0.1, 0.1, 0.8]]))
    assert C.greedy(lp, 4) == [1, 2] and C.beam_search(lp, 4) == [1, 2]
    # classic case: greedy says "" (blank, blank: 0.36) but P("a") = 0.64
    lp2 = np.log(np.array([[0.6, 0.4], [0.6, 0.4]]))
    assert C.greedy(lp2, 2) == [] and C.beam_search(lp2, 2) == [1]
    # repeated label needs a blank in between
    lp3 = np.log(np.array([[0.05, 0.9, 0.05], [0.9, 0.05, 0.05], [0.05, 0.9, 0.05]]))
    assert C.beam_search(lp3, 3) == [1, 1]


def test_model_shapes_and_groups(tmp_path):
    m = C.CoSignCSLR(tiny_ckpt(tmp_path), n_cls=7)
    x = torch.zeros(2, 64, 77 * 4)
    logit, aux, fl = m(x, torch.tensor([64, 30]))
    assert logit.shape == (2, 16, 8) and aux.shape == (2, 16, 8) and fl.tolist() == [16, 7]
    g = {gg["name"]: gg["lr"] for gg in C.param_groups(m)}
    assert g["head-1"] == pytest.approx(4e-4) and g["enc0"] == pytest.approx(4e-4 * 0.75 ** 3)
    assert all(p.requires_grad for p in m.parameters())


def test_run_end_to_end(tmp_path, monkeypatch):
    root = tmp_path / "c"; root.mkdir()
    ids = {"train": [f"tr{i}" for i in range(6)], "dev": ["dv0", "dv1"], "test": ["te0"]}
    write_canon(root / "phoenix14t.h5", {f"phoenix14t__{s}__{u}": (40 + 3 * k, s if s != "dev" else "val")
                                         for s, us in ids.items() for k, u in enumerate(us)})
    ann = tmp_path / "ann"; ann.mkdir()
    ev = tmp_path / "ev"; ev.mkdir()
    gl = ["REGEN SONNE", "WIND", "REGEN WIND SONNE"]
    for s, us in ids.items():
        rows = [{"id": u, "h5_key": "", "gloss": gl[k % 3], "text": ""} for k, u in enumerate(us)]
        (ann / f"phoenix14t_{s}.json").write_text(json.dumps(rows), encoding="utf-8")
        if s != "train":
            (ev / f"PHOENIX-2014-T-groundtruth-{s}.stm").write_text(
                "\n".join(f"{r['id']} 1 x 0.0 1.79769e+308 {r['gloss']}" for r in rows) + "\n",
                encoding="utf-8")
    monkeypatch.setenv("FS_ANN_DIR", str(ann))
    monkeypatch.setenv("FS_PHOENIX_EVAL_DIR", str(ev))
    monkeypatch.setenv("FS_CANON_ROOT", str(root))
    monkeypatch.setenv("FS_RESULTS_DIR", str(tmp_path / "res"))
    out = C.run(tiny_ckpt(tmp_path), "t", "ph14t", device="cpu", epochs=2, root=str(root))
    assert out["coverage"]["train"] == {"ann": 6, "found": 6} and out["vocab"] == 3
    assert len(out["history"]) == 2 and out["best_epoch"] in (1, 2)
    assert 0 <= out["test_wer_at_best_dev"] and (tmp_path / "res" / "cosign_t_ph14t_s0.json").exists()
