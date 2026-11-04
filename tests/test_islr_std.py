import numpy as np
import pytest
import torch

import evals.islr_std as S
from foresign.augment import MIRROR_PERM, islr_augment, mirror
from tests.test_canon_data import write_canon
from tests.test_canon_evals import tiny_ckpt


def test_sample_indices():
    c = S.sample_indices(64, 32)
    assert c.tolist() == list(range(1, 64, 2))                          # segment centres
    r1 = S.sample_indices(64, 32, np.random.default_rng(0))
    r2 = S.sample_indices(64, 32, np.random.default_rng(0))
    assert np.array_equal(r1, r2) and np.all(np.diff(r1) > 0)          # one per segment
    assert all(2 * k <= r1[k] < 2 * k + 2 for k in range(32))
    short = S.sample_indices(10, 32, np.random.default_rng(1))
    assert short.min() >= 0 and short.max() <= 9 and np.all(np.diff(short) >= 0)


def test_mirror_and_augment():
    assert np.array_equal(MIRROR_PERM[MIRROR_PERM], np.arange(77))
    x = np.random.default_rng(0).normal(size=(5, 77, 3)).astype(np.float32)
    m = np.ones((5, 77), np.uint8); m[:, 9] = 0                         # left hand wrist absent
    y, my = mirror(x, m)
    assert np.allclose(y[:, 30, 0], -x[:, 9, 0]) and my[0, 30] == 0 and my[0, 9] == 1
    z, mz = mirror(y, my)
    assert np.allclose(z, x) and np.array_equal(mz, m)                  # involution
    a, ma = islr_augment(x, m, np.random.default_rng(3))
    b, mb = islr_augment(x, m, np.random.default_rng(3))
    assert np.allclose(a, b) and a.shape == x.shape and np.array_equal(ma, mb)


@pytest.fixture
def roots(tmp_path):
    root = tmp_path / "c"; root.mkdir()
    write_canon(root / "lsa64.h5", {f"lsa64__00{c}_{s:03d}_00{r}": (40, "")
                                    for c in (1, 2, 3) for s in (1, 2, 9) for r in range(1, 6)})
    write_canon(root / "asl_citizen.h5", {
        "asl_citizen__001-CAT": (30, "train"), "asl_citizen__002-CAT": (30, "test"),
        "asl_citizen__003-DOG": (30, "train"), "asl_citizen__004-DOG": (30, "val"),
        "asl_citizen__005-EMU": (30, "test")})
    return root


def test_collect_splits(roots, tmp_path):
    tr, te, cls, _ = S.collect("lsa64_signer", str(roots), cache_dir=str(tmp_path / "k"))
    assert len(cls) == 3 and len(tr) == 30 and len(te) == 15
    tr0, te0, _, _ = S.collect("lsa64_rand", str(roots), seed=0, cache_dir=str(tmp_path / "k"))
    tr1, te1, _, _ = S.collect("lsa64_rand", str(roots), seed=1, cache_dir=str(tmp_path / "k"))
    from collections import Counter
    assert Counter(r[2] for r in tr0) == {0: 12, 1: 12, 2: 12}           # stratified 80/20
    assert Counter(r[2] for r in te0) == {0: 3, 1: 3, 2: 3}
    assert {r[1] for r in te0} != {r[1] for r in te1}                     # split seed matters
    assert S.collect("lsa64_rand", str(roots), seed=0, cache_dir=str(tmp_path / "k"))[1] == te0
    tr, te, cls, _ = S.collect("asl_citizen", str(roots), cache_dir=str(tmp_path / "k"))
    assert cls == ["cat", "dog"] and len(tr) == 2
    assert [r[1] for r in te] == ["asl_citizen__002-CAT"]                 # val unused, EMU unseen


def test_label_subset_random_20pct():
    items = [(0, f"c{i:03d}", i % 7) for i in range(100)]
    a = S.label_subset(items, 20, 0)
    assert len(a) == 20 and a == S.label_subset(items, 20, 0) and a != S.label_subset(items, 20, 1)
    assert S.label_subset(items, 100, 5) == items


def test_model_forward_and_masking(tmp_path):
    m = S.StdModel(tiny_ckpt(tmp_path), n_cls=4).eval()
    x = torch.randn(3, 32, 77 * 4)
    assert m(x, torch.full((3,), 32)).shape == (3, 4)
    assert all(p.requires_grad for p in m.parameters())
    assert not m.body.grad_checkpoint


def test_run_end_to_end_and_metrics(roots, tmp_path, monkeypatch):
    monkeypatch.setenv("FS_RESULTS_DIR", str(tmp_path / "res"))
    out = S.run(tiny_ckpt(tmp_path, clip_len=64), "t", "lsa64_rand", 100, 0, device="cpu",
                epochs=2, root=str(roots), cache_dir=str(tmp_path / "k"))
    assert out["complete"] and out["n_train"] == 36 and out["n_test"] == 9 and out["n_cls"] == 3
    for k in ("pi_top1", "pi_top5", "pc_top1", "pc_top5"):
        assert 0.0 <= out[k] <= 1.0
    assert out["pi_top5"] == 1.0                                          # 3 classes < 5
    assert (tmp_path / "res" / "std_t_lsa64_rand_100pct_s0.json").exists()


def test_per_class_metric():
    class Fixed(torch.nn.Module):
        compiled, in_dim = False, 3

        def forward(self, x, nv):                     # always predicts class 0
            return torch.tensor([[5.0, 0.0, -1.0]]).repeat(x.size(0), 1)
    clips = [(np.zeros((8, 77, 3), np.float16), np.ones((8, 77), np.uint8))] * 4
    y = np.array([0, 0, 0, 1])
    met = S.evaluate(Fixed(), clips, y, 3, "cpu")
    assert met["pi_top1"] == pytest.approx(0.75) and met["pc_top1"] == pytest.approx(0.5)


def test_sota_launcher_jobs(tmp_path, monkeypatch):
    import json as _j
    from foresign.build import build_model
    from tests.test_canon_evals import TINY
    import tools.run_sota_p2 as L
    import tools.run_queue as rq
    cfg = {**TINY, "total_steps": 100}
    run = tmp_path / "p2_pred_long_s0"; run.mkdir()
    torch.save({"model": build_model(cfg).state_dict(), "step": 100, "cfg": cfg}, run / "last.pt")
    rnd = tmp_path / "p2_random_init_s0"; rnd.mkdir()
    torch.save({"model": build_model(cfg).state_dict(), "step": 0,
                "cfg": {**cfg, "random_init": True}}, rnd / "last.pt")
    ck = [str(rnd / "last.pt"), str(run / "last.pt"), str(tmp_path / "missing.pt")]
    js, probs = L.jobs(ck, ["islr", "cslr"])
    n_islr = (1 + 3 + 1 + 1 + 3 + 1) * 2
    assert len(js) == n_islr + 2 * 2 and len(probs) == 1
    assert js[0]["name"] == "std_p2_random_init_s0_k0_L0_wlasl_100pct_s0"
    assert js[-1]["name"] == "cosign_p2_pred_long_s0_k100_L0_ph14_s0"
    mod, env = L.job_env(js[0], True)
    assert mod == "evals.islr_std" and env["FS_STD_COMPILE"] == "1" and env["FS_STD_BUDGET"] == "100"
    mod, env = L.job_env(js[-1], True)
    assert mod == "evals.cslr_cosign" and env["FS_CC_DS"] == "ph14"
    res = tmp_path / "res"; res.mkdir()
    monkeypatch.setenv("FS_RESULTS_DIR", str(res))
    (res / (js[0]["name"] + ".json")).write_text(_j.dumps({"complete": True}))
    calls = []
    monkeypatch.setattr(rq, "_run", lambda m, e, l, a=(): calls.append(e["FS_TAG"]) or 0)
    L.main(["--ckpts", ",".join(ck), "--protocols", "islr", "--datasets", "wlasl"])
    assert len(calls) == (1 + 3) * 2 - 1
