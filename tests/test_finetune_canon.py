import json
import os
from collections import Counter

import numpy as np
import pytest
import torch

from tests.test_canon_data import write_canon
from tests.test_canon_evals import tiny_ckpt
import evals.finetune_islr_canon as ft


def _items(n_cls=5, per=(1, 3, 10, 20, 7)):
    return [(0, f"c{c}_{i:03d}", c) for c in range(n_cls) for i in range(per[c])]


def test_stratified_subset_deterministic_and_stratified():
    items = _items()
    a = ft.stratified_subset(items, 10, 0)
    assert a == ft.stratified_subset(list(reversed(items)), 10, 0)        # order-independent
    cnt = Counter(it[2] for it in a)
    assert cnt == {0: 1, 1: 1, 2: 1, 3: 2, 4: 1}                         # max(1, round(0.1 n))
    many = _items(per=(100, 100, 100, 100, 100))
    assert ft.stratified_subset(many, 10, 0) != ft.stratified_subset(many, 10, 1)
    assert ft.stratified_subset(items, 100, 3) == items


def test_param_groups_llrd_and_all_trainable(tmp_path):
    m = ft.FTModel(tiny_ckpt(tmp_path), n_cls=7)
    assert all(p.requires_grad for p in m.parameters())
    groups = ft.param_groups(m)
    n_grouped = sum(len(g["params"]) for g in groups)
    assert n_grouped == len(list(m.parameters()))
    lr = {g["name"]: g["lr"] for g in groups}
    D = m.depth                                                           # 2
    assert lr["head-1"] == pytest.approx(1e-3)
    assert lr["enc0"] == pytest.approx(5e-4 * 0.75 ** (D + 1))           # tokenizer
    assert lr["enc1"] == pytest.approx(5e-4 * 0.75 ** D)                 # first layer
    assert lr["enc3_nd"] == pytest.approx(5e-4)                           # norm_final
    for g in groups:
        assert g["weight_decay"] == (0.0 if g["name"].endswith("_nd") else 0.05)


def test_forward_shapes_and_tap(tmp_path):
    ck = tiny_ckpt(tmp_path)
    m = ft.FTModel(ck, n_cls=7)
    x = torch.randn(3, 32, 77 * 4)
    assert m(x, torch.tensor([32, 10, 1])).shape == (3, 7)
    m1 = ft.FTModel(ck, n_cls=7, tap_depth=1)
    assert m1.tap == 1 and m1(x, torch.tensor([32, 10, 1])).shape == (3, 7)
    # gradients reach the tokenizer (full fine-tuning)
    m(x, torch.tensor([32, 10, 1])).sum().backward()
    assert m.tok.proj[0].weight.grad is not None


def test_pack_batch_crops():
    rng0 = np.random.default_rng(0)
    clips = [(np.arange(100 * 231, dtype=np.float16).reshape(100, 77, 3) % 7,
              np.ones((100, 77), np.uint8)),
             (np.ones((10, 77, 3), np.float16), np.ones((10, 77), np.uint8))]
    x, nv = ft.pack_batch(clips, [0, 1], 32, 3)
    assert x.shape == (2, 32, 308) and nv.tolist() == [32, 10]
    assert torch.all(x[1, 10:] == 0)
    x2, _ = ft.pack_batch(clips, [0, 1], 32, 3)
    assert torch.equal(x, x2)                                             # centre crop fixed
    a, _ = ft.pack_batch(clips, [0], 32, 3, np.random.default_rng(0))
    b, _ = ft.pack_batch(clips, [0], 32, 3, np.random.default_rng(0))
    assert torch.equal(a, b)


@pytest.fixture
def lsa_root(tmp_path):
    root = tmp_path / "c"
    root.mkdir()
    write_canon(root / "lsa64.h5", {f"lsa64__00{c}_00{s}_00{r}": (40, "")
                                    for c in (1, 2, 3) for s in (1, 2, 9) for r in (1, 2)})
    return root


def test_run_writes_result_and_resumes(tmp_path, lsa_root, monkeypatch):
    monkeypatch.setenv("FS_RESULTS_DIR", str(tmp_path / "res"))
    ck = tiny_ckpt(tmp_path)
    out = ft.run(ck, "t", "lsa64", 100, 0, device="cpu", epochs=2, root=str(lsa_root),
                 cache_dir=str(tmp_path / "cache"))
    assert out["complete"] and out["n_train"] == 12 and out["n_test"] == 6 and out["n_cls"] == 3
    assert 0 <= out["top1"] <= out["top5"] <= 1
    assert (tmp_path / "res" / "ft_t_lsa64_100pct_s0.json").exists()
    assert not (tmp_path / "res" / "ft_state" / "ft_t_lsa64_100pct_s0.pt").exists()
    # resume: a stored epoch-2 state of a 3-epoch run -> only one more epoch is trained
    state = tmp_path / "res" / "ft_state"
    state.mkdir(exist_ok=True)
    m = ft.FTModel(ck, 3)
    opt = torch.optim.AdamW(ft.param_groups(m))
    torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "epoch": 2},
               state / "ft_t2_lsa64_100pct_s0.pt")
    out2 = ft.run(ck, "t2", "lsa64", 100, 0, device="cpu", epochs=3, root=str(lsa_root),
                  cache_dir=str(tmp_path / "cache"))
    assert out2["steps"] == 3 * 1                                         # 12 clips = 1 step/epoch


def test_launcher_jobs_skip_and_unfinished(tmp_path, monkeypatch):
    from foresign.build import build_model
    import tools.run_finetune_p2 as L
    monkeypatch.setenv("FS_RESULTS_DIR", str(tmp_path / "res"))
    cfg = {**json.loads(json.dumps(__import__("tests.test_canon_evals").test_canon_evals.TINY)),
           "total_steps": 100}
    run = tmp_path / "p2_ms_s0"
    run.mkdir()
    sd = build_model(cfg).state_dict()
    torch.save({"model": sd, "step": 20, "cfg": cfg}, run / "step_20.pt")
    torch.save({"model": sd, "step": 50, "cfg": cfg}, run / "last.pt")      # still training
    rnd = tmp_path / "p2_random_init_s0"
    rnd.mkdir()
    torch.save({"model": sd, "step": 0, "cfg": {**cfg, "random_init": True}}, rnd / "last.pt")
    ckpts = [str(rnd / "last.pt"), str(run / "step_20.pt"), str(run / "last.pt"),
             str(tmp_path / "nope.pt")]
    js, probs = L.jobs(ckpts, ["autsl", "wlasl"], [10, 100])
    assert len(js) == 2 * (3 + 1) * 2 and len(probs) == 2
    assert any("unfinished" in p for p in probs) and any("missing" in p for p in probs)
    assert js[0]["name"] == "ft_p2_random_init_s0_k0_L0_autsl_10pct_s0"
    assert js[1]["name"] == "ft_p2_ms_s0_k20_L0_autsl_10pct_s0"
    res = tmp_path / "res"
    res.mkdir()
    (res / (js[0]["name"] + ".json")).write_text(json.dumps({"complete": True}))
    (res / (js[1]["name"] + ".json")).write_text(json.dumps({"complete": False}))
    assert L.is_done(js[0]["name"]) and not L.is_done(js[1]["name"])
    calls = []
    import tools.run_queue as rq
    monkeypatch.setattr(rq, "_run", lambda m, env, log, args=(): calls.append(env["FS_TAG"]) or 0)
    L.main(["--ckpts", ",".join(ckpts), "--datasets", "autsl", "--budgets", "10"])
    assert len(calls) == 2 * 3 - 1                                       # one already done


def test_ft_fast_ops_and_ckpt_layers_same_output(tmp_path, monkeypatch):
    ck = tiny_ckpt(tmp_path)
    x = torch.randn(3, 32, 77 * 4)
    nv = torch.tensor([32, 10, 1])
    monkeypatch.setenv("FS_FAST_OPS", "0")
    torch.manual_seed(0)
    base = ft.FTModel(ck, n_cls=7).eval()
    monkeypatch.delenv("FS_FAST_OPS")                    # c77 default: fast kernels on
    monkeypatch.setenv("FS_FT_CKPT_LAYERS", "1")
    torch.manual_seed(0)
    fast = ft.FTModel(ck, n_cls=7).eval()
    assert fast.fast and not base.fast and fast.body.ckpt_layers == 1
    assert torch.allclose(base(x, nv), fast(x, nv), atol=1e-5)
