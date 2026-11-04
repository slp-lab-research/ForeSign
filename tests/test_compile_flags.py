import json
import sys

import numpy as np
import pytest
import torch
import yaml

from foresign.build import build_model, compile_flag, compile_jepa
from tests.test_canon_evals import tiny_ckpt

CFG = dict(d_model=24, depth=2, n_heads=2, pred_dim=12, pred_depth=1, pred_heads=2,
           t_patch=2, ema_base=0.99, input_format="c77", in_dim=3)


def _has_triton():
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


def test_compile_flag_precedence(monkeypatch):
    monkeypatch.delenv("FS_COMPILE", raising=False)
    assert compile_flag({}) is False and compile_flag({"compile": True}) is True
    monkeypatch.setenv("FS_COMPILE", "0")
    assert compile_flag({"compile": True}) is False
    monkeypatch.setenv("FS_COMPILE", "1")
    assert compile_flag({}) is True
    monkeypatch.setenv("FS_FT_COMPILE", "1")
    assert compile_flag({}, "FS_FT_COMPILE") is True


def test_compile_in_place_keeps_state_dict_keys():
    m = build_model(CFG)
    keys = list(m.state_dict())
    done = compile_jepa(m)                          # lazy: nothing compiles until forward
    assert done == ["tokenizer", "context", "target_tok", "target_ctx", "predictor"]
    assert list(m.state_dict()) == keys
    assert all(k.find("_orig_mod") < 0 for k in m.state_dict())


@pytest.mark.skipif(not _has_triton() or not torch.cuda.is_available(), reason="needs triton+CUDA")
def test_compiled_forward_matches_eager():
    torch.manual_seed(0)
    a = build_model(CFG).cuda().eval()
    b = build_model(CFG).cuda().eval()
    b.load_state_dict(a.state_dict())
    compile_jepa(b)
    x = torch.randn(2, 16, 77 * 4, device="cuda")
    mask = torch.zeros(2, 8, 18, dtype=torch.bool, device="cuda"); mask[:, :4] = True
    valid = torch.ones(2, 8, 18, dtype=torch.bool, device="cuda")
    with torch.no_grad():
        assert torch.allclose(a(x, mask, valid)[0], b(x, mask, valid)[0], atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="train loop is CUDA-only")
def test_train_compile_flag_plumbing(tmp_path, monkeypatch):
    """FS_COMPILE=1 -> the five pretraining modules are compiled (mocked here), provenance
    records it, checkpoint keys are the uncompiled ones."""
    import foresign.canon_splits as cs
    from foresign import paths, train
    from tests.test_train_c77 import TINY, tiny_canon
    calls = []
    monkeypatch.setattr(torch.nn.Module, "compile",
                        lambda self, *a, **k: calls.append((type(self).__name__, k)))
    root, ex = tiny_canon(tmp_path)
    monkeypatch.setenv("FS_CANON_ROOT", str(root))
    monkeypatch.setenv("FS_COMPILE", "1")
    monkeypatch.setattr(paths, "CANON_EXCLUDE", str(ex))
    monkeypatch.setattr(paths, "CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(cs, "DEFAULT_PRETRAIN", ["autsl", "youtube_asl"])
    cfg = {**TINY, "total_steps": 3, "output_dir": str(tmp_path / "run")}
    cp = tmp_path / "c.yaml"
    cp.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(cp)])
    train.main()
    assert len(calls) == 5 and all(k == {"mode": "default", "dynamic": False} for _, k in calls)
    ck = torch.load(tmp_path / "run" / "last.pt", weights_only=False)
    assert ck["provenance"]["compile"] is True
    assert set(ck["model"]) == set(build_model(cfg).state_dict())


def test_pack_batch_pad_to():
    import evals.finetune_islr_canon as ft
    clips = [(np.ones((10, 77, 3), np.float16), np.ones((10, 77), np.uint8))] * 3
    x, nv = ft.pack_batch(clips, [0, 1, 2], 16, 3, pad_to=8)
    assert x.shape == (8, 16, 308) and nv.tolist() == [10, 10, 10, 1, 1, 1, 1, 1]
    assert torch.all(x[3:] == 0)
    x2, _ = ft.pack_batch(clips, [0, 1, 2], 16, 3)
    assert torch.equal(x[:3], x2)


def test_padded_rows_do_not_change_real_outputs(tmp_path):
    import evals.finetune_islr_canon as ft
    m = ft.FTModel(tiny_ckpt(tmp_path), n_cls=5).eval()
    rng = np.random.default_rng(0)
    clips = [(rng.normal(size=(20, 77, 3)).astype(np.float16), np.ones((20, 77), np.uint8))
             for _ in range(3)]
    x, nv = ft.pack_batch(clips, [0, 1, 2], 32, 3)
    xp, nvp = ft.pack_batch(clips, [0, 1, 2], 32, 3, pad_to=8)
    with torch.no_grad():
        assert torch.allclose(m(x, nv), m(xp, nvp)[:3], atol=1e-5)


def test_ft_compile_flag_plumbing(tmp_path, monkeypatch):
    """FS_FT_COMPILE=1 -> FTModel compiled (mocked), padded fixed-size batches, JSON records it."""
    import evals.finetune_islr_canon as ft
    from tests.test_canon_data import write_canon
    root = tmp_path / "c"; root.mkdir()
    write_canon(root / "lsa64.h5", {f"lsa64__00{c}_00{s}_00{r}": (40, "")
                                    for c in (1, 2, 3) for s in (1, 2, 9) for r in (1, 2)})
    calls, shapes = [], []
    monkeypatch.setattr(torch.nn.Module, "compile",
                        lambda self, *a, **k: calls.append((type(self).__name__, k)))
    orig = ft.FTModel.forward
    monkeypatch.setattr(ft.FTModel, "forward",
                        lambda self, x, nv: shapes.append(x.shape[0]) or orig(self, x, nv))
    monkeypatch.setenv("FS_RESULTS_DIR", str(tmp_path / "res"))
    monkeypatch.setenv("FS_FT_COMPILE", "1")
    out = ft.run(tiny_ckpt(tmp_path), "t", "lsa64", 100, 0, device="cpu", epochs=1,
                 root=str(root), cache_dir=str(tmp_path / "cache"))
    assert calls == [("FTModel", {"mode": "default", "dynamic": False})]
    assert out["compile"] is True and set(shapes) == {ft.BATCH, 128}   # train 64, eval 128
    monkeypatch.setenv("FS_FT_COMPILE", "0")
    shapes.clear()
    out0 = ft.run(tiny_ckpt(tmp_path), "t0", "lsa64", 100, 0, device="cpu", epochs=1,
                  root=str(root), cache_dir=str(tmp_path / "cache"))
    assert out0["compile"] is False and set(shapes) == {12, 6}         # unpadded


def test_queue_taps_env(monkeypatch, tmp_path):
    import tools.run_queue as rq
    monkeypatch.delenv("FS_QUEUE_TAPS", raising=False)
    assert rq.taps() == (0, 1)
    monkeypatch.setenv("FS_QUEUE_TAPS", "0")
    jobs = rq.eval_jobs("n", {"last.pt": 5}, str(tmp_path), canon=True)
    assert {j["tag"] for j in jobs} == {"n_k5_L0"} and len(jobs) == 3
    old = rq.eval_jobs("n", {"last.pt": 5}, str(tmp_path))
    assert {j["tag"] for j in old} == {"n_k5_L0"}
    monkeypatch.setenv("FS_QUEUE_TAPS", "1,3")
    assert rq.taps() == (1, 3)
    env = rq.child_env({})
    assert "FS_QUEUE_TAPS" not in env


def test_equiv_log_parsing(tmp_path):
    from tools.equiv_compile import compare_train, parse_train_log, rate_after
    lines = []
    for step, rate in ((0, 0.10), (10, 0.50), (20, 0.70), (30, 0.80), (40, 0.85)):
        lines.append(f"step {step:6d} | loss {1.0 - step / 100:.4f} | l_pred 0.1 l_nce 0.0 "
                     f"l_bound 0.0 | eff_rank 120.5 feat_std 0.9 offdiag 0.1 pred_var 0.2 | "
                     f"lr 1.00e-04 | {rate:.2f} it/s")
    p = tmp_path / "a.log"
    p.write_text("\n".join(lines), encoding="utf-8")
    rows = parse_train_log(str(p))
    assert sorted(rows) == [0, 10, 20, 30, 40] and rows[10] == (0.9, 120.5, 0.5)
    t20, t40 = 21 / 0.70, 41 / 0.85
    assert rate_after(rows, 20) == pytest.approx(20 / (t40 - t20))
    common, dl, de = compare_train(rows, {s: (v[0] + 0.001, v[1], v[2]) for s, v in rows.items()})
    assert len(common) == 5 and max(dl) == pytest.approx(0.001) and max(de) == 0
