import json
import sys

import pytest
import torch
import yaml

from tests.test_canon_data import write_canon

TINY = dict(d_model=24, depth=2, n_heads=2, pred_dim=12, pred_depth=1, pred_heads=2,
            ema_base=0.99, masking="rand78", drop_conf=False, group_layout="anatomical",
            batch_size=4, lr=1e-4, weight_decay=0.05, grad_clip=1.0, warmup_steps=2,
            total_steps=6, const_until=4, temperature=0.5, num_workers=0, resume=True,
            collapse_std=0.0, collapse_patience=300, log_every=1, ckpt_every=3, datasets=None,
            tau=0.1, seed=0, t_patch=2, clip_len=32, input_format="c77", in_dim=3,
            sample_unit="windows")


def tiny_canon(tmp_path):
    root = tmp_path / "canon"
    root.mkdir()
    write_canon(root / "autsl.h5", {f"autsl__train__s{i}": (40 + i, "train") for i in range(6)})
    write_canon(root / "youtube_asl_b00001.h5", {"youtube_asl__v1": (300, "")})
    ex = tmp_path / "ex.json"
    ex.write_text(json.dumps({"exclude": {}}), encoding="utf-8")
    return root, ex


def test_provenance_c77(monkeypatch):
    from foresign.inductive import provenance
    monkeypatch.setenv("FS_CANON_ROOT", "/x/canon_v4")
    p = provenance({"input_format": "c77", "in_dim": 3})
    assert p["input_format"] == "c77" and p["canon_root"] == "/x/canon_v4"
    assert p["exclude_file"] == "canon_exclude.json" and len(p["dup_sha256"]) == 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="train loop is CUDA-only")
@pytest.mark.parametrize("obj", [{"alpha_pred": 1.0, "beta": 0.3, "lambda_bound": 0.2},
                                 {"alpha_pred": 0.0, "beta": 1.0, "lambda_bound": 0.0}])
def test_train_c77_end_to_end(tmp_path, monkeypatch, obj):
    import foresign.canon_splits as cs
    from foresign import paths, train
    root, ex = tiny_canon(tmp_path)
    monkeypatch.setenv("FS_CANON_ROOT", str(root))
    monkeypatch.setattr(paths, "CANON_EXCLUDE", str(ex))
    monkeypatch.setattr(paths, "CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(cs, "DEFAULT_PRETRAIN", ["autsl", "youtube_asl"])
    cfg = {**TINY, **obj, "output_dir": str(tmp_path / "run")}
    cp = tmp_path / "c.yaml"
    cp.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(cp)])
    train.main()
    ck = torch.load(tmp_path / "run" / "last.pt", weights_only=False)
    assert ck["step"] == 6 and ck["provenance"]["input_format"] == "c77"
    assert ck["provenance"]["canon_datasets"] == ["autsl", "youtube_asl"]
    assert ck["model"]["tokenizer.proj.0.weight"].shape[1] == 3 * 2 * 3
    assert (tmp_path / "run" / "step_3.pt").exists()
