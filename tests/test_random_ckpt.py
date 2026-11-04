import os
import yaml
import torch

from tools import make_random_ckpt as m

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "p2", "p2_random_init_s0.yaml")


def test_random_ckpt_loads(tmp_path):
    out = str(tmp_path / "r.pt")
    small = yaml.safe_load(open(CFG, encoding="utf-8"))
    small.update(d_model=32, depth=2, n_heads=2, pred_dim=16, pred_depth=1, pred_heads=2)
    cp = str(tmp_path / "c.yaml")
    yaml.safe_dump(small, open(cp, "w"))
    m.main(["--config", cp, "--out", out, "--seed", "0"])
    ck = torch.load(out, map_location="cpu", weights_only=False)
    assert ck["step"] == 0 and "model" in ck and ck["cfg"]["d_model"] == 32
    from foresign.adapters import ForeSignAdapter
    ad = ForeSignAdapter(out, device="cpu")
    assert ad.clip_len == small["clip_len"]
    from evals.cslr_phoenix import WindowEncoder
    enc = WindowEncoder(out, freeze=True)
    assert not any(p.requires_grad for p in enc.parameters())


def test_seed_deterministic(tmp_path):
    cfgp = str(tmp_path / "c.yaml")
    small = yaml.safe_load(open(CFG, encoding="utf-8"))
    small.update(d_model=32, depth=2, n_heads=2, pred_dim=16, pred_depth=1, pred_heads=2)
    yaml.safe_dump(small, open(cfgp, "w"))
    a, b = str(tmp_path / "a.pt"), str(tmp_path / "b.pt")
    m.main(["--config", cfgp, "--out", a, "--seed", "5"])
    m.main(["--config", cfgp, "--out", b, "--seed", "5"])
    sa = torch.load(a, weights_only=False)["model"]; sb = torch.load(b, weights_only=False)["model"]
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
