import os

import yaml

import tools.run_queue as rq
from tools.make_configs import BASE, p2_specs, write_configs


def test_p2_specs_match_phase1a_constants():
    specs = {s["name"]: s for s in p2_specs()}
    assert set(specs) == {"p2_random_init_s0", "p2_pred_s0", "p2_nce_s0", "p2_ms_s0"}
    for s in specs.values():
        for k in ("d_model", "depth", "batch_size", "lr", "total_steps", "masking", "ckpt_every"):
            assert s[k] == BASE[k]
        assert s["t_patch"] == 2 and s["clip_len"] == 128 and s["seed"] == 0
        assert s["input_format"] == "c77" and s["in_dim"] == 3 and s["drop_conf"] is False
        assert s["sample_unit"] == "windows"
    assert specs["p2_random_init_s0"]["random_init"] is True
    assert specs["p2_ms_s0"]["beta"] == 0.3 and specs["p2_ms_s0"]["lambda_bound"] == 0.2


def test_p2_written_yaml_is_portable(tmp_path):
    ps = write_configs(p2_specs(), tmp_path, run_dir="runs/p2")
    cfg = yaml.safe_load(open(ps[0], encoding="utf-8"))
    assert cfg["output_dir"].startswith("runs/p2/")
    assert cfg["canon_root"] is None            # supplied by FS_CANON_ROOT


def test_relative_output_dir_resolves_to_project(monkeypatch):
    from foresign import paths
    assert paths.resolve_out("runs/p2/x") == os.path.join(paths.PROJECT, "runs", "p2", "x")
    assert paths.resolve_out("/abs/run") == "/abs/run"


def test_p2_eval_jobs(tmp_path):
    jobs = rq.eval_jobs("p2_ms_s0", {"step_2000.pt": 2000, "last.pt": 16000}, str(tmp_path),
                        canon=True)
    assert len(jobs) == 2 * 2 * 3
    kinds = {j["kind"] for j in jobs}
    assert kinds == {"islr_canon", "cslr", "dsid"}
    by = {(j["kind"], j["tag"]): j["env"] for j in jobs}
    e = by[("islr_canon", "p2_ms_s0_k16000_L1")]
    assert e["FS_DATASETS"] == "autsl,wlasl,lsa64" and e["FS_SEEDS"] == "0,1,2"
    assert e["FS_ADAPTER"].startswith("foresign:") and e["FS_TAP_DEPTH"] == "1"
    c = by[("cslr", "p2_ms_s0_k2000_L0")]
    assert c["FS_CSLR_DATASET"] == "canon_ph14t" and c["FS_CSLR_FREEZE"] == "1"
    assert c["FS_CSLR_EPOCHS"] == "40"
    d = by[("dsid", "p2_ms_s0_k2000_L0")]
    assert d["FS_DSID_CKPT"].endswith("step_2000.pt")
    assert rq.result_path("islr_canon", "t").endswith("islr_canon_t.json")
    assert rq.result_path("dsid", "t").endswith("dsid_t.json")
    assert rq.MODULES == {"islr_canon": "evals.islr_canon",
                          "cslr": "evals.cslr_phoenix", "dsid": "evals.dsid_probe"}


def test_p2_extra_ph14(monkeypatch, tmp_path):
    monkeypatch.setenv("FS_QUEUE_EXTRA", "cslr_ph14")
    jobs = rq.eval_jobs("n", {"last.pt": 16000}, str(tmp_path), canon=True)
    assert {j["kind"] for j in jobs} == {"cslr"}
    assert all(j["env"]["FS_CSLR_DATASET"] == "canon_ph14" and j["tag"].endswith("_ph14")
               for j in jobs)


def test_p2_smoke_jobs(monkeypatch, tmp_path):
    monkeypatch.setenv("FS_QUEUE_SMOKE", "1")
    jobs = rq.eval_jobs("n", {"last.pt": 60}, str(tmp_path), canon=True)
    env = {j["kind"]: j["env"] for j in jobs if j["tag"].endswith("L0")}
    assert env["islr_canon"]["FS_SEEDS"] == "0" and env["islr_canon"]["FS_ISLR_LIMIT"] == "300"
    assert env["cslr"]["FS_CSLR_EPOCHS"] == "1" and env["cslr"]["FS_CSLR_LIMIT"] == "64"
    assert env["dsid"]["FS_DSID_N"] == "30"


def test_child_env_passes_canon_root(monkeypatch):
    monkeypatch.setenv("FS_CANON_ROOT", "/x/canon")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    env = rq.child_env({})
    assert env["FS_CANON_ROOT"] == "/x/canon" and env["CUDA_VISIBLE_DEVICES"] == "3"


def _p2_cfg(tmp_path, **over):
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    cfg = {"output_dir": str(run), "total_steps": 16000, "input_format": "c77", "in_dim": 3,
           "seed": 0, **over}
    p = tmp_path / "p2_x.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(p), run


def test_random_init_writes_ckpt_instead_of_training(monkeypatch, tmp_path):
    monkeypatch.setenv("FS_RESULTS_DIR", str(tmp_path / "res"))
    cfgp, run = _p2_cfg(tmp_path, random_init=True)
    calls = []

    def fake(module, env, log, args=()):
        calls.append((module, args))
        if module == "tools.make_random_ckpt":
            open(run / "last.pt", "w").write("x")
        return 0
    monkeypatch.setattr(rq, "_run", fake)
    monkeypatch.setattr(rq, "_ckpt_step", lambda p: 0)
    assert rq.process(cfgp) == []
    mods = [m for m, _ in calls]
    assert mods[0] == "tools.make_random_ckpt" and "foresign.train" not in mods
    assert "--seed" in calls[0][1] and calls[0][1][calls[0][1].index("--seed") + 1] == "0"
    assert {"evals.islr_canon", "evals.cslr_phoenix", "evals.dsid_probe"} <= set(mods)
    tags = {a for m, a in calls}
    assert len([m for m in mods if m == "evals.dsid_probe"]) == 2      # k0 x L0/L1


def test_p2_trains_then_evaluates(monkeypatch, tmp_path):
    monkeypatch.setenv("FS_RESULTS_DIR", str(tmp_path / "res"))
    monkeypatch.setenv("FS_QUEUE_CKPTS", "last.pt")
    cfgp, run = _p2_cfg(tmp_path)
    calls = []

    def fake(module, env, log, args=()):
        calls.append(module)
        if module == "foresign.train":
            open(run / "last.pt", "w").write("x")
        return 0
    monkeypatch.setattr(rq, "_run", fake)
    monkeypatch.setattr(rq, "_ckpt_step", lambda p: 16000)
    assert rq.process(cfgp) == []
    assert calls[0] == "foresign.train" and calls.count("evals.islr_canon") == 2
