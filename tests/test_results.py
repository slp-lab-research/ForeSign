import json

from foresign.results import write_result


def test_write_result_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("FS_RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("FS_SEED", "3")
    p = write_result("cslr_demo", {"dev_wer": 0.5})
    rec = json.loads(p.read_text(encoding="utf-8"))
    assert rec["name"] == "cslr_demo" and rec["dev_wer"] == 0.5
    assert rec["env"]["FS_SEED"] == "3" and "time" in rec


def test_nan_written_as_null(tmp_path, monkeypatch):
    monkeypatch.setenv("FS_RESULTS_DIR", str(tmp_path))
    p = write_result("nan_demo", {"sd": float("nan"), "rows": [{"x": float("inf")}]})
    rec = json.loads(p.read_text(encoding="utf-8"))
    assert rec["sd"] is None and rec["rows"][0]["x"] is None
