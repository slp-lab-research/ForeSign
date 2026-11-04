import pytest

import evals.cslr_phoenix as cp
from foresign import paths


def test_default_dataset_is_canon_ph14t(monkeypatch):
    monkeypatch.delenv("FS_CSLR_DATASET", raising=False)
    assert cp.dataset_name() == "canon_ph14t"
    assert cp.result_tag("abc") == "abc"
    assert cp.ann_path("dev", "canon_ph14t").endswith("phoenix14t_dev.json")


def test_unknown_dataset_rejected(monkeypatch):
    for bad in ("nope", "ph14t", "csl_daily"):
        monkeypatch.setenv("FS_CSLR_DATASET", bad)
        with pytest.raises(ValueError):
            cp.dataset_name()


def test_ph14_paths_and_tags(monkeypatch, tmp_path):
    monkeypatch.setenv("FS_CANON_ROOT", str(tmp_path))
    assert cp.ann_path("train", "canon_ph14").endswith("phoenix14_train.json")
    assert cp.h5_path("canon_ph14") == str(tmp_path / "phoenix14.h5")
    assert cp.h5_path("canon_ph14t") == str(tmp_path / "phoenix14t.h5")
    assert cp.result_tag("t", "canon_ph14") == "t_ph14"
    assert cp.result_tag("t_ph14", "canon_ph14") == "t_ph14"      # queue pre-suffixed tag


def test_ann_dir_override(monkeypatch, tmp_path):
    (tmp_path / "phoenix14_dev.json").write_text("[]", encoding="utf-8")
    monkeypatch.setenv("FS_ANN_DIR", str(tmp_path))
    assert cp.ann_path("dev", "canon_ph14") == str(tmp_path / "phoenix14_dev.json")


def test_ann_falls_back_to_project_data_ann(monkeypatch, tmp_path):
    monkeypatch.delenv("FS_ANN_DIR", raising=False)
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "phoenix14_dev.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(paths, "ANN_DIRS", (str(proj),))
    assert cp.ann_path("dev", "canon_ph14") == str(proj / "phoenix14_dev.json")
