from foresign.inductive import provenance


def test_provenance_fields_c77(monkeypatch, tmp_path):
    monkeypatch.setenv("FS_INDUCTIVE", "1")
    monkeypatch.setenv("FS_CANON_ROOT", str(tmp_path))
    p = provenance({"input_format": "c77", "in_dim": 3})
    assert p["inductive"] is True and p["drop_conf"] is False
    assert p["input_format"] == "c77" and p["in_dim"] == 3
    assert p["canon_root"] == str(tmp_path) and p["exclude_file"] == "canon_exclude.json"
    assert len(p["dup_sha256"]) == 64          # data_ann/canon_exclude.json ships with the code


def test_provenance_missing_list_has_no_hash(tmp_path):
    p = provenance({"input_format": "c77"}, dup_path=str(tmp_path / "missing.json"))
    assert p["dup_sha256"] is None
