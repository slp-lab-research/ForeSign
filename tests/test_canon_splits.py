import gzip
import json

import pytest

from foresign.canon_splits import (CANON_POOL, DEFAULT_PRETRAIN, EVAL_DATASETS, canon_split,
                                   filter_canon_clips, load_exclude)


def test_attr_splits_for_eval_datasets():
    assert canon_split("autsl", "autsl__train__signer0_sample1_color", "train") == "train"
    assert canon_split("autsl", "autsl__val__x", "val") == "val"
    assert canon_split("wlasl", "wlasl__test__1-wlasl-a", "test") == "test"
    assert canon_split("phoenix14t", "phoenix14t__dev__a", "dev") == "val"
    assert canon_split("how2sign", "how2sign__val__a", "val") == "val"
    # missing / unknown split attr on an eval dataset fails closed
    assert canon_split("how2sign", "how2sign__x__a", "") == "excluded"
    assert canon_split("phoenix14", "phoenix14__x__a", "weird") == "excluded"


def test_lsa64_signer_rule():
    assert canon_split("lsa64", "lsa64__001_008_001", "") == "train"
    assert canon_split("lsa64", "lsa64__001_009_001", "") == "test"
    assert canon_split("lsa64", "lsa64__001_010_005", "") == "test"
    assert canon_split("lsa64", "lsa64__garbage", "") == "excluded"


def test_pool_and_unknown():
    assert canon_split("youtube_asl", "youtube_asl__abc", "") == "train"
    assert canon_split("isign", "isign__test_mp4__x", "test") == "train"     # not an eval set
    assert canon_split("autsl", "wlasl__train__x", "train") == "excluded"    # id/dataset mismatch
    with pytest.raises(ValueError):
        canon_split("csl_daily", "csl_daily__x", "train")
    assert "phoenix14" in CANON_POOL and "phoenix14" not in DEFAULT_PRETRAIN
    assert {"autsl", "wlasl", "lsa64", "phoenix14t", "phoenix14", "how2sign"} <= EVAL_DATASETS


def _clips():
    return {"autsl": [(0, "autsl__train__a", 50, "train"), (0, "autsl__test__b", 50, "test")],
            "phoenix14t": [(0, "phoenix14t__train__p", 80, "train"),
                           (0, "phoenix14t__train__q", 80, "train"),
                           (0, "phoenix14t__dev__r", 80, "val")],
            "youtube_asl": [(0, "youtube_asl__y", 900, ""), (1, "youtube_asl__z", 900, "")]}


def test_filter_canon_clips(tmp_path):
    ex = tmp_path / "ex.json"
    ex.write_text(json.dumps({"exclude": {"phoenix14t": ["phoenix14t__train__q"]}}), encoding="utf-8")
    out = filter_canon_clips(_clips(), load_exclude(str(ex)))
    assert [c[1] for c in out["autsl"]] == ["autsl__train__a"]
    assert [c[1] for c in out["phoenix14t"]] == ["phoenix14t__train__p"]
    assert len(out["youtube_asl"]) == 2


def test_missing_exclude_fails_closed(tmp_path, monkeypatch):
    monkeypatch.delenv("FS_ALLOW_NO_DUP", raising=False)
    with pytest.raises(FileNotFoundError):
        load_exclude(str(tmp_path / "missing.json"))
    monkeypatch.setenv("FS_ALLOW_NO_DUP", "1")
    assert load_exclude(str(tmp_path / "missing.json")) == {}


def test_repo_exclude_file_is_present_and_nonempty():
    from foresign import paths
    ex = load_exclude(paths.CANON_EXCLUDE)
    assert len(ex.get("phoenix14t", ())) > 0          # PHOENIX cross-release overlap
    assert len(ex.get("wlasl", ())) > 0               # Phase 1a near-duplicate list, mapped


# ---- exclusion builder -----------------------------------------------------------------

def _write_manifest(path, rows):
    cols = ["clip_id", "dataset", "split", "dup_of"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(r.get(c, "") for c in cols) + "\n")


def _write_md5(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for cid, md5s in rows:
            f.write(f"{cid}\t{','.join(md5s)}\n")


def test_build_exclude(tmp_path):
    from tools.build_canon_exclude import build
    m = tmp_path
    _write_manifest(m / "phoenix14t.csv", [
        {"clip_id": "phoenix14t__train__a", "split": "train"},
        {"clip_id": "phoenix14t__train__b", "split": "train"},
        {"clip_id": "phoenix14t__test__c", "split": "test"}])
    _write_manifest(m / "phoenix14.csv", [
        {"clip_id": "phoenix14__train__x", "split": "train"},
        {"clip_id": "phoenix14__dev__y", "split": "val"}])
    _write_md5(m / "phoenix14t_frame_md5.tsv.gz", [
        ("phoenix14t__train__a", ["f1", "f2"]),      # shares f2 with ph14 dev y -> exclude
        ("phoenix14t__train__b", ["f3"]),            # clean
        ("phoenix14t__test__c", ["f4"])])
    _write_md5(m / "phoenix14_frame_md5.tsv.gz", [
        ("phoenix14__train__x", ["f4", "f9"]),       # shares f4 with ph14t test c -> exclude
        ("phoenix14__dev__y", ["f2", "f8"])])
    _write_manifest(m / "wlasl.csv", [
        {"clip_id": "wlasl__train__00338-wlasl-cat", "split": "train"},
        {"clip_id": "wlasl__train__00400-wlasl-dog", "split": "train", "dup_of": "wlasl__test__00500-wlasl-dog"},
        {"clip_id": "wlasl__test__00500-wlasl-dog", "split": "test"},
        {"clip_id": "wlasl__train__00600-wlasl-eat", "split": "train"}])
    _write_manifest(m / "chicagofswild.csv", [
        {"clip_id": "chicagofswild__a", "split": "train"},
        {"clip_id": "chicagofswild__b", "split": "test", "dup_of": "chicagofswild__a"}])
    old = tmp_path / "dup_exclude.json"
    old.write_text(json.dumps({"wlasl": ["wlasl2000/all/00338", "wlasl2000/all/99999"]}),
                   encoding="utf-8")
    rep = build(str(m), str(old))
    ex = {k: set(v) for k, v in rep["exclude"].items()}
    assert ex["phoenix14t"] == {"phoenix14t__train__a"}
    assert ex["phoenix14"] == {"phoenix14__train__x"}
    assert ex["wlasl"] == {"wlasl__train__00338-wlasl-cat", "wlasl__train__00400-wlasl-dog"}
    assert "chicagofswild" not in ex                      # not an evaluation dataset
    assert rep["sources"]["old_near_dup"]["unmapped"] == ["wlasl2000/all/99999"]
