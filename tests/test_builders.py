"""Annotation builders (tools.build_autsl_labels, tools.build_phoenix_ann) on synthetic files."""
import json

from tools import build_autsl_labels as BA
from tools import build_phoenix_ann as BP


def test_autsl_labels(tmp_path):
    (tmp_path / "SignList_ClassId_TR_EN.csv").write_text(
        "ClassId,TR,EN\n0,abla,sister\n1,afiyet_olsun,enjoy_your_meal\n2,sen,you\n3,siz,you\n",
        encoding="utf-8")
    (tmp_path / "train_labels.csv").write_text("signer0_sample1,1\nsigner0_sample2,2\n")
    (tmp_path / "validation_labels.csv").write_text("signer1_sample1,3\n")
    (tmp_path / "test_labels.csv").write_text("signer34_sample1,0\n")
    out = BA.build(str(tmp_path))
    assert out == {"test/signer34_sample1": "sister", "train/signer0_sample1": "enjoy your meal",
                   "train/signer0_sample2": "you", "val/signer1_sample1": "you"}


def test_phoenix_ann_both_releases(tmp_path):
    t = tmp_path / "t"
    m = tmp_path / "m"
    t.mkdir()
    m.mkdir()
    for sp in ("train", "dev", "test"):
        (t / f"PHOENIX-2014-T.{sp}.corpus.csv").write_text(
            "name|video|start|end|speaker|orth|translation\n"
            f"a-{sp}|a-{sp}/1/*.png|-1|-1|Signer01|REGEN  NORD|es regnet\n", encoding="utf-8")
        (m / f"{sp}.corpus.csv").write_text(
            "id|folder|signer|annotation\n"
            f"b-{sp}|b-{sp}/1/*.png|Signer04|SONNE  WARM\n"
            "13April_2011_Wednesday_tagesschau_default-14|x/1/*.png|Signer01|X\n",
            encoding="utf-8")
    out = tmp_path / "out"
    assert BP.main(["--release", "ph14t", "--corpus-dir", str(t), "--out", str(out)]) == 0
    assert BP.main(["--release", "ph14", "--corpus-dir", str(m), "--out", str(out)]) == 0
    dev = json.load(open(out / "phoenix14t_dev.json", encoding="utf-8"))
    assert dev == [{"id": "a-dev", "gloss": "REGEN NORD", "text": "es regnet"}]
    tr = json.load(open(out / "phoenix14_train.json", encoding="utf-8"))
    assert [r["id"] for r in tr] == ["b-train"]          # default drop of the known bad clip
    te = json.load(open(out / "phoenix14_test.json", encoding="utf-8"))
    assert len(te) == 2 and te[0]["gloss"] == "SONNE WARM" and te[0]["text"] == ""
