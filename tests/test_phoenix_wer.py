"""The Python port must reproduce sclite on the example hypotheses shipped with the releases."""
import os
import re

import pytest

from evals import phoenix_wer as pw

# Official release evaluation folders (not included; obtain the PHOENIX releases from RWTH):
#   FS_PH14_RELEASE_EVAL  = <...>/phoenix-2014-multisigner/evaluation
#   FS_PH14T_RELEASE_EVAL = <...>/PHOENIX-2014-T/evaluation/sign-recognition
# The ground-truth .stm files are read from $FS_PHOENIX_EVAL_DIR (default
# <project>/data_ann/phoenix_eval; copy them there from the release evaluation folders).
REL14 = os.environ.get("FS_PH14_RELEASE_EVAL", "")
REL14T = os.environ.get("FS_PH14T_RELEASE_EVAL", "")


def _stm_or_skip(ds, split):
    try:
        return pw.read_stm(ds, split)
    except OSError:
        pytest.skip("PHOENIX ground-truth .stm not available (set FS_PHOENIX_EVAL_DIR)")


def _sys_sum(path):
    """Sum/Avg row of an sclite .sys file -> (#snt, #wrd, corr, sub, del, ins, err)."""
    for line in open(path, encoding="utf-8", errors="replace"):
        if "Sum/Avg" in line:
            nums = re.findall(r"[\d.]+", line)
            return tuple(float(x) for x in nums[:7])
    raise AssertionError("no Sum/Avg")


CASES = [("ph14", REL14, "test"), ("ph14t", REL14T, "dev"), ("ph14t", REL14T, "test")]


@pytest.mark.parametrize("ds,rel,split", CASES)
def test_matches_official_sclite_on_shipped_examples(ds, rel, split):
    sysf = os.path.join(rel, f"out.example-hypothesis-{split}.ctm.sys")
    if not rel or not os.path.exists(sysf):
        pytest.skip("PHOENIX release evaluation folder not available (set FS_PH14*_RELEASE_EVAL)")
    ctm = [l.rstrip("\n") for l in open(os.path.join(rel, f"example-hypothesis-{split}.ctm"),
                                        encoding="utf-8") if l.strip()]
    res = pw.score_lines(ds, ctm, _stm_or_skip(ds, split))
    snt, wrd, corr, sub, dele, ins, err = _sys_sum(sysf)
    assert res["sent"] == snt and res["N"] == wrd
    got = [round(res[k], 1) for k in ("corr_pct", "sub_pct", "del_pct", "ins_pct", "wer")]
    assert got == [corr, sub, dele, ins, err], (got, (corr, sub, dele, ins, err))


def test_rules_units():
    # 2014 hypothesis rewrites
    assert pw._hyp_rules_ph14("x 1 0.00 0.01 loc-NORD") == "x 1 0.00 0.01 NORD"
    assert pw._hyp_rules_ph14("x 1 0.00 0.01 KLASSENRAUM") == "x 1 0.00 0.01 KLASSEN"
    # 2014T: __x__ tokens dropped, repeats merged, empty utterance filled
    lines = pw.ctm_lines({"a": ["__ON__", "REGEN", "REGEN", "loc-NORD"], "b": ["__OFF__"],
                          "c": ["SONNE"]})
    out = pw.hyp_pipeline_ph14t(lines)
    hyps = pw.merge_ctm_stm(out, ["a 1 s 0.0 1 REGEN NORD", "b 1 s 0.0 1 X", "c 1 s 0.0 1 SONNE"])
    assert hyps == {"a": ["REGEN", "NORD"], "b": ["[EMPTY]"], "c": ["SONNE"]}


def test_align_counts():
    assert pw.align("A B C".split(), "A B C".split()) == (3, 0, 0, 0)
    assert pw.align("A B C".split(), "A X C".split()) == (2, 1, 0, 0)
    assert pw.align("A B C".split(), "A C".split()) == (2, 0, 1, 0)
    assert pw.align("A B".split(), "A Y B".split()) == (2, 0, 0, 1)
    assert pw.align("A".split(), []) == (0, 0, 1, 0)
    assert pw.align("ab".split(), "AB".split()) == (1, 0, 0, 0)        # case-insensitive


def test_wer_api_reference_as_hypothesis():
    """Feeding the processed reference back is ~0 but not exactly 0 under the official 2014T
    script: its hypothesis step merges consecutive repeated glosses, which the reference keeps
    (official behaviour, reproduced on purpose)."""
    stm = _stm_or_skip("ph14t", "dev")
    ref = pw.ref_pipeline_ph14t(stm)
    hyps = {l.split()[0]: l.split()[5:] for l in ref}
    r = pw.wer("ph14t", "dev", hyps)
    assert 0.0 <= r["wer"] < 0.1 and r["S"] == 0 and r["I"] == 0
    raw14 = {l.split()[0]: l.split()[5:] for l in _stm_or_skip("ph14", "dev")}
    assert pw.wer("ph14", "dev", raw14)["wer"] < 5.0
