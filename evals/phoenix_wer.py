"""Official PHOENIX WER evaluation, ported to Python (no sclite/perl needed).

Reproduces the scoring pipelines shipped with the releases:
- PHOENIX-2014 `evaluatePhoenix2014.sh`: sed/perl rewrites applied to the hypothesis CTM lines,
  [EMPTY] filling (awk), `sort -k1,1 -k3,3`, raw ground-truth STM (sorted by id),
  `mergectmstm.py`, sclite `-f 0`.
- PHOENIX-2014T `evaluatePHOENIX-2014-T-signrecognition.sh`: grep/sed rewrites of the hypothesis
  (drop `__x__` tokens, strip loc-/cl-/-PLUSPLUS/...RAUM), [EMPTY] filling, merge of consecutive
  repeated words; rewrites of the reference STM; `mergectmstm.py`; sclite `-f 0`.
sclite scoring is emulated with its default alignment weights (correct 0, substitution 4,
insertion 3, deletion 3), case-insensitive; WER = (S + D + I) / N. Verified against the sclite
outputs of the example hypotheses shipped with both releases (tests/test_phoenix_wer.py).

Hypotheses are written as CTM lines like VAC/CoSign: `<id> 1 <k/100 :.2f> <(k+1)/100 :.2f> <GLOSS>`.
"""
from __future__ import annotations

import os
import re

from foresign import paths

EVAL_DIR = os.path.join(paths.PROJECT, "data_ann", "phoenix_eval")
STM = {"ph14": "phoenix2014-groundtruth-{}.stm", "ph14t": "PHOENIX-2014-T-groundtruth-{}.stm"}

_PERL_DEDUP = re.compile(r"(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])", re.ASCII)
_SPLIT_LETTERS = [  # shared by the 2014 hypothesis and 2014T reference rewrites
    (r"^([A-Z]) ([A-Z][+ ])", r"\1+\2"),
    (r"[ +]([A-Z]) ([A-Z]) ", r" \1+\2 "),
]
_JOIN_LETTERS = [
    (r"([ +][A-Z]) ([A-Z][ +])", r"\1+\2"),
    (r"([ +][A-Z]) ([A-Z][ +])", r"\1+\2"),
    (r"([ +][A-Z]) ([A-Z][ +])", r"\1+\2"),
    (r"([ +]SCH) ([A-Z][ +])", r"\1+\2"),
    (r"([ +]NN) ([A-Z][ +])", r"\1+\2"),
    (r"([ +][A-Z]) (NN[ +])", r"\1+\2"),
    (r"([ +][A-Z]) ([A-Z])$", r"\1+\2"),
]


def _sed(line, rules):
    for pat, rep in rules:
        line = re.sub(pat, rep, line)
    return line


def _perl_dedup(line):
    for _ in range(4):
        line = _PERL_DEDUP.sub(r"\1", line)
    return line


def ctm_lines(hyps: dict) -> list:
    """{id: [glosses]} -> CTM lines in the VAC/CoSign format (empty list -> no line)."""
    out = []
    for sid, words in hyps.items():
        for k, w in enumerate(words):
            out.append(f"{sid} 1 {k / 100:.2f} {(k + 1) / 100:.2f} {w}")
    return out


# ---- PHOENIX-2014 -------------------------------------------------------------------------

def _hyp_rules_ph14(line):
    line = _sed(line, [("loc-", ""), ("cl-", ""), ("qu-", ""), ("poss-", ""), ("lh-", ""),
                       ("S0NNE", "SONNE"), ("HABEN2", "HABEN")])
    line = _sed(line, [("__EMOTION__", ""), ("__PU__", ""), ("__LEFTHAND__", "")])
    line = _sed(line, [("WIE AUSSEHEN", "WIE-AUSSEHEN"), ("ZEIGEN ", "ZEIGEN-BILDSCHIRM "),
                       ("ZEIGEN$", "ZEIGEN-BILDSCHIRM")] + _SPLIT_LETTERS)
    line = _sed(line, _JOIN_LETTERS)
    line = _sed(line, [(r"([A-Z][A-Z])RAUM", r"\1"), ("-PLUSPLUS", "")])
    return _perl_dedup(line)


def _fill_empty(lines):
    """The awk step: utterances whose words were all erased get '<last row> [EMPTY]'.
    (As in the original, the check fires when the id changes, so the last utterance is not
    checked.)"""
    out, cnt, last_id, last_row = [], {}, "", ""
    for row in lines:
        row = re.sub(r"\s*$", "", row)
        f = row.split()
        rid = f[0] if f else ""
        if last_id != rid and cnt.get(last_id, 0) < 1 and last_row != "":
            out.append(last_row + " [EMPTY]")
        if len(f) >= 5 and f[4] != "":
            cnt[rid] = cnt.get(rid, 0) + 1
            out.append(row)
        last_id, last_row = rid, row
    return out


def hyp_pipeline_ph14(lines):
    lines = [_hyp_rules_ph14(l) for l in lines]
    lines = [l for l in lines if not any(t in l for t in
                                         ("__LEFTHAND__", "__EPENTHESIS__", "__EMOTION__"))]
    lines = _fill_empty(lines)
    # sort -k1,1 -k3,3 (C locale; ties by the whole line)
    return sorted(lines, key=lambda l: (l.split()[0], l.split()[2] if len(l.split()) > 2 else "",
                                        l))


def ref_pipeline_ph14(stm_lines):
    return sorted(stm_lines, key=lambda l: l.split()[0] if l.split() else "")


# ---- PHOENIX-2014T ------------------------------------------------------------------------

def hyp_pipeline_ph14t(lines):
    drop = ("__LEFTHAND__", "__EPENTHESIS__", "__EMOTION__")
    lines = [l for l in lines if not any(t in l for t in drop)]
    lines = [l for l in lines if not re.search(r" __[^_ ]*__$", l)]
    lines = [_sed(l, [(r" loc-([^ ]*)$", r" \1"), ("-PLUSPLUS$", ""), (r" cl-([^ ]*)$", r" \1"),
                      (r"\b([A-Z][A-Z]*)RAUM$", r"\1"), (r"\s*$", "")]) for l in lines]
    lines = _fill_empty(lines)
    out, prec, prec_id = [], None, None          # merge consecutive identical words
    for l in lines:
        f = l.split()
        if not f:
            continue
        if f[-1] != prec or f[0] != prec_id:
            out.append(l)
        prec_id, prec = f[0], f[-1]
    return out


def ref_pipeline_ph14t(stm_lines):
    lines = sorted(stm_lines, key=lambda l: l.split()[0] if l.split() else "")
    out = []
    for l in lines:
        l = _sed(l, [("__LEFTHAND__ ", ""), (" __LEFTHAND__", ""), (" __EPENTHESIS__", ""),
                     ("__EPENTHESIS__ ", ""), (" __EMOTION__", ""), ("__EMOTION__ ", "")])
        l = _sed(l, [(r"\b__[^_ ]*__\b", ""), (r"\bloc-([^ ]*)\b", r"\1"),
                     (r"\bcl-([^ ]*)\b", r"\1"), (r"\b([^ ]*)-PLUSPLUS\b", r"\1"),
                     (r"\b([A-Z][A-Z]*)RAUM\b", r"\1"), ("WIE AUSSEHEN", "WIE-AUSSEHEN")]
                 + _SPLIT_LETTERS)
        l = _sed(l, _JOIN_LETTERS)
        out.append(_perl_dedup(l))
    return out


# ---- merge + sclite-style scoring ---------------------------------------------------------

def merge_ctm_stm(ctm, stm):
    """mergectmstm.py: order hypothesis lines by the STM order and insert '[EMPTY]' for STM
    utterances without any hypothesis line. Returns {id: [hyp words]}."""
    by_id = {}
    for l in ctm:
        f = l.split()
        by_id.setdefault(f[0], []).append(f)
    out = {}
    for l in stm:
        f = l.split()
        if not f:
            continue
        rows = by_id.get(f[0])
        out[f[0]] = [r[4] for r in rows if len(r) > 4] if rows else ["[EMPTY]"]
    return out


def align(ref, hyp, sub=4, ins=3, dele=3):
    """sclite-style weighted Levenshtein (case-insensitive) -> (C, S, D, I)."""
    r = [w.lower() for w in ref]
    h = [w.lower() for w in hyp]
    n, m = len(r), len(h)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0], op[i][0] = i * dele, "D"
    for j in range(1, m + 1):
        cost[0][j], op[0][j] = j * ins, "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = cost[i - 1][j - 1] + (0 if r[i - 1] == h[j - 1] else sub)
            best, o = diag, ("C" if r[i - 1] == h[j - 1] else "S")
            if cost[i - 1][j] + dele < best:
                best, o = cost[i - 1][j] + dele, "D"
            if cost[i][j - 1] + ins < best:
                best, o = cost[i][j - 1] + ins, "I"
            cost[i][j], op[i][j] = best, o
    c = s = d = k = 0
    i, j = n, m
    while i > 0 or j > 0:
        o = op[i][j]
        if o in ("C", "S"):
            c += o == "C"; s += o == "S"; i -= 1; j -= 1
        elif o == "D":
            d += 1; i -= 1
        else:
            k += 1; j -= 1
    return c, s, d, k


def score_lines(ds, ctm, stm_lines):
    """ctm: raw hypothesis CTM lines; stm_lines: raw ground-truth STM lines."""
    if ds == "ph14":
        hyp, ref = hyp_pipeline_ph14(ctm), ref_pipeline_ph14(stm_lines)
    elif ds == "ph14t":
        hyp, ref = hyp_pipeline_ph14t(ctm), ref_pipeline_ph14t(stm_lines)
    else:
        raise ValueError(ds)
    hyps = merge_ctm_stm(hyp, ref)
    tot = {"N": 0, "C": 0, "S": 0, "D": 0, "I": 0, "sent": 0, "sent_err": 0}
    for l in ref:
        f = l.split()
        if not f:
            continue
        words = f[5:]
        c, s, d, i = align(words, hyps[f[0]])
        tot["N"] += len(words); tot["C"] += c; tot["S"] += s; tot["D"] += d; tot["I"] += i
        tot["sent"] += 1; tot["sent_err"] += (s + d + i) > 0
    n = max(1, tot["N"])
    tot.update({"wer": 100.0 * (tot["S"] + tot["D"] + tot["I"]) / n,
                "corr_pct": 100.0 * tot["C"] / n, "sub_pct": 100.0 * tot["S"] / n,
                "del_pct": 100.0 * tot["D"] / n, "ins_pct": 100.0 * tot["I"] / n})
    return tot


def read_stm(ds, split, eval_dir=None):
    p = os.path.join(eval_dir or os.environ.get("FS_PHOENIX_EVAL_DIR") or EVAL_DIR,
                     STM[ds].format(split))
    with open(p, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f if l.strip()]


def wer(ds, split, hyps: dict, eval_dir=None):
    """Official-protocol WER (%) for {utterance id: [glosses]} on dev/test of ph14 / ph14t."""
    return score_lines(ds, ctm_lines(hyps), read_stm(ds, split, eval_dir))
