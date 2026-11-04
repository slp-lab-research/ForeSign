"""Build autsl_labels.json (AUTSL clip -> class label) from the official AUTSL label files.

  python -m tools.build_autsl_labels --autsl-dir <dir with the official csv files> [--out <file>]

Inputs (from the AUTSL release; obtain them from the dataset authors under the AUTSL licence):
  SignList_ClassId_TR_EN.csv   ClassId,TR,EN
  train_labels.csv             signerX_sampleY,<class id>      (no header)
  validation_labels.csv        signerX_sampleY,<class id>
  test_labels.csv              signerX_sampleY,<class id>

Output: {"<train|val|test>/<signerX_sampleY>": "<english label>"}, written to
<project>/data_ann/autsl_labels.json by default (evals.islr_canon reads it via $FS_ANN_DIR or
<project>/data_ann). Labels are the lower-cased English names with "_" -> " ". Two class ids share
the English name "you", so the protocol has 225 label classes (as in all reported results).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

SPLITS = (("train", "train_labels.csv"), ("val", "validation_labels.csv"),
          ("test", "test_labels.csv"))


def english_names(path: str) -> dict:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {int(r["ClassId"]): " ".join(r["EN"].strip().lower().replace("_", " ").split())
                for r in csv.DictReader(f)}


def read_labels(path: str) -> list:
    """-> [(sample name, class id)]; header / malformed lines are skipped."""
    out = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            p = [x.strip() for x in line.strip().split(",")]
            if len(p) >= 2 and p[1].isdigit():
                out.append((p[0], int(p[1])))
    return out


def build(autsl_dir: str) -> dict:
    names = english_names(os.path.join(autsl_dir, "SignList_ClassId_TR_EN.csv"))
    out = {}
    for split, fn in SPLITS:
        p = os.path.join(autsl_dir, fn)
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} missing")
        for name, cid in read_labels(p):
            out[f"{split}/{name}"] = names[cid]
    return dict(sorted(out.items()))


def main(argv=None):
    from foresign import paths
    ap = argparse.ArgumentParser()
    ap.add_argument("--autsl-dir", required=True)
    ap.add_argument("--out", default=os.path.join(paths.DATA_ANN, "autsl_labels.json"))
    a = ap.parse_args(argv)
    out = build(a.autsl_dir)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=0)
    print(f"[autsl_labels] {len(out)} clips, {len(set(out.values()))} classes -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
