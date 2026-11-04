"""Build the CSLR annotation JSONs for PHOENIX-2014T and PHOENIX-2014 (multisigner) from the
official corpus files of the releases (obtain the releases from RWTH Aachen under their licence).

  python -m tools.build_phoenix_ann --release ph14t --corpus-dir <...>/PHOENIX-2014-T/annotations/manual
  python -m tools.build_phoenix_ann --release ph14  --corpus-dir <...>/phoenix-2014-multisigner/annotations/manual

Inputs:  ph14t  PHOENIX-2014-T.{train,dev,test}.corpus.csv  (name|video|start|end|speaker|orth|translation)
         ph14   {train,dev,test}.corpus.csv                 (id|folder|signer|annotation)
Output:  <out>/phoenix14t_{split}.json | phoenix14_{split}.json, each a list of
         {"id", "gloss", "text"} in corpus order (gloss whitespace collapsed; text = the German
         translation for 2014T, "" for 2014). Default <out> = <project>/data_ann (evals.cslr_phoenix
         and evals.cslr_cosign look in $FS_ANN_DIR, then there).
The annotation ids equal the frame-folder names, i.e. the canon clip id suffix
(<dataset>__<split>__<id>).
PHOENIX-2014 train: 13April_2011_Wednesday_tagesschau_default-14 is left out by default, as in the
common VAC/CorrNet preprocessing that the reported runs followed (5671 train utterances);
--keep-all keeps it.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

SPLITS = ("train", "dev", "test")
RELEASES = {
    "ph14t": {"csv": "PHOENIX-2014-T.{}.corpus.csv", "out": "phoenix14t_{}.json",
              "id": "name", "gloss": "orth", "text": "translation"},
    "ph14": {"csv": "{}.corpus.csv", "out": "phoenix14_{}.json",
             "id": "id", "gloss": "annotation", "text": None},
}
DEFAULT_DROP = {"ph14": {"train": {"13April_2011_Wednesday_tagesschau_default-14"}}}


def read_corpus(path: str, release: str) -> list:
    spec = RELEASES[release]
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="|", quoting=csv.QUOTE_NONE))
    return [{"id": r[spec["id"]].strip(), "gloss": " ".join(r[spec["gloss"]].split()),
             "text": (r[spec["text"]].strip() if spec["text"] else "")} for r in rows]


def main(argv=None):
    from foresign import paths
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", choices=sorted(RELEASES), required=True)
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--out", default=paths.DATA_ANN)
    ap.add_argument("--keep-all", action="store_true")
    a = ap.parse_args(argv)
    spec = RELEASES[a.release]
    os.makedirs(a.out, exist_ok=True)
    for split in SPLITS:
        rows = read_corpus(os.path.join(a.corpus_dir, spec["csv"].format(split)), a.release)
        if not a.keep_all:
            drop = DEFAULT_DROP.get(a.release, {}).get(split, set())
            rows = [r for r in rows if r["id"] not in drop]
        dst = os.path.join(a.out, spec["out"].format(split))
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=0)
        print(f"[phoenix_ann] {a.release} {split}: {len(rows)} utterances -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
