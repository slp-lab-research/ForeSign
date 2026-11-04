# data_ann/

Annotation files used by the evaluations. Only `canon_exclude.json` is distributed; everything
else is derived from licence-restricted dataset releases and must be built locally (the directory
is git-ignored apart from these two files). The evaluations look in `$FS_ANN_DIR` first, then here
(`$FS_PROJECT_ROOT/data_ann`).

| file | used by | how to obtain |
|---|---|---|
| `canon_exclude.json` (shipped) | pretraining pool (`foresign.canon_splits`) | clip-id lists only (no annotation text): PHOENIX-2014T training clips that share frames with a PHOENIX-2014/2014T dev/test clip, and WLASL training clips that duplicate / near-duplicate an evaluation clip. Rebuild: `python -m tools.build_canon_exclude --manifest-dir <work>/manifest` |
| `autsl_labels.json` | `evals.islr_canon`, `evals.islr_std`, fine-tuning | `python -m tools.build_autsl_labels --autsl-dir <AUTSL release dir>` (needs `SignList_ClassId_TR_EN.csv`, `train_labels.csv`, `validation_labels.csv`, `test_labels.csv`) |
| `phoenix14t_{train,dev,test}.json` | `evals.cslr_phoenix`, `evals.cslr_cosign` | `python -m tools.build_phoenix_ann --release ph14t --corpus-dir <PHOENIX-2014-T>/annotations/manual` |
| `phoenix14_{train,dev,test}.json` | same | `python -m tools.build_phoenix_ann --release ph14 --corpus-dir <phoenix-2014-multisigner>/annotations/manual` |
| `phoenix_eval/phoenix2014-groundtruth-{dev,test}.stm`, `phoenix_eval/PHOENIX-2014-T-groundtruth-{dev,test}.stm` | official-protocol WER (`evals.phoenix_wer`) | copy from the `evaluation/` folders of the PHOENIX releases; or point `FS_PHOENIX_EVAL_DIR` at a folder holding them |

WLASL, LSA64 and ASL Citizen labels are read from the clip ids (WLASL gloss, LSA64 sign number,
ASL Citizen video file name); official splits come from the canon clips' `split` attribute, which
the extraction manifest copied from each release's split files.
