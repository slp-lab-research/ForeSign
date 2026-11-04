# ForeSign

Self-supervised pose JEPA for sign language. ForeSign pretrains a spatio-temporal transformer on
3D body / hand / face keypoints by predicting the latent representation of masked tokens
(joint-embedding predictive architecture with an EMA target encoder), and evaluates the frozen and
fine-tuned encoder on isolated (ISLR) and continuous (CSLR) sign language recognition.

This repository contains the full pipeline used for the paper (under review):

1. **extract**: one frozen pose extractor for every corpus (MediaPipe Holistic legacy,
   `mediapipe==0.10.21`, model complexity 2, refined face, plus a MediaPipe Hands crop step for
   hands that Holistic misses; configuration `MP-L-V4V0` frozen in `configs/extract/mp_l_v4v0.json`);
2. **verify**: per-dataset completeness / frame-count / detection-rate checks;
3. **canonicalise (canon-v4)**: raw MediaPipe output -> **C77-3D**, 77 joints x (x, y, z) + a
   joint-validity mask at 25 fps (body from pose world landmarks, hands and face re-attached at the
   body wrists / nose, per-frame shoulder-midpoint centring, shoulder-width unit, yaw + roll to a
   frontal frame, face in its own scale, Savitzky-Golay smoothing, hips never used);
4. **pretrain**: ForeSign JEPA on 17 corpora; the 77 joints are grouped into **18 anatomical
   groups** (trunk, two arms, palm + 5 fingers per hand, mouth, eyes/brows, face contour), one
   token per group per 2 frames; DSTformer (dual-stream spatial/temporal) encoder and predictor;
5. **evaluate**: frozen probes (ISLR linear / attentive / kNN, CSLR CTC head, dataset-ID probe),
   low-label fine-tuning, and SOTA-comparable protocols (ISLR standard recipe; CoSign-style CSLR
   fine-tuning with the official PHOENIX WER scoring ported to Python).

No data, poses, annotations or checkpoints are included.

## Layout

```
foresign/            model (jepa.py, tokenize.py, groups.py, masking.py), data (canon_data.py,
                     canon_splits.py, inputs.py, augment.py), train.py, paths.py
foresign/_vendor/    minimal vendored modules from the lab's earlier code (see below)
evals/               islr_canon, finetune_islr_canon, islr_std, cslr_phoenix, cslr_cosign,
                     phoenix_wer, dsid_probe, probes
tools/               run_queue, run_finetune_p2, run_sota_p2, make_configs, make_random_ckpt,
                     build_canon_exclude, build_autsl_labels, build_phoenix_ann,
                     equiv_compile, profile_step
tools/extract/       extraction: datasets (registry), build_manifest, extract_dataset,
                     extractors, common, freeze_config, verify_dataset, cleanup, stream_extract
tools/canon/         canonicalisation: canon (all versions; canon-v4 adopted), build_canon, index,
                     fingerprint (dataset-fingerprint gate), quality (geometry checks, calibration)
configs/p2/          pretraining configs;  configs/extract/  frozen extractor config + reference rates
data_ann/            annotation files (only canon_exclude.json is shipped; see data_ann/README.md)
docs/                design records, extractor bake-off, fingerprint gate, canonicalisation
                     quality, SOTA protocol notes, docs/extraction.md (how to run), experiments.md
tests/               unit tests (synthetic data; data-dependent tests skip)
```

The extraction and canonicalisation code lives under `tools/` (not in top-level packages) because
the frozen extractor configuration records the sha256 of `tools/extract/extractors.py` and
`tools/extract/common.py` by path; the extractor refuses to run if they change.

## Installation

Two environments were used:

```bash
# pretraining / evaluation (Python 3.11, CUDA GPU)
pip install -r requirements.txt          # torch 2.11.0 (+cu128 build), numpy, h5py, scikit-learn, ...
# pose extraction / canonicalisation (Python 3.10-3.12, CPU)
pip install -r requirements-extract.txt  # mediapipe 0.10.21, opencv-contrib-python 4.11, numpy 1.26
```
Run every command from the repository root (packages are imported as `foresign`, `evals`,
`tools`; no installation step).

## Configuration (environment variables)

| variable | meaning |
|---|---|
| `FS_CANON_ROOT` | directory with the canon-v4 HDF5 files (`<dataset>.h5`, `youtube_asl_b*.h5`); required for training and evaluation (or set `canon_root` in a config) |
| `FS_PROJECT_ROOT` | working root for `data_ann/`, `cache/`, `results/`, `runs/` (default: the repository root) |
| `FS_ANN_DIR` | extra directory searched first for annotation files |
| `FS_PHOENIX_EVAL_DIR` | directory with the official PHOENIX ground-truth `.stm` files (default `data_ann/phoenix_eval`) |
| `FS_RESULTS_DIR` | where result JSONs are written (default `results/`) |
| `CUDA_VISIBLE_DEVICES` | GPU selection (the code never picks a GPU itself) |
| `REEXTRACT_SRC_ROOT`, `REEXTRACT_ROOT` | extraction source videos / work directory (`docs/extraction.md`) |

## Usage

### 1-3. Extraction, verification, canonicalisation

See `docs/extraction.md` for the source layout and all options. Per dataset:

```bash
export REEXTRACT_SRC_ROOT=/path/to/videos REEXTRACT_ROOT=/path/to/work
python -m tools.extract.build_manifest autsl --workers 16
python -m tools.extract.extract_dataset autsl --workers 16
python -m tools.extract.verify_dataset autsl --workers 16
python -m tools.canon.build_canon --version canon-v4 --src-root $REEXTRACT_ROOT \
    --ds autsl --out /path/to/canon_v4/autsl.h5 --workers 16
python -m tools.canon.index /path/to/canon_v4
```

### Annotations

```bash
python -m tools.build_autsl_labels --autsl-dir /path/to/AUTSL
python -m tools.build_phoenix_ann --release ph14t --corpus-dir /path/to/PHOENIX-2014-T/annotations/manual
python -m tools.build_phoenix_ann --release ph14  --corpus-dir /path/to/phoenix-2014-multisigner/annotations/manual
# official WER: copy the PHOENIX ground-truth .stm files into data_ann/phoenix_eval/
```

### 4. Pretraining

```bash
export FS_CANON_ROOT=/path/to/canon_v4 CUDA_VISIBLE_DEVICES=0
python -u -m foresign.train --config configs/p2/p2_pred_s0.yaml        # -> runs/p2/p2_pred_s0/
python -m tools.make_random_ckpt --config configs/p2/p2_random_init_s0.yaml \
    --out runs/p2/p2_random_init_s0/last.pt --seed 0                      # random-init baseline
```

### 5. Evaluation

```bash
# train (if needed) + frozen probes for every config x checkpoint x tap (resumable)
python -u -m tools.run_queue "configs/p2/p2_*_s0.yaml"
FS_QUEUE_EXTRA=cslr_ph14 python -u -m tools.run_queue "configs/p2/p2_*_s0.yaml"   # PHOENIX-2014 CSLR

# single runs
FS_ADAPTER=foresign:runs/p2/p2_pred_s0/last.pt FS_TAG=pred python -m evals.islr_canon
FS_CSLR_CKPT=runs/p2/p2_pred_s0/last.pt FS_CSLR_FREEZE=1 python -m evals.cslr_phoenix
FS_DSID_CKPT=runs/p2/p2_pred_s0/last.pt FS_TAG=pred python -m evals.dsid_probe

# low-label fine-tuning ISLR (checkpoints x AUTSL/WLASL/LSA64 x 10%/100% x seeds)
python -m tools.run_finetune_p2 --ckpts runs/p2/p2_pred_s0/last.pt --dry-run
python -m tools.run_finetune_p2 --ckpts runs/p2/p2_pred_s0/last.pt

# SOTA-comparable protocols: islr_std (MASA/Zhao recipe) and cslr_cosign (CoSign recipe)
python -m tools.run_sota_p2 --ckpts runs/p2/p2_pred_long_s0/last.pt --dry-run
python -m tools.run_sota_p2 --ckpts runs/p2/p2_pred_long_s0/last.pt
python -m tools.run_sota_p2 --summary
```
Each evaluator documents its environment variables in its module docstring. Results are written as
JSON to `results/`.

### Tests

```bash
python -m pytest tests            # training/eval environment; extraction tests need cv2 (skip otherwise)
```

## Data sources

No data is distributed. Obtain each dataset from its authors and use it under its licence; several
require a signed agreement or are restricted to non-commercial research. Pretraining pool
(`foresign/canon_splits.py`) and evaluation sets:

| dataset | sign language | data type | task(s) the annotations support | used here for | source |
|---|---|---|---|---|---|
| AUTSL | Turkish SL (TİD) | isolated signs (226 classes), RGB-D studio recordings | ISLR | pretraining (train split), ISLR | https://cvml.ankara.edu.tr/datasets/ |
| WLASL (WLASL2000) | American SL (ASL) | isolated signs (2,000 glosses), web videos | ISLR | pretraining (train split), ISLR | https://dxli94.github.io/WLASL/ |
| ASL Citizen | American SL (ASL) | isolated signs (2,731 glosses), crowdsourced webcam | ISLR | pretraining, ISLR (standard recipe) | https://www.microsoft.com/en-us/research/project/asl-citizen/ |
| LSA64 | Argentinian SL (LSA) | isolated signs (64 classes), 10 signers, controlled recordings | ISLR | pretraining (signers 1-8), ISLR | https://facundoq.github.io/datasets/lsa64/ |
| RWTH-PHOENIX-Weather 2014T | German SL (DGS) | continuous sentences, TV weather forecasts; gloss + German translation | CSLR, SLT | pretraining (train split, minus overlap), CSLR | https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/ |
| RWTH-PHOENIX-Weather 2014 (multisigner) | German SL (DGS) | continuous sentences, TV weather forecasts; gloss | CSLR | CSLR | https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/ |
| How2Sign | American SL (ASL) | continuous sentences, instructional videos; English translation | SLT | pretraining (train split); dataset-ID probe | https://how2sign.github.io/ |
| OpenASL | American SL (ASL) | continuous sentences, online videos (news, vlogs); English subtitles | SLT | pretraining | https://github.com/chevalierNoir/OpenASL |
| YouTube-ASL | American SL (ASL) | continuous signing, YouTube videos; English captions | SLT | pretraining | https://github.com/google-research/google-research/tree/master/youtube_asl |
| ChicagoFSWild / ChicagoFSWild+ | American SL (ASL) | fingerspelling sequences in the wild; letter transcriptions | fingerspelling recognition | pretraining | https://home.ttic.edu/~klivescu/ChicagoFSWild.htm |
| INCLUDE | Indian SL (ISL) | isolated signs (263 words) | ISLR | pretraining | https://zenodo.org/records/4010759 |
| iSign | Indian SL (ISL) | continuous sentences; English translation | SLT | pretraining | https://exploration-lab.github.io/iSign/ |
| LSE_UVIGO | Spanish SL (LSE) | isolated signs and continuous signing; gloss annotations | ISLR, CSLR | pretraining | see the dataset publication (Docío-Fernández et al., LREC 2020) |
| LSA-T | Argentinian SL (LSA) | continuous sentences, TV news; Spanish subtitles | SLT | pretraining | https://github.com/midusi/LSA-T |
| Slovo | Russian SL (RSL) | isolated signs (1,000 classes), crowdsourced | ISLR | pretraining | https://github.com/hukenovs/slovo |
| KArSL | Arabic SL (Saudi) | isolated signs (502 classes), Kinect recordings | ISLR | pretraining | https://hamzah-luqman.github.io/KArSL/ |
| Sign2MINT | German SL (DGS) | isolated signs, STEM lexicon | ISLR (lexicon) | pretraining | https://sign2mint.de/ |
| Ghanaian Sign Language lexicon (HANDS! lab) | Ghanaian SL | isolated signs, lexicon | ISLR (lexicon) | pretraining | see the dataset publication |

ISLR = isolated sign language recognition, CSLR = continuous sign language recognition (gloss
sequences), SLT = sign language translation (spoken-language text). Only poses are used for
pretraining; text annotations are not used.

Evaluation clips (val/test splits, the PHOENIX cross-release overlap, and duplicates of evaluation
clips listed in `data_ann/canon_exclude.json`) are excluded from pretraining.

## Results

TBD (paper under review). An overview of the experiment set is in `docs/experiments.md`.

## Vendored code and licence

This code is released under the MIT License (`LICENSE`).

`foresign/_vendor/resona/` contains the minimal modules of the lab's earlier code (RESONA) that the
model imports: `models/v4_blocks.py` (DSTformer body, attention, RMSNorm) and
`data/language_map.py` (dataset -> sign language). `v4_blocks.py` is ported with modifications
from **MotionBERT** (Zhu et al., ICCV 2023), which is licensed under the **Apache License 2.0**; its
licence text is kept in `foresign/_vendor/resona/LICENSE.MotionBERT` and that file remains subject
to it. See `foresign/_vendor/README.md`.

Notes: `design §x` references in some docstrings point to the internal design record of the
earlier project phase, which is not part of this release. The resona77 (2D) input format still
recognised by the model code is that earlier phase's format; this release trains and evaluates on
C77-3D only.

## Citation

TBD.
