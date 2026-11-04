# Phase 2 — adapt ForeSign to C77-3D (canon) data, pilot prep

*Internal design/record document, lightly edited for the code release; dates are those of the original record.*

- Date: 2026-10-30.
- Input format: C77-3D (canon-v1/v4), spec `docs/design/2026-10-29-canonicalisation-fingerprint-gate-design.md` §2, §9.
- Scope: code + tests + a small-scale smoke test. No long training in this step.

## Design decisions

1. **Input packing.** A canon clip (`x (T,77,3)` f16, `mask (T,77)` u8) is packed per frame into
   `(77, in_dim+1)` = coordinates (first `in_dim` of x, y, z; zeroed where mask 0) + the joint mask,
   flattened to `(T, 77*(in_dim+1))`. The tokenizer projects only the coordinate channels and uses
   the mask channel as its validity gate (the role conf had). Padding frames are all zero (mask 0).
   Config: `input_format: c77` (new) | `resona77` (default, Phase 1a path unchanged, checkpoints
   still load); `in_dim: 3`. `drop_conf` must be false for c77 (enforced).
2. **Data root.** `FS_CANON_ROOT` env (wins) or cfg `canon_root`. Files `<root>/<ds>.h5`,
   YouTube-ASL `<root>/youtube_asl_b*.h5`. Index (file, clip id, T, split attr) cached under
   `cache/canon_index_<hash(root)>.pt`, invalidated by file size/mtime.
3. **Sampling.** Temperature 0.5 over datasets as before. Because continuous corpora are now whole
   videos (How2Sign/OpenASL/YouTube-ASL: one clip = minutes), the sampling unit is configurable:
   `sample_unit: windows` (default for p2: dataset size and within-dataset clip weight = number of
   `clip_len` windows, ceil(T/clip_len)) or `clips` (Phase 1a behaviour). Random crop (partial H5
   read), pad with mask 0.
4. **Splits / inductive filter (fail-closed).** `foresign/canon_splits.py`: eval datasets
   autsl, wlasl, phoenix14t, phoenix14, how2sign use the clip's `split` attr (train → pool, val/test
   → held out, anything else → excluded); lsa64 uses signers 1–8 train / 9–10 test (id field 2);
   other known pool datasets → train; unknown dataset → error. phoenix14 is not in the default pool
   (as Phase 1a). Exclusion list `data_ann/canon_exclude.json` built by `tools/build_canon_exclude.py`:
   (a) every PHOENIX clip (either release) sharing ≥1 frame md5 with a val/test clip of either
   release, (b) manifest `dup_of` groups touching a held-out clip, (c) the Phase 1a WLASL
   near-duplicate list mapped to canon ids. Missing file → error unless `FS_ALLOW_NO_DUP=1`.
5. **Evals on canon.** `evals/islr_canon.py` (AUTSL/WLASL/LSA64, official splits from attrs / rules,
   same linear/attentive/kNN probes and 3 seeds, center crop) → `islr_canon_<tag>.json`.
   `evals/cslr_phoenix.py` gains datasets `canon_ph14t`, `canon_ph14`: annotation ids mapped to
   canon clip ids, missing clips scored as deletions and counted; no per-stream normalisation.
   Taps L0/L1, 40 epochs, same head. `evals/dsid_probe.py`: dataset-ID linear probe
   (asl_citizen, wlasl, how2sign; 300 train clips each, rng 0; one 50-frame window per clip; pooled
   frozen features; 5-fold stratified logistic regression C=1 on standardised features) plus an
   input-level mean-pose baseline → `dsid_<tag>.json`.
6. **Configs + queue.** `tools/make_configs.py p2` → `configs/p2/p2_{random_init,pred,nce,ms}_s0.yaml`
   (Phase 1a constants; `input_format c77`, `in_dim 3`, `drop_conf false`, canon dataset list).
   `tools/run_queue.py`: c77 configs get the p2 job set (ISLR canon, CSLR canon_ph14t, dsid) per
   checkpoint × tap; `random_init: true` configs write a step-0 checkpoint instead of training.
   GPU only via `CUDA_VISIBLE_DEVICES` from the launcher.

## Tasks (test-first)
1. inputs + tokenizer/model in_dim (tests: packing, gate, old-shape compatibility).
2. canon_splits + inductive filter + exclusion builder (tests on synthetic ids/files).
3. canon data loader + index cache (tests on a tiny synthetic H5).
4. train.py wiring (motion/boundary on 3D layout).
5. adapters, islr_canon, cslr canon datasets, dsid probe (tests on synthetic H5 + tiny model).
6. configs + queue (tests).
7. Smoke test on small copies of canon-v1 files.

## Launching the pilot

Run from the repository root (the model blocks needed are vendored in `foresign/_vendor/`):

```bash
export FS_CANON_ROOT=<dir with canon *.h5>
export CUDA_VISIBLE_DEVICES=<gpu>
python -u -m tools.run_queue "configs/p2/p2_*.yaml"
# PHOENIX-2014 CSLR afterwards: FS_QUEUE_EXTRA=cslr_ph14 python -m tools.run_queue "configs/p2/p2_*.yaml"
```
Results: `results/islr_canon_<tag>.json`, `cslr_frozen_<tag>_s0.json`, `dsid_<tag>.json`,
tags `p2_<obj>_s0_k{2000,6000,16000}_L{0,1}` and `p2_random_init_s0_k0_L{0,1}`.

## Addendum 2026-10-31: low-label fine-tuning ISLR evaluation (fixed before coding)

Why: masked/JEPA features often pay off only under fine-tuning, so the standard SSL low-label
fine-tuning protocol is added.
One fixed recipe for every checkpoint, including random init; no per-checkpoint tuning.

- **Datasets:** AUTSL, WLASL (official train/test split attrs; val unused), LSA64 (signers 1–8
  train / 9–10 test). Labels and classes as in `evals/islr_canon.py` (classes = labels in the full
  train split).
- **Label budgets:** 10% and 100% of the train split. 10% = class-stratified: per class
  max(1, round(0.1·n_c)) clips drawn with `numpy.random.default_rng(subset_seed)` from the clip-id
  sorted list (independent of the checkpoint). 10%: subset seeds 0, 1, 2; 100%: seed 0 only
  (seed also sets head init and batch order).
- **Model:** EMA/target encoder (tokenizer + body) from the checkpoint; tap convention as the probes
  (`FS_TAP_DEPTH` 0 = full depth incl. final norm = L0; launcher uses L0). Joint-mean → time
  tokens → attentive-pooling head (learned query, 8-head attention over valid time tokens, LayerNorm,
  Linear). Full fine-tuning: every encoder weight trainable (drop-path as built, 0.1).
- **Optimisation:** AdamW, weight decay 0.05 (not on biases / norm weights / embeddings), head lr
  1e-3, encoder base lr 5e-4 with layer-wise decay 0.75: lr(layer l) = 5e-4 · 0.75^(depth+1−l),
  l = 0 tokenizer, 1..depth DSTformer layers, depth+1 final norm. 30 epochs (100%) / 60 epochs (10%),
  linear warmup 2 epochs then cosine to 0, batch 64, label smoothing 0.1, bf16 autocast, grad clip 1.0.
- **Data:** clip_len from the checkpoint (128); train = random temporal crop (pad with mask 0 if
  shorter), test = centre crop. No other augmentation.
- **Reporting:** final-epoch test top-1 / top-5 (no model selection on test); per run JSON
  `results/ft_<tag>_<ds>_<budget>_s<seed>.json` (tag e.g. `p2_ms_s0_k2000_L0`); mean ± sd over seeds
  in the summary.
- **Checkpoints:** random_init k0, ms k2000, ms k16000, pred k2000, pred k16000 (a list; missing
  checkpoints are skipped and reported).
- **Implementation:** `evals/finetune_islr_canon.py` (one run) + `tools/run_finetune_p2.py`
  (checkpoints × datasets × budgets × seeds, skips existing result JSONs, per-epoch resumable state).

## Addendum 2026-11-02: pretraining-only 3D view augmentation (pre-registered ablation)

Fixed before coding. Compared later against the un-augmented 64k runs
(`p2_{pred,ms}_long_s0`) at equal steps (16k / 32k / 64k). All evaluation and fine-tuning data stay
un-augmented.

- **Transform:** per sampled pretraining window (on the fly, never stored), one rotation of the whole
  skeleton about the canonical origin (shoulder midpoint): R = R_x(pitch) · R_y(yaw),
  yaw ~ U(−θy, θy) about the vertical y axis, pitch ~ U(−θp, θp) about x, constant over the window.
  θy = 30°, θp = 15°. Applied to the 3D coordinates before packing (so it also holds for in_dim 2);
  absent joints stay 0 with mask 0. No scale jitter or other factor (one factor only).
- **Context and target see the same rotated view:** the rotated window is the single model input,
  so the online/context encoder and the EMA target encoder get identical input and the JEPA target
  is the rotated view. This is a view (data) augmentation, not a cross-view consistency objective.
  Motion/boundary targets are computed on the rotated window (3D motion magnitude is
  rotation-invariant).
- **Config:** `aug_rot_yaw_deg`, `aug_rot_pitch_deg` (default 0 = off; no random draws when off, so
  existing configs and their sample streams are unchanged). Rotation angles come from the
  per-worker sampling RNG (deterministic given seed and worker id).
- **Runs:** `configs/p2/p2_{pred,ms}_rot_s0.yaml` = `p2_{pred,ms}_long_s0.yaml` (64k steps,
  const_until 51200, ckpt_every 4000) + rotation; output `runs/p2/p2_{pred,ms}_rot_s0`. Evals at
  16k / 32k / 64k (`FS_QUEUE_CKPTS=step_16000.pt,step_32000.pt,last.pt`); fine-tuning launcher takes
  these checkpoints via `--ckpts` (tags `p2_{pred,ms}_rot_s0_k{16000,32000,64000}_L0`).

## Addendum 2026-11-03: SOTA-comparable protocols (pre-registered before coding)

Basis: `docs/sota_pose_slr.md` (research summary, 2026-11-03), checked against the primary sources
where accessible: MASA arXiv 2405.20666 Sec. IV-A, Zhao et al. arXiv 2406.10501 Sec. IV, CoSign
ICCV 2023 Sec. 3.3/4.1 (CVF PDF), the public VAC/CoSign code, and the
official PHOENIX evaluation scripts in the dataset releases. Checkpoints for every protocol: random
init (k0), `p2_pred_long_s0` k64000, `p2_ms_long_s0` k64000, `p2_pred_rot_s0` k64000; identical
recipes. Fast ops on (verified); compile where shapes are static. Existing protocols and tables are
kept unchanged; these are new recipes.

### A. ISLR standard recipe `islr_std` (MASA / Zhao lineage)
- **Frames:** 32 frames per clip. Train: the clip [0, T) is cut into 32 equal segments and one frame
  is drawn uniformly in each ("random temporal sampling"). Test: the centre frame of each segment
  ("centre sampling"). Sampled from our 25 fps canon clips (the papers sample from 25–30 fps
  video), so short clips repeat frames. 32 frames = 16 time tokens at t_patch 2 (our encoder
  was pretrained on 64 tokens; positions 0–15 are used).
- **Model:** EMA encoder, full depth (L0), all weights trainable. Head (MASA "simple classifier
  after pooling"; SHuBERT recipe): masked mean over time tokens and joint groups → BatchNorm1d →
  Linear.
- **Optimisation (MASA, which states the full schedule):** SGD momentum 0.9, lr 0.01 for all
  parameters, ×0.1 after epochs 20 and 40, 60 epochs, batch 64 (last partial train batch dropped
  so that BatchNorm and compiled shapes stay fixed). Weight decay 1e-4: not stated for fine-tuning
  by MASA or Zhao; 1e-4 is MASA's pretraining value. Plain cross-entropy (no smoothing, none
  stated). bf16 autocast, no gradient clipping (none stated).
- **Augmentation (train only), per clip:** left–right mirror with p = 0.5 (swap left/right joints:
  body 1↔2, 3↔4, 5↔6, 7↔8; hands 9+i↔30+i; mouth 61↔291, 40↔270, 321↔91; face 70↔300,
  105↔334, 107↔336, 33↔263, 133↔362, 234↔454, 172↔397; then negate x); in-plane rotation about z
  U(±15°); isotropic scale U(0.8, 1.2); shift U(±0.1) shoulder widths in x and y. Exact ranges are
  not given by the papers ("rotation, scale, flip"), so these values are fixed here. Absent joints
  stay 0 / mask 0.
- **Test:** single centre-sampled view (no multi-crop).
- **Metrics:** per-instance top-1/top-5 (P-I) and per-class top-1/top-5 (P-C, mean over classes
  present in test).
- **Runs:** WLASL2000 100% (seed 0) and 20% (Zhao Tab. VII: random 20% of the labelled train
  clips, not stratified; subset seeds 0, 1, 2, independent of the checkpoint); AUTSL 100%;
  ASL Citizen 100% (Rec@1/Rec@5 = top-1/top-5 on the official test split); LSA64 100% with the
  signer split (1–8 / 9–10) and with a seeded, class-stratified random 80/20 split (split seeds
  0, 1, 2; 40 train / 10 test per class). Classes = labels in the full train split.

### B. CSLR fine-tuning `cslr_cosign` (CoSign ICCV 2023 single-stream recipe)
- **Model:** EMA encoder (all weights trainable, L0), applied to non-overlapping 128-frame windows
  (its pretraining length; tokens concatenated) → 1D-CNN → 2-layer BiLSTM (2×512) → linear
  classifier; auxiliary linear classifier on the 1D-CNN output. Loss = CTC(primary) +
  CTC(auxiliary), weights 1:1 (CoSign config). CoSign's complementary-regularisation term and
  normalised classifier are method-specific and not included.
- **Temporal downsampling:** CoSign uses conv–pool–conv–pool (4×) on 25 fps frames. Our encoder
  already halves the rate (t_patch 2), so the 1D-CNN is conv(k5)–pool(2)–conv(k5), giving the same
  overall 4× (6.25 Hz) as CoSign. Hidden size 1024 (CoSign).
- **Optimisation:** AdamW, lr 4e-4, ÷10 after epochs 20 and 35, 40 epochs, batch 8, weight decay
  1e-4 (CoSign code config; not stated in the paper). Encoder layer-wise lr decay 0.75 (lr(l) =
  4e-4 · 0.75^(depth+1−l), as in our ISLR fine-tuning); new layers at 4e-4. bf16, grad clip 1.0.
- **Augmentation:** random temporal scaling ±20% only (CoSign): resample the sequence to
  round(T·U(0.8, 1.2)) frames (nearest frame index). No spatial augmentation.
- **Decoding:** CTC prefix beam search, beam width 10, no language model (VAC/CoSign
  `ctcdecode`, beam 10); greedy also reported.
- **Selection / reporting:** dev WER after every epoch; test WER of the best-dev epoch (as VAC/CoSign).
- **WER:** the official evaluation of each release ported to Python: PHOENIX-2014
  `evaluatePhoenix2014.sh` (hypothesis rewrites on CTM lines, raw ground-truth STM) and PHOENIX-2014T
  `evaluatePHOENIX-2014-T-signrecognition.sh` (rewrites of hypothesis and reference, removal of
  `__x__` tokens, merge of repeated words), `mergectmstm.py` ([EMPTY] for missing hypotheses), and
  sclite-style alignment (ins 3 / del 3 / sub 4; WER = (S+D+I)/N). Verified on the example
  hypotheses shipped with both releases against their sclite outputs (2014 test 45.1; 2014T dev
  24.7, test 26.7) before use. The official ground-truth STM files must be copied from the PHOENIX
  releases into the directory given by `FS_PHOENIX_EVAL_DIR` (default `data_ann/phoenix_eval/`,
  not distributed).
- **Runs:** PHOENIX-2014T and PHOENIX-2014, seed 0, the four checkpoints. Training targets = the
  annotation glosses as given (vocabulary from train).
