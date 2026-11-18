# Experiments (overview)

Short public summary of the experiment set run with this code. Headline numbers are in
`docs/results_phase2.md` (paper in preparation). Detailed, pre-registered protocols are in
`docs/design/2026-10-30-phase2-c77-3d-adaptation.md`.

## Data

All poses re-extracted from raw video with one frozen extractor (MediaPipe Holistic legacy,
`MP-L-V4V0`, `configs/extract/mp_l_v4v0.json`) and canonicalised to C77-3D with canon-v4
(`docs/design/2026-10-29-canonicalisation-fingerprint-gate-design.md` §9). Pretraining pool:
17 corpora (`foresign/canon_splits.py: DEFAULT_PRETRAIN` = `CANON_POOL` minus PHOENIX-2014, which
shares recordings with PHOENIX-2014T), inductive: every val/test clip of the evaluation datasets and the clips in
`data_ann/canon_exclude.json` are excluded.

## Pretraining configs (`configs/p2/`)

| config | objective | steps | notes |
|---|---|---|---|
| `p2_random_init_s0` | none (step-0 checkpoint) | 0 | random-init baseline, same architecture |
| `p2_pred_s0` | latent prediction (JEPA) | 16k | d192 x 8 layers |
| `p2_nce_s0` | temporal InfoNCE only | 16k | |
| `p2_ms_s0` | prediction + InfoNCE (beta 0.3) + boundary BCE (lambda 0.2) | 16k | |
| `p2_pred_long_s0`, `p2_ms_long_s0` | as above | 64k | |
| `p2_pred_rot_s0`, `p2_ms_rot_s0` | as the long runs + 3D view augmentation (yaw +-30 deg, pitch +-15 deg) | 64k | pretraining only |
| `p2_pred_large_s0` | prediction | 64k | d384 x 12 layers |

All: clip length 128 frames (25 fps), t_patch 2, 18 anatomical joint groups, scattered random token
masking at 14/18 ~ 78% of the tokens (`rand78`), EMA target encoder, AdamW (lr 1.5e-4, wd 0.05), batch 64, seed 0.

## Evaluation protocols

| protocol | entry point | launcher |
|---|---|---|
| frozen ISLR probes (linear x3 seeds, attentive x3 seeds, kNN@5) on AUTSL, WLASL2000, LSA64 | `evals.islr_canon` | `tools.run_queue` |
| frozen CSLR (conv + BiLSTM + CTC head, 40 epochs) on PHOENIX-2014T / -2014 | `evals.cslr_phoenix` | `tools.run_queue` |
| dataset-ID probe (ASL Citizen / WLASL / How2Sign) on frozen features vs input-level baseline | `evals.dsid_probe` | `tools.run_queue` |
| low-label fine-tuning ISLR (10% / 100% labels) | `evals.finetune_islr_canon` | `tools.run_finetune_p2` |
| ISLR standard recipe (MASA / Zhao lineage; WLASL 100% and 20%, AUTSL, ASL Citizen, LSA64 signer and random splits) | `evals.islr_std` | `tools.run_sota_p2` |
| CSLR fine-tuning, CoSign-style single-stream recipe, official-protocol WER | `evals.cslr_cosign` (+ `evals.phoenix_wer`) | `tools.run_sota_p2` |

Readout taps: L0 = full depth (final norm), L1 = after the first DSTformer layer (`FS_TAP_DEPTH`).
Engineering checks: `tools.equiv_compile` (compiled vs eager equivalence and speed),
`tools.profile_step` (per-stage step timing).
