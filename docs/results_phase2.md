# Phase-2 results summary (snapshot 2026-11-08)

Short public summary of the phase-2 experiments described in `docs/experiments.md`, including the
follow-up runs started after the initial code release. The paper is in preparation; this page reports
headline numbers only. Training code for the follow-up variants is not part of this release.

Snapshot: logs up to 2026-11-08 09:49 (server time). Runs marked *in progress* had not finished at
that time. All numbers are single runs with pretraining seed 0 unless a ± (sd over evaluation seeds)
is shown.

## 1. Fine-tuned evaluation (main protocols)

### ISLR, standard recipe (60 epochs, full fine-tuning, top-1 per-instance accuracy)

| checkpoint (64k steps) | WLASL2000 | WLASL2000 20% labels (3 seeds) | WLASL100 | WLASL300 | AUTSL | ASL Citizen | LSA64 signer split |
|---|---|---|---|---|---|---|---|
| random init (same architecture) | 0.340 | 0.025 ± 0.003 | 0.539 | 0.514 | 0.922 | 0.615 | 0.950 |
| prediction (JEPA) | **0.483** | **0.103 ± 0.009** | **0.779** | **0.720** | 0.951 | **0.721** | 0.981 |
| prediction + 3D view aug. | 0.474 | 0.092 ± 0.006 | **0.779** | 0.717 | **0.952** | 0.717 | **0.986** |
| multi-signal (pred + InfoNCE + boundary) | 0.350 | 0.028 ± 0.004 | 0.605 | 0.572 | 0.940 | 0.673 | 0.947 |
| multi-signal + 3D view aug. | 0.396 | 0.024 ± 0.003 | 0.616 | 0.617 | 0.942 | 0.678 | 0.970 |

LSA64 with the random split is at ceiling for every checkpoint (0.994-0.998) and is not discriminative.

### CSLR, CoSign-style single-stream fine-tuning (40 epochs, official-protocol WER %, beam search)

| checkpoint (64k steps) | PHOENIX-2014T dev | PHOENIX-2014T test | PHOENIX-2014 dev | PHOENIX-2014 test |
|---|---|---|---|---|
| random init | 39.41 | 39.21 | 40.32 | 40.38 |
| prediction (JEPA) | **31.35** | **31.31** | **35.74** | **35.76** |
| prediction + 3D view aug. | 31.54 | **31.31** | 35.78 | 35.90 |
| multi-signal | 33.80 | 33.68 | 36.59 | 37.07 |
| multi-signal + 3D view aug. | 34.15 | 35.06 | 37.22 | 37.82 |

### CSLR, signer-independent split of PHOENIX-2014T (new follow-up)

Training signers are disjoint from a held-out signer group (train 5,246 sentences); dev/test are the
official sets, scored separately for seen and unseen signers.

| checkpoint | test WER | test seen / unseen signers | seen-unseen gap |
|---|---|---|---|
| random init | 45.22 | 41.82 / 59.83 | 18.01 |
| prediction (JEPA, 64k) | **35.86** | **32.14 / 51.87** | 19.73 |

Pretraining lowers WER on unseen signers by 8.0 points, about as much as on seen signers (9.7);
the seen-unseen gap itself does not shrink.

### Low-label fine-tuning (top-1)

| checkpoint | AUTSL 10% | AUTSL 100% | WLASL2000 100% | LSA64 10% | LSA64 100% |
|---|---|---|---|---|---|
| random init | 0.694 | 0.922 | 0.321 | 0.683 | 0.952 |
| prediction, 16k | 0.810 | 0.935 | 0.407 | 0.895 | 0.983 |
| prediction, 32k | 0.819 | 0.940 | 0.433 | 0.888 | 0.984 |
| prediction, 64k | **0.826** | **0.946** | **0.448** | 0.890 | **0.991** |

(10% columns: mean of 3 seeds. WLASL2000 with 10% labels stays below 0.04 top-1 for every checkpoint.)

## 2. Frozen-encoder probes

- **Frozen probes and fine-tuning disagree.** For the prediction objective, frozen WLASL2000 linear
  accuracy falls from 0.114 (16k) to 0.070 (64k) while fine-tuned WLASL2000 accuracy rises from 0.407
  to 0.448 over the same checkpoints. Frozen probes are therefore not used to select checkpoints.
- **Multi-signal training degrades frozen features as training continues** (WLASL2000 linear 0.042 →
  0.018 → 0.011 at 16k/32k/64k; frozen CSLR dev WER 0.73 → 0.88 → 0.91), and the fine-tuned results
  above confirm it trails the prediction objective. This pattern suggests that the InfoNCE term is
  solved through a shortcut (tested in §3.1).
- A random-init encoder is a strong frozen baseline (frozen CSLR dev WER 0.690 at full depth, 0.563
  after the first layer); the first-layer tap gives lower frozen CSLR WER than full depth for 13 of
  14 checkpoints.
- **Dataset-ID probe** (ASL Citizen / WLASL / How2Sign, chance 0.333): every encoder (0.80-0.94) stays
  at or below the input-level mean-pose baseline (0.942), i.e. pretraining does not add dataset identity
  beyond what the raw poses already carry.

## 3. Follow-up experiments after the code release

### 3.1 InfoNCE-shortcut variants (*in progress*)

To test the InfoNCE-shortcut hypothesis, three variants of the multi-signal objective were
pre-registered, each changing one setting of the InfoNCE term (A: `nce_target_pos: false`,
B: `nce_pos_jitter: 8`, C: `nce_cross_clip: true`, i.e. removing the positional cue from the target,
jittering the positive position, and drawing pairs across clips). All run on a 32k schedule together
with same-schedule references (prediction-only 32k, multi-signal 32k).

Pre-registered rule: a variant is adopted only if its WLASL2000 standard-recipe accuracy is at least
1.0 point above the prediction-only 32k reference; otherwise it is reported as a negative result
without further tuning.

Status at the snapshot:

| run | training | frozen probes | standard recipe (rule metric) |
|---|---|---|---|
| prediction-only 32k (reference) | done | WLASL2000 linear 0.106, frozen CSLR dev 0.637 | not run yet |
| variant A | done | WLASL2000 linear 0.023, frozen CSLR dev 0.670 | not run yet |
| variant B | 29.5k / 32k steps | – | – |
| variant C, multi-signal 32k | not started | – | – |

In both variants with logs, InfoNCE accuracy still reaches 0.99 by 4k steps and 1.000 by 8k steps, so
neither removes the shortcut by this measure. No adoption decision has been made yet.

### 3.2 Larger model: learning-rate sweep (done) and main run (*in progress*)

Large encoder: d=384, 12 layers (118.4M parameters vs 20.3M for the base model). Short 8k-step sweep,
evaluated with the standard recipe:

| model | lr | WLASL100 | LSA64 signer split |
|---|---|---|---|
| base, 8k (reference) | 1.5e-4 | 0.686 | **0.966** |
| large, 8k | 1.5e-4 | 0.713 | 0.961 |
| large, 8k | **1e-4 (selected)** | **0.717** | 0.963 |
| large, 8k | 5e-5 | 0.678 | 0.948 |

Selection rule (fixed before the sweep): best WLASL100, preferring the lower learning rate when the top
two are within 1.0 point. lr 1e-4 was selected automatically and a 128k-step main run started
(37.2k / 128k steps at the snapshot; no evaluation yet).

## 4. Summary of findings so far

1. The latent-prediction (JEPA) objective gives the best fine-tuned ISLR and CSLR results on every
   benchmark that is not at ceiling (e.g. WLASL2000 0.483 vs 0.340 random init; PHOENIX-2014T test WER
   31.3 vs 39.2), and it keeps improving with longer pretraining under fine-tuning.
2. Adding InfoNCE + boundary terms hurts, and its frozen features degrade with training, pointing to
   a shortcut in the InfoNCE term; the first follow-up variants still solve that task almost
   immediately.
3. 3D view augmentation during pretraining gives no clear gain over the plain prediction objective.
4. Pretraining helps unseen signers as much as seen signers in CSLR, but does not close the
   seen-unseen gap.
5. Frozen probe accuracy is not a reliable proxy for fine-tuned performance in this setting.
