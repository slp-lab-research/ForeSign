# SOTA comparison: pose/skeleton-based sign language recognition

*Literature notes compiled during the project (2026-11); lightly edited for the code release.*

Compiled 2026-11-03 as background for the ForeSign-MS (pose JEPA) evaluation protocols.

**How the numbers were verified.** Unless marked otherwise, every number was read directly from the table of the primary paper: the arXiv HTML or CVF PDF was retrieved and the table parsed programmatically, without going through secondary summaries. The source table is cited in each row. Some numbers are taken from another paper's comparison table because the original was not opened; those are marked *(secondary: X Tab. n)*. Anything that could not be checked is marked **unverified**.

Abbreviations: P-I = per-instance top-1, P-C = per-class top-1, T5 = top-5. "Pose-only" means only keypoint coordinates (or keypoint heatmaps, which are noted as such) at inference. "+R" means fused with RGB.

---

## 1. WLASL (official Li et al. split; test = 2,878 videos for WLASL2000)

### 1a. WLASL2000: pose/keypoint-input methods (sorted by P-I)

| Method (venue) | Input at inference | Keypoint extractor / #joints | Pre-training data | P-I T1 | P-I T5 | P-C T1 | P-C T5 | Source |
|---|---|---|---|---|---|---|---|---|
| Uni-Sign, pose-only (ICLR 2025) | pose | RTMPose-x (MMPose), hands+body+face subsets | YouTube-ASL 984 h, **sign→text supervised** (not SSL); mT5-Base LM (580M total) | **63.13** | – | 60.90 | – | [arXiv 2501.15187](https://arxiv.org/abs/2501.15187) Tab. 3 |
| MSLU (Zhou et al., 2024) | pose | MMPose whole-body (+InterWild hands) | SL-1.5M (~1,780 h, 1.55M sign-text pairs; masked pose modeling + sign-text contrastive) | 56.29 | 88.74 | 53.29 | 88.10 | [arXiv 2408.08544](https://arxiv.org/abs/2408.08544) Tab. V (journal version **unverified**) |
| Zhao et al., ST-consistency SSL (IEEE TIP; arXiv says "Accepted by TIP2023") | pose | MMPose 2D, 49 joints (7 trunk + 42 hand) | train sets of WLASL+MSASL+NMFs-CSL+SLR500 (Tab. II) | 51.98 | 85.44 | 49.46 | 84.32 | [arXiv 2406.10501](https://arxiv.org/abs/2406.10501) Tab. III |
| SL-GCN 4-stream (SAM-SLR-v2 "Keypoints") | pose | HRNet whole-body (MMPose), 27-node graph | none (from scratch) | 51.50 | 84.94 | 48.87 | 84.02 | [arXiv 2110.06161](https://arxiv.org/abs/2110.06161) Tab. VIII |
| NLA-SLR Keypoint-64 single stream (CVPR 2023) | keypoint **heatmaps** (S3D) | HRNet COCO-WholeBody, 63 kp | Kinetics-400 init, then SLR | 49.10 | 82.00 | 46.18 | 80.71 | [arXiv 2303.12080](https://arxiv.org/abs/2303.12080) Tab. 4 |
| MASA (IEEE TCSVT 2024) | pose | MMPose 2D, 49 joints | train sets of WLASL+MSASL+NMFs-CSL+SLR500 (160,313 samples) | 49.06 | 82.90 | 46.91 | 81.80 | [arXiv 2405.20666](https://arxiv.org/abs/2405.20666) Tab. II |
| SignBERT+ (TPAMI 2023) | pose | MMPose, 133 kp (hand-model-aware) | 230,246 videos (all SL datasets + extra) | 48.85 | 82.48 | 46.37 | 81.33 | [arXiv 2305.04868](https://arxiv.org/abs/2305.04868) Tab. XIII |
| SignBERT (H+P) (ICCV 2021) | hand + body pose | MMPose 2D | ISLR train sets + extra | 47.46 | 83.32 | 45.17 | 82.32 | [arXiv 2110.05382](https://arxiv.org/abs/2110.05382) Tab. 9 |
| NLA-SLR Keypoint-32 | keypoint heatmaps | as above | as above | 46.66 | 79.95 | 43.81 | 78.49 | 2303.12080 Tab. 4 |
| BEST (AAAI 2023) | pose | MMPose 2D (hand + body) | ISLR train sets (MSASL, WLASL, NMFs-CSL, SLR500) | 46.25 | 79.33 | 43.52 | 77.65 | [arXiv 2302.05075](https://arxiv.org/abs/2302.05075) Tab. 3 |
| HP3D / "P3D" | pose | – | – | 44.47 | 79.97* | 42.18 | 78.52 | (secondary: 2406.10501 Tab. III; MASA Tab. II prints T5 = 79.69). **Primary not checked** |
| SignBERT (H, hand only), the number usually cited | hand pose | MMPose | as above | 39.40 | 73.35 | 36.74 | 72.38 | 2110.05382 Tab. 9 |
| ST-GCN (as reproduced by SignBERT) | pose | MMPose | none | 34.40 | 66.57 | 32.53 | 65.45 | 2110.05382 Tab. 9 |
| OpenHands SL-GCN | pose | MediaPipe Holistic | none | 30.6 | – | – | – | [arXiv 2110.05877](https://arxiv.org/abs/2110.05877) Tab. 2 |
| OpenHands SL-GCN + DPC SSL | pose | MediaPipe | Indian SL 1,129 h (cross-lingual) | 27.4 (ST-GCN+DPC) | – | – | – | 2110.05877 Tab. 6 |
| Pose-TGCN (WACV 2020 dataset paper) | pose | OpenPose | none | 23.65 | 51.75 | – | – | [arXiv 1910.11006](https://arxiv.org/abs/1910.11006) Tab. 3 |
| Pose-GRU (WACV 2020) | pose | OpenPose | none | 22.54 | 49.81 | – | – | 1910.11006 Tab. 3 |
| PoseFormer (one-shot ISLR, trained on ASL Citizen, tested zero-shot on WLASL) | pose | MediaPipe | ASL Citizen | Rec@1 0.391 | – | – | – | [arXiv 2502.20171](https://arxiv.org/abs/2502.20171) Tab. 3 (**different protocol**: one-shot dictionary retrieval) |

Not directly comparable:
- **SHuBERT (ACL 2025)**: 60.90 / 58.01 (P-I / P-C) with rank-1 LoRA ([arXiv 2411.16765](https://arxiv.org/abs/2411.16765) Tab. 2). The input is *not* pose-only: it uses DINOv2 features of MediaPipe-cropped hands and the face plus body pose. It has 86M parameters and was pretrained SSL on YouTube-ASL (984 h).
- **SignDEG (arXiv 2609.06296, 2026)**: 68.5 ± 0.4 / 65.9 ± 0.4 with full fine-tuning, 66.9 with rank-1 LoRA, 60.1 with a frozen encoder + linear head (Tab. 3). The input is RGB (DINOv3 crops).
- **SignRep (ICCV 2025)**: 61.05 / 58.89 ([arXiv 2503.08529](https://arxiv.org/abs/2503.08529) Tab. 1). RGB (Hiera) with pose-derived pretraining targets; no keypoints at inference.
- **StepNet**: 56.89 P-I, RGB (TSM) ([arXiv 2212.12857](https://arxiv.org/abs/2212.12857) Tab. 4). It is sometimes listed as pose-based, but it is **RGB, not pose**.
- **SignBart (arXiv 2506.21592)**: reports WLASL2000 68.00 with 3.8M parameters (Tab. 2). The paper's own tables label results as "Validation", and the comparison mixes P-I and P-C numbers from other papers. Treat it as **not comparable / not reliable**.
- **"Gloss-based WLASL2000" from Sandoval-Castaneda et al. (arXiv 2309.02450)**: Bi-GCN 77.43, MaskFeat 79.02. This is a *re-split* of WLASL (a different label set); on the original WLASL2000 they report MaskFeat 52.83 and cite SAM-SLR 58.73 (Tab. 4).

### 1b. WLASL2000 RGB / fusion reference

| Method | P-I T1 | P-C T1 | Source |
|---|---|---|---|
| Uni-Sign pose+RGB | 63.52 | 61.32 | 2501.15187 Tab. 3 |
| NLA-SLR (RGB+heatmap, 3-crop) | 61.26 | 58.31 | 2303.12080 Tab. 2 |
| NLA-SLR (single crop) | 61.05 | 58.05 | 2303.12080 Tab. 2 |
| StepNet (R+F) | 61.17 | 58.43 | 2212.12857 Tab. 5 |
| SAM-SLR-v2 (7 modalities, 5-crop) | 59.39 | 56.63 | 2110.06161 Tab. VIII |
| Zhao et al. TIP (+R) | 58.06 | 55.66 | 2406.10501 Tab. III |
| MASA (+R) | 55.77 | 53.13 | 2405.20666 Tab. II |
| SignBERT+ (+R) | 55.59 | 53.33 | 2305.04868 Tab. XIII |
| I3D baseline | 32.48 | – | 1910.11006 Tab. 3 |

### 1c. WLASL100 / WLASL300, pose-only (P-I / P-C top-1)

| Method | WL100 P-I | WL100 P-C | WL300 P-I | WL300 P-C | Source |
|---|---|---|---|---|---|
| Uni-Sign pose-only | 92.24 | 92.75 | 88.17 | 88.69 | 2501.15187 Tab. 18 (appendix) |
| MSLU | 88.76 | 89.25 | 82.04 | 82.71 | 2408.08544 Tab. V |
| Zhao et al. TIP | 86.43 | 87.13 | 77.84 | 78.42 | 2406.10501 Tab. III |
| MASA | 83.72 | 84.47 | 73.65 | 74.33 | 2405.20666 Tab. II |
| SignBERT+ | 79.84 | 80.72 | 73.20 | 73.77 | 2305.04868 Tab. XIII |
| BEST | 77.91 | 77.83 | 67.66 | 68.31 | 2302.05075 Tab. 3 |
| SignBERT (H) | 76.36 | 77.68 | 62.72 | 63.43 | 2110.05382 Tab. 9 |
| SPOTER (WACV-W 2022) | 63.18 | – | 43.78 | – | [CVF PDF](https://openaccess.thecvf.com/content/WACV2022W/HADCV/papers/Bohacek_Sign_Pose-Based_Transformer_for_Word-Level_Sign_Language_Recognition_WACVW_2022_paper.pdf) Tab. 4. The paper calls this "top-1 macro average", i.e. per-class-like |
| SPOTER + MediaPipe (CVPR-W 2022 ext. abstract) | 78.29 (WL100 test) | – | – | – | [arXiv 2210.00893](https://arxiv.org/abs/2210.00893) Tab. 1 |
| Siformer (ACM MM 2024) | 86.50 | – | – | – | [arXiv 2503.20436](https://arxiv.org/abs/2503.20436) Tab. 6. **Non-standard test split** (8 random samples per class); their re-run of SPOTER gives 58.52 |
| Pose-TGCN | 55.43 | – | 38.32 | – | 1910.11006 Tab. 3 |
| Pose-GRU | 46.51 | – | 33.68 | – | 1910.11006 Tab. 3 |

---

## 2. AUTSL (official signer-independent test set, 226 classes, top-1)

| Method | Modality | Test top-1 | Notes | Source |
|---|---|---|---|---|
| SL-GCN 4-stream, 3D keypoints (SAM-SLR-v2) | skeleton only | **96.53** | HRNet 2D + depth-lifted 3D | 2110.06161 Tab. IV |
| SL-GCN 4-stream, 2D keypoints (SAM-SLR-v2) | skeleton only | 96.47 | HRNet whole-body, 27 nodes | 2110.06161 Tab. IV |
| "Skel 3D" ensemble (K3 + SSTCN features) | keypoints + HRNet feature maps (image-derived) | 97.01 | not purely coordinate-based | 2110.06161 Tab. V |
| SL-GCN 4-stream (SAM-SLR v1), **validation** set | skeleton only | 95.45 (val) | | [arXiv 2103.08833](https://arxiv.org/abs/2103.08833) Tab. 2 |
| SL-TSSI-DenseNet | skeleton (MediaPipe) | 93.13 | MediaPipe-based, so a closer extractor match to ours | [arXiv 2304.05403](https://arxiv.org/abs/2304.05403) Tab. 3 |
| OpenHands SL-GCN | skeleton (MediaPipe) | 91.9 | | 2110.05877 Tab. 2 |
| SAM-SLR-v2 RGB-D (all modalities, fine-tuned with val) | RGB+D+pose | **98.53** | best overall (ChaLearn 2021 winner) | 2110.06161 Tab. VI |
| SAM-SLR-v2 RGB (fine-tuned with val) | RGB+pose | 98.42 | | 2110.06161 Tab. VI |
| SAM-SLR-v2 RGB-D, no val fine-tune | RGB+D+pose | 98.10 | | 2110.06161 Tab. VI |
| AUTSL dataset baseline (CNN+FPM+BLSTM+Att.) | RGB-D / RGB | 62.02 / 49.22 | | [arXiv 2008.00932](https://arxiv.org/abs/2008.00932) Tab. 3/4 |

I found no skeleton results on AUTSL for SignBERT+, BEST, MASA or Zhao TIP; none of them evaluate on AUTSL. Note that AUTSL's official split is already signer-independent: 31 train, 5 val and 7 test signers (2103.08833 Tab. 1).

---

## 3. LSA64 (64 classes, 10 signers × 5 repetitions)

**LSA64 has no official split.** The dataset paper ([arXiv 2310.17429](https://arxiv.org/abs/2310.17429), Sec. 4) uses 30 runs of a stratified random 80/20 split, which is signer-dependent. Almost every "~100%" number uses a random 80/20 split. **Our signer-independent split (signers 1–8 train, 9–10 test) is not directly comparable to any of them.**

| Method | Modality | Top-1 | Protocol | Source |
|---|---|---|---|---|
| SPOTER | pose (Apple Vision API) | 100.00 ± 0 | random 80/20, stratified; "*" = 5-repetition CV | SPOTER CVF PDF Tab. 5 + text |
| Siformer | pose (SPOTER keypoints) | 99.84 | random 80/20 (their SPOTER re-run gives 99.52) | 2503.20436 Tab. 6 + Sec. 4.1 |
| SSL-SLR (arXiv 2509.05188) | video SSL (modality not confirmed) | 99.07 | **protocol unverified** | [arXiv 2509.05188](https://arxiv.org/abs/2509.05188) Tab. 6 |
| LSTM+DSC (Konstantinidis 2018a) | skeleton | 99.84 ± 0.19 per the SPOTER table; 92.15 per the Siformer table | random (as cited) | secondary (SPOTER Tab. 5 / Siformer Tab. 6). **The two secondary sources disagree**, so treat as unverified |
| I3D | RGB | 98.91 | random | secondary (SPOTER Tab. 5) |
| OpenHands SL-GCN | pose (MediaPipe) | 97.8 | OpenHands split (**unverified**) | 2110.05877 Tab. 2 |
| SignBart | pose | 96.04 | reported on "validation" | 2506.21592 Tab. 4 |
| Dataset baseline (HMM-GMM) | hand tracking (gloves) | 95.95 ± 0.95 | 30 × random 80/20 | 2310.17429 Sec. 4 |
| **Signer-independent results** | | | | |
| Alyami et al., pose Transformer (ACM TALLIP 23(1), 2024) | pose (MediaPipe) | **91.09** SI / 98.25 SD | SI protocol (which signers were held out) **unverified**; numbers from the authors' [GitHub README](https://github.com/snalyami/Pose_SLR), DOI 10.1145/3584984 | README (primary PDF not accessed) |
| Marais et al. 2022 (InceptionV3-GRU) | RGB | 74.22 SI / 97.03 SD | search snippets say leave-one-signer-out (**unverified**) | **unverified** (ResearchGate / Rhodes repository blocked) |

---

## 4. ASL Citizen (official split; val/test signers unseen in training)

The metric is Rec@1, which equals top-1 accuracy over 2,731 classes. The official test set has unseen users: "we establish splits such that users in the validation and test sets are unseen during training" ([arXiv 2304.05934](https://arxiv.org/abs/2304.05934) Sec. 3).

| Method | Input | Rec@1 | Rec@5 | Notes | Source |
|---|---|---|---|---|---|
| ST-GCN (dataset baseline) | pose: MediaPipe Holistic, 27 kp (OpenHands set) | 0.5952 | 0.8268 | Adam 1e-3, cosine, ≤75 epochs, shear/rotation aug. | 2304.05934 Tab. 5 |
| I3D (dataset baseline) | RGB | 0.6310 | 0.8609 | | 2304.05934 Tab. 5 |
| SignCLIP | pose (MediaPipe) | 0.60 | 0.84 | | secondary (SHuBERT Tab. 2) |
| SHuBERT, rank-1 LoRA | hand/face crops + pose | 0.65 | 0.87 | 86M parameters, SSL on 984 h | 2411.16765 Tab. 2 |
| SignDEG, full fine-tuning | RGB (DINOv3 crops) | 0.704 | 0.89 | | arXiv 2609.06296 Tab. 3 |
| **PoseFormer** | **pose (MediaPipe)** | **0.751** | 0.932 | strongest pose-only result found | 2502.20171 Tab. 2 |
| SignBart | pose | 75.22% | – | labelled "Validation" in their tables; protocol unclear | 2506.21592 Tab. 5 |
| SignRep | RGB | 0.8137 | 0.9611 | | 2503.08529 Tab. 3 |

---

## 5. Continuous SLR: PHOENIX-2014 and PHOENIX-2014T (WER %, lower is better)

### 5a. Pose/keypoint-only CSLR

| Method | Keypoints | Pre-training | P14 dev | P14 test | P14T dev | P14T test | Source |
|---|---|---|---|---|---|---|---|
| CoSign-2s (ICCV 2023) | MMPose 2D + conf., **77 kp**, 5 groups; skeleton + motion streams | none | **19.7** | **20.1** | **19.5** | **20.1** | [CVF PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Jiao_CoSign_Exploring_Co-occurrence_Signals_in_Skeleton-based_Continuous_Sign_Language_Recognition_ICCV_2023_paper.pdf) Tab. 1 |
| MSKA-SLR (arXiv 2405.05672; journal version **unverified**) | HRNet 79 kp, 4 streams | none | 21.7 | 22.1 | 20.1 | 20.5 | [arXiv 2405.05672](https://arxiv.org/abs/2405.05672) Tab. 1 |
| CoSign-1s | 77 kp single stream | none | 20.9 | 21.2 | 20.4 | 20.6 | CoSign Tab. 1 |
| MSLU | MMPose | SL-1.5M multimodal | 21.2 | 21.2 | 20.1 | 21.3 | 2408.08544 Tab. VIII |
| STARK (arXiv 2603.16163, 2026) | HRNet 79 kp; 3M-param encoder | none | – | – | 21.0 | 21.9 | [arXiv 2603.16163](https://arxiv.org/abs/2603.16163) text, Sec. 4. Its Tab. 1 is mis-aligned, so I used only the text |
| CoSign baseline (all kp, 1 GCN) | 77 kp | none | 24.3 | 24.4 | 23.6 | 23.8 | CoSign Tab. 1 |
| TwoStream-SLR, keypoint branch only (NeurIPS 2022) | HRNet heatmaps (79 kp), S3D | Kinetics-400 init | 28.56 | 28.00 | 27.14 | 27.19 | [arXiv 2211.01367](https://arxiv.org/abs/2211.01367) Tab. 7 / Tab. 3 |
| SignBERT+ (pose, fine-tuned) | MMPose 133 kp | 230k videos SSL | 34.0 | 34.1 | 32.9 | 33.6 | 2305.04868 Tab. XVI/XVII |
| SignBERT+, from scratch (same backbone) | | none | 43.6 | 43.4 | 42.2 | 42.6 | 2305.04868 Tab. XI |
| P-BLSTM / P-Trans (older pose baselines) | | | 39.6 / 40.9 | 39.3 / 40.4 | 40.2 / 39.4 | 40.2 / 39.8 | 2305.04868 Tab. XVI/XVII |

Signer-independent PHOENIX14-SI (CoSign Tab. 3): CoSign-2s 31.1 / 29.6 and CoSign-1s 32.1 / 31.8 (dev / test). The best video method in the same table is RadialCTC at 33.8 / 32.2.

**Frozen-encoder CSLR:** I found **no published PHOENIX result that uses a frozen pose SSL encoder plus a CTC head.** SignBERT+, MSLU and Uni-Sign all fine-tune the encoder end to end. The nearest analogues are frozen-feature results for *translation*: SHuBERT Tab. 7, where frozen vs fine-tuned gives BLEU 4.7 vs 7.5 in its ablation, and SignDEG Tab. 20. A frozen-encoder CTC protocol is therefore a novel, stricter setting; it should be presented as a representation probe, not as a SOTA claim.

### 5b. RGB / fusion reference (best published)

| Method | Input | P14 dev | P14 test | P14T dev | P14T test | Source |
|---|---|---|---|---|---|---|
| C2ST (ICCV 2023) | RGB (+LM) | 17.5 | 17.7 | 17.3 | 18.9 | *(secondary: MSLU Tab. VIII)* |
| SlowFastSign (arXiv 2309.12304) | RGB | 18.0 | 18.3 | 17.7 | 18.7 | [arXiv 2309.12304](https://arxiv.org/abs/2309.12304) Tab. 1 |
| TwoStream-SLR | RGB + keypoints | 18.4 | 18.8 | 17.7 | 19.3 | 2211.01367 Tab. 1 |
| CorrNet (CVPR 2023) | RGB | 18.8 | 19.4 | 18.9 | 20.5 | 2309.12304 Tab. 1 (secondary) |
| SignBERT+ (+R) | pose + RGB | 19.9 | 20.0 | 18.8 | 19.9 | 2305.04868 Tab. XVI/XVII |

---

## 6. Pose-based self-supervised / pre-training methods: summary

| Method | Venue | SSL objective | Pre-training data | Model size | Downstream evaluation | Low-label / probe results |
|---|---|---|---|---|---|---|
| SignBERT | ICCV 2021 | masked joint + frame modeling with a MANO hand-model decoder | ISLR train sets + extra (hand pose, MMPose) | not stated (3-layer Transformer, Tab. 6) | full fine-tuning, 32 frames | pretraining-data ratio 0–100% on MSASL (Tab. 7): MSASL1000 P-I 41.85 → 49.54 |
| SignBERT+ | TPAMI 2023 | joint / frame / clip masking, hand-model-aware | 230,246 videos, MMPose 133 kp | not stated | full fine-tuning, ISLR + CSLR + SLT | data ratio 0–100% (Tab. X): MSASL1000 53.31 → 62.42 |
| BEST | AAAI 2023 | BERT-style masked unit modeling with a coupled hand/body pose tokenizer | ISLR train sets | Transformer, D = 1536, 8 heads | full fine-tuning, Adam 1e-4 | data ratio (Tab. 8): MSASL1000 51.96 → 58.82 |
| MASA | TCSVT 2024 | motion-aware masked AE (90% mask) + semantic alignment | 160,313 samples (4 ISLR train sets), MMPose 49 joints | not stated | full fine-tuning (SGD, 60 epochs) | mask-ratio ablation: 0% mask gives 25.64 P-I on WLASL2000 vs 49.06 (Tab. IX) |
| Zhao et al. (ST-consistency, contrastive) | TIP (arXiv 2406.10501) | contrastive, spatial-temporal consistency + knowledge transfer | 4 ISLR train sets, MMPose 49 joints | not stated | full fine-tuning, **linear probe**, **semi-supervised 20/40/60%** | **Linear probe** WLASL2000 P-I: 32.35 (vs SignBERT+ 24.50, BEST 21.19, SignBERT 18.24). **20% labels** WLASL2000 P-I: 21.02 (SignBERT+ 14.11). 40%: 36.52. 60%: 44.37 (Tab. V, VII) |
| MSLU | arXiv 2408.08544 | masked pose modeling + sign-text contrastive | SL-1.5M (~1,780 h, multilingual) | 8 blocks; d = 1024 / 1536 / 512 | full fine-tuning (finetune-rate 0.1) | data scale 0–100% (Tab. XIII): MSASL1000 62.23 → 74.07, P14 WER 28.9 → 21.2 |
| SHuBERT | ACL 2025 | HuBERT-style masked cluster prediction over 4 streams | YouTube-ASL 984 h | 86M | LoRA rank-1 / full fine-tuning; frozen for SLT | 98 h vs 984 h pretraining (Tab. 5) |
| Uni-Sign (not SSL) | ICLR 2025 | generative sign→text pre-training with mT5 | CSL-News 1,985 h / YouTube-ASL 984 h | 580M total (GCN pose encoder 4.5M) | unified fine-tuning as text generation | data scale on CSL-Daily (Tab. 8) |
| OpenHands | ACL 2022 | DPC / masked / contrastive on pose | Indian SL 1,129 h | SL-GCN 4.9M | fine-tuning + few-shot | WLASL2000: 21.4 → 27.4 with DPC (Tab. 6) |
| SignRep | ICCV 2025 | MAE on video with pose-derived "sign prior" targets | YouTube-SL | Hiera (RGB) | fine-tuning + **frozen retrieval** | RGB at inference |
| S-JEPA / pose-JEPA for sign | – | – | – | – | – | **No sign-language pose JEPA found.** S-JEPA (ECCV 2024) is for action recognition only |

"Sign2Vec" (Dense Predictive Coding sign embeddings) appeared only in a search-engine summary; I could not locate a primary paper. **Unverified, do not cite.**

---

## 7. Positioning notes

### 7.1 Strongest comparable pose-only numbers per benchmark

| Benchmark | Like-for-like SSL target (pose, in-domain pretraining, full fine-tuning) | Strongest pose-only overall | Caveat to state |
|---|---|---|---|
| WLASL2000 P-I / P-C | Zhao et al. TIP **51.98 / 49.46**; MASA 49.06 / 46.91; SignBERT+ 48.85 / 46.37; BEST 46.25 / 43.52 | Uni-Sign pose 63.13 / 60.90 (984 h, text-supervised, 580M parameters); MSLU 56.29 / 53.29 (1.5M pairs) | SL-GCN reaches 51.50 / 48.87 **with no pretraining** (4-stream HRNet). It is the honest "strong supervised pose" baseline and is often left out of SSL papers. |
| WLASL100 / 300 P-I | Zhao TIP 86.43 / 77.84; MASA 83.72 / 73.65 | Uni-Sign 92.24 / 88.17 | Siformer 86.50 (WL100) uses a non-standard split |
| AUTSL test | none (no SSL pose method reports AUTSL) | SL-GCN 4-stream 96.53 (3D) / 96.47 (2D) | the MediaPipe-based SL-TSSI gets 93.13, which is a fairer extractor comparison |
| LSA64 | – | random 80/20: SPOTER 100.0, Siformer 99.84. Signer-independent: Alyami et al. 91.09 (MediaPipe) | our 8/2 signer split is custom. Also report a random-80/20 (or 10-fold) number for compatibility and state the protocol explicitly. |
| ASL Citizen Rec@1 | – | PoseFormer 0.751 (MediaPipe); ST-GCN baseline 0.5952 | SHuBERT 0.65 and SignDEG 0.704 are video-crop methods |
| PHOENIX-2014 test WER | SignBERT+ 34.1 (pose SSL, fine-tuned); MSLU 21.2 | CoSign-2s 20.1; MSKA 22.1; CoSign-1s 21.2 | no frozen-encoder baseline exists. Report a supervised from-scratch encoder with the same CTC head next to ours. |
| PHOENIX-2014T test WER | SignBERT+ 33.6; MSLU 21.3 | CoSign-2s 20.1; MSKA 20.5; CoSign-1s 20.6; STARK 21.9 | same as above |

Notes:
- CoSign uses exactly **77 MMPose keypoints**, so its numbers are the natural CSLR reference for our 77-joint input. Ours uses MediaPipe Holistic 3D, so the extractor still differs and should be noted.
- Most WLASL pose methods use **2D MMPose/HRNet** keypoints. MediaPipe is generally regarded as weaker on hand-hand and hand-face contact; PoseFormer discusses this (2502.20171, Sec. 1). A gap that comes from the extractor is to be expected.
- Report **both P-I and P-C** on WLASL. Mixed-up columns are common: SignBart's table cites P-C values as top-1, and Uni-Sign Tab. 18 has an ST-GCN MSASL row copied from WLASL.
- For low-label claims, the comparable published protocol is Zhao et al. TIP Tab. VII (20/40/60% labels, fine-tuning) and Tab. V (linear probe on 100% labels). A **10%-label result has no published pose-SSL comparator** on WLASL. A 20% point allows direct comparison with their table: SignBERT+ 14.11, BEST 12.25, Zhao 21.02 P-I on WLASL2000.

### 7.2 Published fine-tuning recipes

**WLASL ISLR (pose SSL lineage: SignBERT → BEST → SignBERT+ → MASA → Zhao TIP)**
- Frames: **32 frames, random temporal sampling in training and center sampling at test** (SignBERT+, BEST, MASA; 2305.04868 Sec. IV-A3, 2302.05075 "Training Setup").
- Optimizer:
  - MASA and Zhao TIP: **SGD, momentum 0.9, batch 64, 60 epochs**; Zhao uses lr 0.01.
  - SignBERT+: Adam lr 1e-4.
  - BEST: Adam lr 1e-4, ×0.1 every 10 epochs.
  - MSLU: AdamW lr 1e-3, wd 1e-4, batch 64, 60 epochs, step LR, **backbone lr × 0.1**, **label smoothing 0.2** (Tab. III).
- Spatial augmentation: random rotation, translation and scale of the coordinates. SignBERT+ calls this "random moving augmentation"; BEST uses a random perturbation matrix. SL-GCN adds mirroring, jittering and shifting.
- Head:
  - SHuBERT: temporal mean-pool → BatchNorm → Linear; Adam 1e-4, wd 1e-4, 125 epochs, batch 128, early stopping on val, label smoothing 0.1 (2411.16765 Sec. 4.3).
  - Uni-Sign task-specific baseline: mean-pool + classification head with cross-entropy (Sec. 4.3).
- Optional extras that the strongest numbers use:
  - NLA-SLR: mixup on keypoints, label smoothing, **3-crop test-time averaging**; Adam 1e-3, wd 1e-3, cosine, 100 epochs, batch 32.
  - SAM-SLR: 4-stream ensemble (joint, bone, joint-motion, bone-motion) and **5-clip test averaging**.
  - State any multi-crop testing explicitly.

**PHOENIX CSLR (pose)**
- CoSign:
  - Group-specific GCN → 1D-CNN (conv–pool–conv–pool, 4× temporal downsampling) → 2-layer BiLSTM (2×512).
  - **Two CTC losses**: auxiliary CTC on the 1D-CNN output plus primary CTC.
  - **AdamW lr 4e-4, ÷10 at epochs 20 and 35, 40 epochs, batch 8.**
  - Input is 2D coordinates + confidence; **only augmentation is random temporal scaling ±20%**.
  - Two-stream variant: pretrain each stream for 10 epochs.
- MSKA: Adam lr 1e-3, wd 1e-3, cosine over 100 epochs, batch 8, beam 5. The best augmentation is temporal + rotation; adding translation and scale hurt (Tab. 4a).
- SignBERT+: Adam 1e-4, **randomly sample 80% of frames** in training and all frames at test, beam 10.
- MSLU: AdamW 1e-4, wd 1e-5, batch 8, 40 epochs, step LR, beam search.
- For a frozen-encoder probe: CoSign's light 1D-CNN + BiLSTM + dual-CTC head (and 40-epoch schedule) is the most defensible standard head. Report decoder parameter count next to it; CoSign Tab. 2 gives total 21.4M (1s) / 28.2M (2s).
