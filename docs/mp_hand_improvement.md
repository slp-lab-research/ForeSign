# MP-L hand-detection improvement

*Internal design/record document, lightly edited for the code release; dates are those of the original record.*

- Date: 2026-10-25. Follows `docs/extractor_bakeoff.md`, after which **MediaPipe Holistic legacy (MP-L, mediapipe 0.10.21)** was chosen as the single extractor.
- Same 320-clip / 43,716-frame sample as the bake-off, run on a CPU server only: low priority, 48 cores, 48-process pool. No GPU was used. Peak RSS was about 0.9 GB per worker, 42–44 GB summed, which overcounts shared pages.
- Code: `tools/extract/extractors.py` (`MPL_VARIANTS` and `MPLegacy(variant)`, the crop helpers). The metric additions (false-hand rate, crop share) and the report script (tables, rule, side-by-side overlays) belong to the bake-off tooling, which is not included in this release.
- Raw metric files, per-clip outputs and overlay videos are not included in this release.

## Decision

**MP-L-V4V0 = baseline holistic (thresholds 0.5/0.5, native resolution, smoothing on) plus a MediaPipe Hands crop step for hands that holistic misses.**

- **Detection.** Pooled hand detection rises from 78.6 to **81.2 %** (+2.6 points). It improves on every dataset: PHOENIX +4.0, AUTSL +2.3, ASL Citizen +1.1, WLASL +2.9, INCLUDE +4.0, LSE +1.2, How2Sign +3.2, iSign +2.5.
- **Constraints.** All are met, with the following margins:
  - false-hand rate 0.97 against a limit of 1.72;
  - finger CV 0.213 against 0.238;
  - depth-jump rate 3.31 against 3.74;
  - throughput 7.39 against a floor of 2.42 fps/core.
- **Cost.** Throughput is unchanged within noise, because only about 3 % of detected hands come from the crop. The projected time for the full held tier-1 data (49.7 M frames) on 48 cores is **1.6 days**.

**What did not work.**
- **Thresholds and upscaling (V1, V2, V3) do nothing.** Every variant lands within ±0.05 points of the baseline (78.56–78.60). In legacy Holistic, `min_detection_confidence` and `min_tracking_confidence` gate only the pose detector and pose tracking. Hands come from a pose-derived ROI with a fixed internal presence threshold. Holistic resamples that ROI to the hand model's input size anyway, so a bicubic upscale adds no information.
  - Sanity check: V2 and V3 match V0 and V1 exactly on the native ≥720-px datasets (INCLUDE, LSE, How2Sign).
- **Turning smoothing off (V5)** changes detection by +0.15 points, within noise.

**The gap to RTMW3D remains on ASL Citizen and WLASL** (60.9 / 62.3 against 72.7 / 80.1). On those datasets, 77–78 % of the frames where RTMW3D has a hand and V4V0 does not have an MP pose wrist with visibility below 0.5, so the crop step never fires.

**Post-hoc variant V6.** V6 removes the visibility gate. It was added after the V4V0 results, so it is outside the pre-declared set. It gains only +0.36 points (ASL Citizen 61.7, WLASL 62.8): with the gate removed, the palm detector still fails on these low-visibility hands, which are mostly at the bottom edge or motion-blurred. V6 also meets every constraint. Run through the same rule, it ties V4V0 (within 1 point, same number of components) and would win only on the secondary detection key. The pre-declared decision stays **V4V0**. V6 is a one-line switch (`fb_vis=0.0`, variant `MP-L-V6`) for the extra 0.4 points.

**Caveats the rule does not weigh.**
- **WLASL false-hand rate.** It rises from 0.45 to 1.97 % under V4V0. The pooled figure passes, but this dataset's increase is larger than 1 point.
- **Hand jitter.** It rises from 0.0461 to 0.0498. Crop hands come from a per-frame static model with no temporal smoothing. Jitter is not in the rule.
- **The own-pose false-hand metric cannot see frames with a low-visibility MP wrist.** As a cross-check, `false_hand_rate_rtm` measures the same quantity against RTMW3D's body wrist. It rises from 2.02 to 2.28 %, which agrees in direction and size.
- **Resting hands count as detections.** Some recovered hands are resting hands (in the lap on How2Sign). They are correct detections but carry no signing information.

## Frozen extractor config (MP-L-V4V0, `config_hash 30d5b11c99ac`)

```
mediapipe==0.10.21, mediapipe.solutions.holistic.Holistic(
    static_image_mode=False, model_complexity=2, smooth_landmarks=True,
    enable_segmentation=False, refine_face_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)   # native resolution, no upscale
hand crop step (per frame, per side, only when holistic returned no hand for that side):
    requires pose found, pose wrist & elbow visibility >= 0.5, pose wrist inside the image
    L      = max(|wrist-elbow|_px, 0.6 * shoulder width_px, 8)
    centre = wrist + 0.3 * (wrist - elbow);  square side = 2.0 * L;  zero-padded; resized to 256x256
    mediapipe.solutions.hands.Hands(static_image_mode=True, max_num_hands=2,
                                    model_complexity=1, min_detection_confidence=0.5)
    accept the hand whose wrist is nearest the pose wrist if it is <= 1.0 L away, not nearer
    the other pose wrist, and not within 0.25 L of the other side's hand wrist
    x,y mapped back to full-frame normalised coordinates; z scaled by crop side / image width
output: kp (T,553,4) pose33|face478|L21|R21, world pose_world33, valid (T,),
        hand_src (T,2) uint8: 0 none, 1 holistic, 2 crop step
```
The variant name `MP-L-V4V0` in `tools/extract/extractors.py` selects everything above; the frozen configuration is `configs/extract/mp_l_v4v0.json` (written by `tools/extract/freeze_config.py`).

## Variants and pre-declared rule

| id | change vs V0 | components |
|---|---|---|
| V0 | baseline (bake-off MP-L outputs reused; a re-run in the same batch of runs, `MP-L-V0r`, reproduced them bit for bit: 69,800 hands, 0 differing frames, and gave the timing) | 0 |
| V1 | min_detection_confidence = min_tracking_confidence = 0.3 | 1 |
| V2 | bicubic upscale so the short side is ≥ 720 px (PHOENIX 210×260 → 720×891, AUTSL 512² → 720², ASL Citizen 640×480 → 960×720, small WLASL/iSign) | 1 |
| V3 | V1 + V2 | 2 |
| V4V0 | best of V0–V3 (= V0, see below) + hand crop step | 1 |
| V5 | smooth_landmarks off | 1 |
| V6 *(post hoc)* | V4V0 without the pose-visibility gate | 1 |

**Choosing the V4 base.** The rule applied to V0–V3 picks V0: detection 78.60, against 78.56 / 78.56 / 78.57, and every variant is eligible. A tie within 1 point goes to the fewest components, which is V0. So V4 was run on V0 only.

**Rule:** a variant is eligible if all of the following hold:
- false-hand rate ≤ V0 + 1 point;
- finger bone CV ≤ V0 + 0.03;
- depth-jump rate ≤ V0 + 1 point;
- fps/core ≥ V0 / 3.

Among the eligible variants, the one with the highest pooled hand detection wins. Any variant within 1 point of it counts as a tie, and a tie goes to the variant with the fewest components, then the higher detection.

**Metric definitions.**
- **Hand detection, jitter, swap, bone CV, depth jump:** as in the bake-off. The wrist-in-frame denominator is the same consensus of the bake-off MP-L, MP-T and RTMW3D body wrists, so it is identical for every variant and the numbers compare directly with the bake-off tables.
- **False-hand rate:** the share of detected hands whose hand-wrist landmark is more than 0.5 forearm lengths (2D, clip median for that side) from the pose wrist. Only frames where pose wrist and elbow both have score ≥ 0.5 are counted.
- **Crop share:** the share of detected hands that came from the crop step.

**Throughput.** All variants were timed in the same batch of runs at 48 workers, as summed per-clip process time. The machine was shared, with a load average of 25–60 from other jobs, so differences of ±5 % are noise. V0 timed now gives 7.26 fps/core, against 7.0 in the bake-off.

## Results

### Pooled (macro mean over 8 datasets)

| variant | hand det % | false-hand % | false-hand % (RTMW ref) | crop share % | L/R swap % | hand jitter | face jitter | forearm CV | finger CV | depth-jump % | fps/core | tier-1 days @48 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MP-L | 78.6 | 0.72 | 2.02 | 0.0 | 0.046 | 0.0461 | 0.0057 | 0.120 | 0.208 | 2.74 | 7.26 | 1.7 |
| MP-L-V1 | 78.6 | 0.73 | 2.04 | 0.0 | 0.026 | 0.0465 | 0.0057 | 0.119 | 0.208 | 2.79 | 7.52 | 1.6 |
| MP-L-V2 | 78.6 | 0.75 | 2.01 | 0.0 | 0.031 | 0.0461 | 0.0058 | 0.120 | 0.205 | 2.68 | 7.51 | 1.6 |
| MP-L-V3 | 78.6 | 0.81 | 2.00 | 0.0 | 0.046 | 0.0468 | 0.0058 | 0.120 | 0.210 | 2.69 | 7.56 | 1.6 |
| MP-L-V5 | 78.7 | 0.78 | 2.02 | 0.0 | 0.061 | 0.0470 | 0.0057 | 0.118 | 0.209 | 2.94 | 7.75 | 1.5 |
| MP-L-V4V0 | 81.2 | 0.97 | 2.28 | 3.3 | 0.042 | 0.0498 | 0.0057 | 0.120 | 0.213 | 3.31 | 7.39 | 1.6 |
| MP-L-V6 | 81.6 | 0.97 | 2.31 | 3.8 | 0.042 | 0.0506 | 0.0057 | 0.120 | 0.213 | 3.41 | 7.38 | 1.6 |
| *MP-T (bake-off ref)* | 77.3 | 1.58 | 2.76 | 0.0 | 0.072 | 0.0481 | 0.0065 | 0.079 | 0.183 | 11.00 | | |
| *RTMW3D (bake-off ref)* | 87.1 | 0.10 | 0.10 | 0.0 | 0.000 | 0.0485 | 0.0071 | 0.135 | 0.413 | 27.72 | | |

### Hand detection % per dataset

| variant | phoenix14t | autsl | asl_citizen | wlasl | include | lse | how2sign | isign |
|---|---|---|---|---|---|---|---|---|
| MP-L | 72.0 | 85.9 | 59.8 | 59.4 | 81.0 | 90.9 | 91.6 | 88.2 |
| MP-L-V1 | 72.0 | 85.8 | 59.8 | 59.3 | 80.9 | 91.0 | 91.7 | 88.0 |
| MP-L-V2 | 71.7 | 85.8 | 59.8 | 59.5 | 81.0 | 90.9 | 91.6 | 88.1 |
| MP-L-V3 | 71.9 | 85.8 | 59.7 | 59.5 | 80.9 | 91.0 | 91.7 | 88.1 |
| MP-L-V5 | 72.4 | 86.0 | 60.0 | 59.3 | 81.5 | 90.9 | 91.8 | 88.0 |
| MP-L-V4V0 | 76.0 | 88.2 | 60.9 | 62.3 | 85.0 | 92.1 | 94.8 | 90.7 |
| MP-L-V6 | 77.2 | 88.2 | 61.7 | 62.8 | 85.0 | 92.2 | 94.8 | 90.9 |
| RTMW3D | 73.0 | 94.1 | 72.7 | 80.1 | 96.6 | 93.5 | 97.6 | 89.3 |

### phoenix14t

| variant | hand det % | false-hand % | false-hand % (RTMW ref) | crop share % | L/R swap % | hand jitter | face jitter | forearm CV | finger CV | depth-jump % |
|---|---|---|---|---|---|---|---|---|---|---|
| MP-L | 72.0 | 0.46 | 1.62 | 0.0 | 0.054 | 0.0756 | 0.0151 | 0.131 | 0.223 | 3.77 |
| MP-L-V1 | 72.0 | 0.42 | 1.58 | 0.0 | 0.054 | 0.0735 | 0.0151 | 0.128 | 0.222 | 3.89 |
| MP-L-V2 | 71.7 | 0.46 | 1.45 | 0.0 | 0.000 | 0.0718 | 0.0158 | 0.134 | 0.223 | 3.77 |
| MP-L-V3 | 71.9 | 0.54 | 1.48 | 0.0 | 0.000 | 0.0769 | 0.0159 | 0.132 | 0.222 | 3.72 |
| MP-L-V5 | 72.4 | 0.46 | 1.57 | 0.0 | 0.054 | 0.0755 | 0.0151 | 0.129 | 0.220 | 3.69 |
| MP-L-V4V0 | 76.0 | 0.66 | 1.87 | 5.2 | 0.048 | 0.0827 | 0.0151 | 0.131 | 0.229 | 4.78 |
| MP-L-V6 | 77.2 | 0.66 | 1.88 | 6.8 | 0.047 | 0.0861 | 0.0151 | 0.131 | 0.233 | 5.06 |

### autsl

| variant | hand det % | false-hand % | false-hand % (RTMW ref) | crop share % | L/R swap % | hand jitter | face jitter | forearm CV | finger CV | depth-jump % |
|---|---|---|---|---|---|---|---|---|---|---|
| MP-L | 85.9 | 0.94 | 2.11 | 0.0 | 0.059 | 0.0326 | 0.0037 | 0.094 | 0.194 | 2.39 |
| MP-L-V1 | 85.8 | 0.75 | 1.92 | 0.0 | 0.059 | 0.0303 | 0.0037 | 0.095 | 0.195 | 2.48 |
| MP-L-V2 | 85.8 | 0.82 | 1.95 | 0.0 | 0.000 | 0.0322 | 0.0037 | 0.095 | 0.193 | 2.35 |
| MP-L-V3 | 85.8 | 0.68 | 1.61 | 0.0 | 0.177 | 0.0330 | 0.0037 | 0.096 | 0.191 | 2.44 |
| MP-L-V5 | 86.0 | 1.21 | 1.99 | 0.0 | 0.059 | 0.0298 | 0.0037 | 0.091 | 0.191 | 2.77 |
| MP-L-V4V0 | 88.2 | 1.10 | 2.25 | 2.6 | 0.056 | 0.0345 | 0.0037 | 0.094 | 0.194 | 2.73 |
| MP-L-V6 | 88.2 | 1.10 | 2.25 | 2.6 | 0.056 | 0.0345 | 0.0037 | 0.094 | 0.194 | 2.73 |

### asl_citizen

| variant | hand det % | false-hand % | false-hand % (RTMW ref) | crop share % | L/R swap % | hand jitter | face jitter | forearm CV | finger CV | depth-jump % |
|---|---|---|---|---|---|---|---|---|---|---|
| MP-L | 59.8 | 3.19 | 9.25 | 0.0 | 0.000 | 0.0461 | 0.0032 | 0.152 | 0.238 | 3.04 |
| MP-L-V1 | 59.8 | 3.45 | 9.61 | 0.0 | 0.000 | 0.0506 | 0.0032 | 0.145 | 0.230 | 2.95 |
| MP-L-V2 | 59.8 | 3.65 | 9.56 | 0.0 | 0.000 | 0.0492 | 0.0032 | 0.149 | 0.223 | 2.77 |
| MP-L-V3 | 59.7 | 4.08 | 9.76 | 0.0 | 0.000 | 0.0480 | 0.0032 | 0.148 | 0.253 | 3.10 |
| MP-L-V5 | 60.0 | 3.24 | 9.08 | 0.0 | 0.000 | 0.0480 | 0.0032 | 0.151 | 0.245 | 3.98 |
| MP-L-V4V0 | 60.9 | 3.16 | 9.61 | 1.8 | 0.000 | 0.0488 | 0.0032 | 0.152 | 0.242 | 3.34 |
| MP-L-V6 | 61.7 | 3.16 | 9.72 | 3.1 | 0.000 | 0.0508 | 0.0032 | 0.152 | 0.247 | 3.71 |

### wlasl

| variant | hand det % | false-hand % | false-hand % (RTMW ref) | crop share % | L/R swap % | hand jitter | face jitter | forearm CV | finger CV | depth-jump % |
|---|---|---|---|---|---|---|---|---|---|---|
| MP-L | 59.4 | 0.45 | 1.14 | 0.0 | 0.000 | 0.0392 | 0.0030 | 0.114 | 0.184 | 2.51 |
| MP-L-V1 | 59.3 | 0.45 | 1.25 | 0.0 | 0.000 | 0.0399 | 0.0029 | 0.112 | 0.187 | 2.58 |
| MP-L-V2 | 59.5 | 0.40 | 1.24 | 0.0 | 0.000 | 0.0399 | 0.0029 | 0.112 | 0.181 | 2.83 |
| MP-L-V3 | 59.5 | 0.48 | 1.24 | 0.0 | 0.095 | 0.0402 | 0.0029 | 0.112 | 0.187 | 2.54 |
| MP-L-V5 | 59.3 | 0.50 | 1.42 | 0.0 | 0.000 | 0.0413 | 0.0030 | 0.111 | 0.191 | 2.44 |
| MP-L-V4V0 | 62.3 | 1.97 | 2.34 | 4.6 | 0.000 | 0.0421 | 0.0030 | 0.114 | 0.188 | 3.19 |
| MP-L-V6 | 62.8 | 1.97 | 2.42 | 5.4 | 0.000 | 0.0427 | 0.0030 | 0.114 | 0.187 | 3.30 |

### include

| variant | hand det % | false-hand % | false-hand % (RTMW ref) | crop share % | L/R swap % | hand jitter | face jitter | forearm CV | finger CV | depth-jump % |
|---|---|---|---|---|---|---|---|---|---|---|
| MP-L | 81.0 | 0.15 | 0.29 | 0.0 | 0.251 | 0.0257 | 0.0037 | 0.118 | 0.152 | 2.94 |
| MP-L-V1 | 80.9 | 0.17 | 0.27 | 0.0 | 0.063 | 0.0260 | 0.0037 | 0.118 | 0.152 | 3.02 |
| MP-L-V2 | 81.0 | 0.15 | 0.29 | 0.0 | 0.251 | 0.0257 | 0.0037 | 0.118 | 0.152 | 2.94 |
| MP-L-V3 | 80.9 | 0.17 | 0.27 | 0.0 | 0.063 | 0.0260 | 0.0037 | 0.118 | 0.152 | 3.02 |
| MP-L-V5 | 81.5 | 0.17 | 0.32 | 0.0 | 0.374 | 0.0254 | 0.0036 | 0.115 | 0.153 | 2.98 |
| MP-L-V4V0 | 85.0 | 0.16 | 0.33 | 4.6 | 0.230 | 0.0291 | 0.0037 | 0.118 | 0.169 | 3.79 |
| MP-L-V6 | 85.0 | 0.16 | 0.33 | 4.6 | 0.230 | 0.0291 | 0.0037 | 0.118 | 0.169 | 3.79 |

### lse

| variant | hand det % | false-hand % | false-hand % (RTMW ref) | crop share % | L/R swap % | hand jitter | face jitter | forearm CV | finger CV | depth-jump % |
|---|---|---|---|---|---|---|---|---|---|---|
| MP-L | 90.9 | 0.15 | 0.26 | 0.0 | 0.000 | 0.0462 | 0.0053 | 0.101 | 0.200 | 1.79 |
| MP-L-V1 | 91.0 | 0.15 | 0.26 | 0.0 | 0.000 | 0.0472 | 0.0053 | 0.100 | 0.202 | 1.79 |
| MP-L-V2 | 90.9 | 0.15 | 0.26 | 0.0 | 0.000 | 0.0462 | 0.0053 | 0.101 | 0.200 | 1.76 |
| MP-L-V3 | 91.0 | 0.15 | 0.26 | 0.0 | 0.000 | 0.0472 | 0.0053 | 0.100 | 0.202 | 1.76 |
| MP-L-V5 | 90.9 | 0.18 | 0.30 | 0.0 | 0.000 | 0.0483 | 0.0053 | 0.098 | 0.203 | 1.76 |
| MP-L-V4V0 | 92.1 | 0.17 | 0.28 | 1.2 | 0.000 | 0.0484 | 0.0053 | 0.101 | 0.203 | 2.25 |
| MP-L-V6 | 92.2 | 0.17 | 0.28 | 1.4 | 0.000 | 0.0484 | 0.0053 | 0.101 | 0.203 | 2.30 |

### how2sign

| variant | hand det % | false-hand % | false-hand % (RTMW ref) | crop share % | L/R swap % | hand jitter | face jitter | forearm CV | finger CV | depth-jump % |
|---|---|---|---|---|---|---|---|---|---|---|
| MP-L | 91.6 | 0.11 | 0.52 | 0.0 | 0.000 | 0.0474 | 0.0047 | 0.098 | 0.231 | 2.44 |
| MP-L-V1 | 91.7 | 0.10 | 0.52 | 0.0 | 0.035 | 0.0483 | 0.0046 | 0.099 | 0.230 | 2.53 |
| MP-L-V2 | 91.6 | 0.11 | 0.52 | 0.0 | 0.000 | 0.0474 | 0.0047 | 0.098 | 0.231 | 2.44 |
| MP-L-V3 | 91.7 | 0.10 | 0.52 | 0.0 | 0.035 | 0.0483 | 0.0046 | 0.099 | 0.230 | 2.53 |
| MP-L-V5 | 91.8 | 0.20 | 0.58 | 0.0 | 0.000 | 0.0527 | 0.0047 | 0.098 | 0.230 | 2.85 |
| MP-L-V4V0 | 94.8 | 0.17 | 0.54 | 3.3 | 0.000 | 0.0539 | 0.0047 | 0.098 | 0.231 | 2.93 |
| MP-L-V6 | 94.8 | 0.17 | 0.54 | 3.4 | 0.000 | 0.0539 | 0.0047 | 0.098 | 0.231 | 2.95 |

### isign

| variant | hand det % | false-hand % | false-hand % (RTMW ref) | crop share % | L/R swap % | hand jitter | face jitter | forearm CV | finger CV | depth-jump % |
|---|---|---|---|---|---|---|---|---|---|---|
| MP-L | 88.2 | 0.32 | 0.94 | 0.0 | 0.000 | 0.0558 | 0.0067 | 0.151 | 0.241 | 3.00 |
| MP-L-V1 | 88.0 | 0.32 | 0.90 | 0.0 | 0.000 | 0.0565 | 0.0067 | 0.152 | 0.242 | 3.07 |
| MP-L-V2 | 88.1 | 0.30 | 0.83 | 0.0 | 0.000 | 0.0564 | 0.0068 | 0.151 | 0.239 | 2.55 |
| MP-L-V3 | 88.1 | 0.32 | 0.84 | 0.0 | 0.000 | 0.0550 | 0.0068 | 0.152 | 0.240 | 2.43 |
| MP-L-V5 | 88.0 | 0.27 | 0.87 | 0.0 | 0.000 | 0.0547 | 0.0067 | 0.148 | 0.240 | 3.02 |
| MP-L-V4V0 | 90.7 | 0.37 | 1.00 | 2.8 | 0.000 | 0.0590 | 0.0067 | 0.151 | 0.244 | 3.46 |
| MP-L-V6 | 90.9 | 0.37 | 1.03 | 3.1 | 0.000 | 0.0590 | 0.0067 | 0.151 | 0.244 | 3.48 |

Held tier-1 frames (estimate): 49.7 M

### Rule trace (pre-declared variants)

1. MP-L: eligible (det 78.60, false-hand 0.72, finger CV 0.208, depth-jump 2.74, 7.26 fps/core)
2. MP-L-V1: eligible (det 78.56, false-hand 0.73, finger CV 0.208, depth-jump 2.79, 7.52 fps/core)
3. MP-L-V2: eligible (det 78.56, false-hand 0.75, finger CV 0.205, depth-jump 2.68, 7.51 fps/core)
4. MP-L-V3: eligible (det 78.57, false-hand 0.81, finger CV 0.210, depth-jump 2.69, 7.56 fps/core)
5. MP-L-V5: eligible (det 78.75, false-hand 0.78, finger CV 0.209, depth-jump 2.94, 7.75 fps/core)
6. MP-L-V4V0: eligible (det 81.24, false-hand 0.97, finger CV 0.213, depth-jump 3.31, 7.39 fps/core)
7. highest pooled hand detection among eligible: MP-L-V4V0 81.24
8. within 1 pt: ['MP-L-V4V0'] -> fewest components -> MP-L-V4V0
9. DECISION: MP-L-V4V0

### Sensitivity: same rule with the post-hoc variants added

1. MP-L: eligible (det 78.60, false-hand 0.72, finger CV 0.208, depth-jump 2.74, 7.26 fps/core)
2. MP-L-V1: eligible (det 78.56, false-hand 0.73, finger CV 0.208, depth-jump 2.79, 7.52 fps/core)
3. MP-L-V2: eligible (det 78.56, false-hand 0.75, finger CV 0.205, depth-jump 2.68, 7.51 fps/core)
4. MP-L-V3: eligible (det 78.57, false-hand 0.81, finger CV 0.210, depth-jump 2.69, 7.56 fps/core)
5. MP-L-V5: eligible (det 78.75, false-hand 0.78, finger CV 0.209, depth-jump 2.94, 7.75 fps/core)
6. MP-L-V4V0: eligible (det 81.24, false-hand 0.97, finger CV 0.213, depth-jump 3.31, 7.39 fps/core)
7. MP-L-V6: eligible (det 81.60, false-hand 0.97, finger CV 0.213, depth-jump 3.41, 7.38 fps/core)
8. highest pooled hand detection among eligible: MP-L-V6 81.60
9. within 1 pt: ['MP-L-V4V0', 'MP-L-V6'] -> fewest components -> MP-L-V6
10. DECISION: MP-L-V6

## Failures and deviations

- **No failures.** All 7 runs completed 320/320 clips with T equal to the decoded frame count.
- **Fixed crop geometry.** The constants (`FB_*` in `tools/extract/extractors.py`) were fixed before the V4 run and were not tuned. V6 is the only post-hoc change, and it is reported separately.
- **Overlays** (not included in this release). Each one is side by side: V0 on the left, V4V0 on the right. Crop-step hands are drawn in magenta (left) or cyan (right). For each dataset, the 3 clips with the most frames where V4V0 has a hand and V0 does not were chosen.
