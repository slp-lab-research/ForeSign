# Pose-extractor bake-off (spec §2)

*Internal design/record document, lightly edited for the code release; dates are those of the original record.*

- Date: 2026-10-25. Spec: `docs/design/2026-10-25-unified-3d-pose-reextraction-design.md` §2.
- Code: extractor wrappers in `tools/extract/extractors.py`. The bake-off sampling, metric, report and benchmark scripts are not included in this release. Tests: `tests/test_extract_io.py`.
- The sample manifest, raw metric files, per-clip outputs and overlay videos are not included in this release.

> **Outcome (added for the release).** The pre-declared rule below selects RTMW3D-d5. Weighing the
> caveats listed under the decision (clearly worse 3D plausibility of RTMW3D hands, hand detection
> not measured the same way across candidates), MediaPipe Holistic legacy (MP-L) was chosen as the
> single extractor instead, and its hand detection was then improved with a crop step:
> `docs/mp_hand_improvement.md` (production configuration `MP-L-V4V0`,
> `configs/extract/mp_l_v4v0.json`). All released data and results use MP-L-V4V0.

## Decision

**RTMW3D-x (rtmlib ONNX, `rtmw3d-x_8xb64_cocktail14-384x288`, YOLOX-m HumanArt person detector run every 5 frames, largest box).**

The rule was applied twice:

1. **As run** (detector on every frame): RTMW3D processed 20 fps on the shared GPU. That projects to 28.7 days for the held tier-1 data, which is over 3 weeks, so the throughput clause excluded it and **MP-L** won (78.6 vs MP-T 77.3; MP-T was not better on stability).
2. **Production configuration** (detector every 5 frames, box reused in between, `RTMW3D-d5`): 48.8 fps, projecting to 11.8 days. Its quality matches the per-frame run (pooled hand detection 87.0 vs 87.1, jitter and 3D numbers within noise; see the tables). The candidate is kept and has the best hand detection by 8.4 points, so no override is possible. Result: **RTMW3D**.

So the only thing that excluded RTMW3D was how the detector was scheduled, and the model's quality does not depend on it. Read as "would take more than 3 weeks", the rule picks RTMW3D-d5.

**Caveats the pre-declared rule does not weigh:**
- **3D plausibility is clearly worst for RTMW3D.** Pooled finger bone-length CV is 0.41, against 0.21 for MP-L and 0.18 for MP-T. The depth-jump rate is 27 % of frame pairs, against 2.7 % for MP-L and 11 % for MP-T. RTMW3D's z is root-relative metric depth from a single SimCC head. It is reasonable for the body (forearm CV 0.135, against 0.120 and 0.079) but noisy for fingers, except on How2Sign (studio, 1280×720), where finger CV is 0.22. Goal 2 of the spec (3D hands) favours MediaPipe. MP-T has the best 3D: metric world landmarks for the hands too, and the lowest bone CV.
- **Hand detection is not measured the same way.** MediaPipe either returns a hand or it does not. RTMW always outputs 133 joints, so "detected" means a median hand-keypoint score of at least 0.5. At 0.3 its rate rises to 96 %. Even at 0.5 it leads on every dataset, and by a wide margin on ASL Citizen (+13), WLASL (+21) and INCLUDE (+16): webcam clips where the hands are low or motion-blurred. The overlays confirm that RTMW places the hands where MediaPipe returns none.
- **The fingerprint probe does not separate the candidates on raw output** (0.95–0.97 against a chance level of 0.33). Raw coordinates still carry resolution and framing, so the probe needs the §5 canonicalisation before it can say anything.
- The GPU was shared with other jobs throughout (device utilisation 45–97 % while co-resident), so the RTMW3D fps figures are conservative.

## Setup

| | |
|---|---|
| Sample | 40 clips/dataset, `numpy.random.default_rng(0).choice` over the sorted candidate list, 8 datasets, 320 clips, 43,716 frames. All candidates ran on the same files |
| Datasets | PHOENIX-2014T (frame folders 210×260, 25 fps, all splits), AUTSL (`*_color`, 512², 30), ASL Citizen (640×480, 29.9), WLASL (`wlasl_video_format.zip`, mixed 260×240–1920×1080, 30), INCLUDE (members of the 44 zips, 1920×1080, 25), LSE-UVIGO (all mp4, 1280×720, 25, 53–2047 frames), How2Sign (`h2s_train_clips` archive, 1280×720, 24–60 fps), iSign v1.1 (300²–662×720, 25) |
| MP-L | `mediapipe==0.10.21` `solutions.holistic` (model_complexity 2, refine_face_landmarks, smooth_landmarks, video mode). kp = pose33 \| face478 \| L21 \| R21 normalised, world = pose_world33 |
| MP-T | same wheel, Tasks `HolisticLandmarker` VIDEO mode, `holistic_landmarker.task` float16/latest (sha256 `e2dab611…`). world = pose33 \| L-hand21 \| R-hand21 metric world landmarks |
| RTMW3D | `rtmlib==0.0.16` `Wholebody3d` models on onnxruntime-gpu 1.23.2 (CUDA EP checked; the code refuses to fall back to CPU). kp = COCO-WholeBody 133 (x, y from the 2D head, z = root-relative metres), world = raw 3D output plus a bbox per frame |
| Output | per clip `(T,K,4)` float16 (x, y, z, score) + `world` + `valid (T,)`. Every output's T equals the decoded frame count (0 mismatches, 0 failures, for all candidates) |
| Compute | MediaPipe: 48 cores of a shared CPU server, low priority, 48-process pool. RTMW3D: one GPU shared with other jobs (free memory checked before each launch), one process |

Metric definitions (from the bake-off metrics code, not included in this release) were fixed before any results were looked at:
- **Wrist-in-frame denominator.** A consensus of the three candidates' body wrists, counting only wrists with a score of at least 0.5.
- **Jitter.** Mean 2D acceleration divided by the clip's median shoulder width (shoulder widths per frame²).
- **Swap.** A crossed hand-root assignment that is less than 0.5× the straight one.
- **Bone CV.** Computed on each candidate's best 3D:
  - MP-L: pose_world for the forearms; hands as (x·W, y·H, z·W).
  - MP-T: pose_world and hand world landmarks.
  - RTMW3D: image x,y scaled to metres with a 0.36 m shoulder, plus the model z.
- **Depth jump.** A wrist moving more than 0.5 shoulder widths in z, or a fingertip moving more than 0.5 hand sizes in z relative to the hand root, between consecutive frames.

The decision rule was operationalised in the bake-off report script (not included in this release), also before the results:
- **"Better on stability"** means lower on at least 2 of {hand jitter, face jitter, swap rate}.
- **Tie** means a difference of less than 0.5 points.
- **Throughput exclusion** applies when a candidate's projected time for the held tier-1 frames on its resource is over 21 days.

## Results (as run: MP-L, MP-T, RTMW3D with the detector on every frame)

Pooled = mean over the 8 datasets. Rates are frame-weighted per dataset. Jitter and CV are clip medians. Lower is better for everything except hand detection.

### Pooled (macro mean over 8 datasets)

| candidate | hand det % | hand det % (RTM thr .3) | hand jitter | face jitter | L/R swap % | forearm CV | finger CV | depth-jump % |
|---|---|---|---|---|---|---|---|---|
| MP-L | 78.6 | 78.6 | 0.0461 | 0.0057 | 0.046 | 0.120 | 0.208 | 2.74 |
| MP-T | 77.3 | 77.3 | 0.0481 | 0.0065 | 0.072 | 0.079 | 0.183 | 11.00 |
| RTMW3D | 87.1 | 96.3 | 0.0485 | 0.0071 | 0.000 | 0.135 | 0.413 | 27.72 |

### Fingerprint probe (ASL Citizen / WLASL / How2Sign, 5-fold LR, chance 0.333)

| candidate | accuracy | fold std | clips |
|---|---|---|---|
| MP-L | 0.958 | 0.026 | 120 |
| MP-T | 0.950 | 0.049 | 120 |
| RTMW3D | 0.967 | 0.049 | 120 |

### phoenix14t

| candidate | hand det % | hand det % (RTM thr .3) | hand jitter | face jitter | L/R swap % | forearm CV | finger CV | depth-jump % |
|---|---|---|---|---|---|---|---|---|
| MP-L | 72.0 | 72.0 | 0.0756 | 0.0151 | 0.054 | 0.131 | 0.223 | 3.77 |
| MP-T | 71.9 | 71.9 | 0.0724 | 0.0229 | 0.000 | 0.085 | 0.192 | 15.55 |
| RTMW3D | 73.0 | 89.8 | 0.0816 | 0.0106 | 0.000 | 0.197 | 0.571 | 38.91 |

### autsl

| candidate | hand det % | hand det % (RTM thr .3) | hand jitter | face jitter | L/R swap % | forearm CV | finger CV | depth-jump % |
|---|---|---|---|---|---|---|---|---|
| MP-L | 85.9 | 85.9 | 0.0326 | 0.0037 | 0.059 | 0.094 | 0.194 | 2.39 |
| MP-T | 83.0 | 83.0 | 0.0323 | 0.0038 | 0.064 | 0.081 | 0.187 | 6.38 |
| RTMW3D | 94.1 | 99.8 | 0.0376 | 0.0069 | 0.000 | 0.138 | 0.438 | 42.46 |

### asl_citizen

| candidate | hand det % | hand det % (RTM thr .3) | hand jitter | face jitter | L/R swap % | forearm CV | finger CV | depth-jump % |
|---|---|---|---|---|---|---|---|---|
| MP-L | 59.8 | 59.8 | 0.0461 | 0.0032 | 0.000 | 0.152 | 0.238 | 3.04 |
| MP-T | 57.2 | 57.2 | 0.0540 | 0.0030 | 0.000 | 0.144 | 0.156 | 6.90 |
| RTMW3D | 72.7 | 91.6 | 0.0491 | 0.0055 | 0.000 | 0.180 | 0.423 | 11.18 |

### wlasl

| candidate | hand det % | hand det % (RTM thr .3) | hand jitter | face jitter | L/R swap % | forearm CV | finger CV | depth-jump % |
|---|---|---|---|---|---|---|---|---|
| MP-L | 59.4 | 59.4 | 0.0392 | 0.0030 | 0.000 | 0.114 | 0.184 | 2.51 |
| MP-T | 58.1 | 58.1 | 0.0381 | 0.0028 | 0.102 | 0.074 | 0.175 | 7.67 |
| RTMW3D | 80.1 | 94.6 | 0.0351 | 0.0052 | 0.000 | 0.140 | 0.319 | 13.65 |

### include

| candidate | hand det % | hand det % (RTM thr .3) | hand jitter | face jitter | L/R swap % | forearm CV | finger CV | depth-jump % |
|---|---|---|---|---|---|---|---|---|
| MP-L | 81.0 | 81.0 | 0.0257 | 0.0037 | 0.251 | 0.118 | 0.152 | 2.94 |
| MP-T | 80.1 | 80.1 | 0.0241 | 0.0035 | 0.388 | 0.065 | 0.160 | 7.21 |
| RTMW3D | 96.6 | 99.7 | 0.0410 | 0.0069 | 0.000 | 0.148 | 0.545 | 48.99 |

### lse

| candidate | hand det % | hand det % (RTM thr .3) | hand jitter | face jitter | L/R swap % | forearm CV | finger CV | depth-jump % |
|---|---|---|---|---|---|---|---|---|
| MP-L | 90.9 | 90.9 | 0.0462 | 0.0053 | 0.000 | 0.101 | 0.200 | 1.79 |
| MP-T | 90.4 | 90.4 | 0.0506 | 0.0050 | 0.000 | 0.056 | 0.184 | 14.08 |
| RTMW3D | 93.5 | 96.8 | 0.0479 | 0.0077 | 0.000 | 0.086 | 0.368 | 38.63 |

### how2sign

| candidate | hand det % | hand det % (RTM thr .3) | hand jitter | face jitter | L/R swap % | forearm CV | finger CV | depth-jump % |
|---|---|---|---|---|---|---|---|---|
| MP-L | 91.6 | 91.6 | 0.0474 | 0.0047 | 0.000 | 0.098 | 0.231 | 2.44 |
| MP-T | 90.8 | 90.8 | 0.0546 | 0.0041 | 0.018 | 0.057 | 0.206 | 14.64 |
| RTMW3D | 97.6 | 99.4 | 0.0477 | 0.0065 | 0.000 | 0.084 | 0.224 | 4.26 |

### isign

| candidate | hand det % | hand det % (RTM thr .3) | hand jitter | face jitter | L/R swap % | forearm CV | finger CV | depth-jump % |
|---|---|---|---|---|---|---|---|---|
| MP-L | 88.2 | 88.2 | 0.0558 | 0.0067 | 0.000 | 0.151 | 0.241 | 3.00 |
| MP-T | 86.9 | 86.9 | 0.0588 | 0.0065 | 0.000 | 0.072 | 0.205 | 15.60 |
| RTMW3D | 89.3 | 99.1 | 0.0476 | 0.0077 | 0.000 | 0.106 | 0.417 | 23.65 |

Decision trace as run (3 candidates):
1. MP-L: projected tier-1 time 1.7 d <= 21 d (kept)
2. MP-T: projected tier-1 time 1.0 d <= 21 d (kept)
3. RTMW3D: excluded, projected tier-1 time 28.7 d > 21 d
4. hand detection (pooled): MP-L 78.60, MP-T 77.30 -> best = MP-L
5. MP-T vs MP-L: gap 1.31 pt (<= 2); probe 0.950 vs 0.958 (better); stability wins [] (not better) -> no override
6. no tie within 0.5 pt
7. DECISION: MP-L

## Sensitivity: RTMW3D-d5 (detector every 5 frames; same wrist consensus as above)

| dataset | hand det % | hand det % (thr .3) | hand jitter | face jitter | L/R swap % | forearm CV | finger CV | depth-jump % |
|---|---|---|---|---|---|---|---|---|
| pooled | 87.0 | 96.4 | 0.0471 | 0.0064 | 0.000 | 0.136 | 0.414 | 26.87 |
| phoenix14t | 73.2 | 89.9 | 0.0797 | 0.0100 | 0.000 | 0.193 | 0.574 | 38.05 |
| autsl | 94.0 | 99.8 | 0.0355 | 0.0054 | 0.000 | 0.140 | 0.446 | 41.66 |
| asl_citizen | 72.5 | 91.6 | 0.0464 | 0.0047 | 0.000 | 0.179 | 0.423 | 10.79 |
| wlasl | 79.9 | 94.9 | 0.0341 | 0.0045 | 0.000 | 0.159 | 0.312 | 14.09 |
| include | 96.4 | 99.7 | 0.0398 | 0.0061 | 0.000 | 0.144 | 0.539 | 47.22 |
| lse | 93.6 | 96.8 | 0.0466 | 0.0072 | 0.000 | 0.085 | 0.384 | 35.58 |
| how2sign | 97.5 | 99.4 | 0.0473 | 0.0061 | 0.000 | 0.084 | 0.220 | 4.46 |
| isign | 89.2 | 99.2 | 0.0473 | 0.0073 | 0.000 | 0.106 | 0.414 | 23.15 |

Fingerprint probe: 0.967 ± 0.031 (n = 120).

Decision trace (candidates MP-L, MP-T, RTMW3D-d5):
1. MP-L: projected tier-1 time 1.7 d <= 21 d (kept)
2. MP-T: projected tier-1 time 1.0 d <= 21 d (kept)
3. RTMW3D-d5: projected tier-1 time 11.8 d <= 21 d (kept)
4. hand detection (pooled): MP-L 78.60, MP-T 77.30, RTMW3D-d5 87.05 -> best = RTMW3D-d5
5. MP-L vs RTMW3D-d5: gap 8.44 pt (> 2) -> no override (probe 0.958 vs 0.967 better; stability wins jitter_hand, jitter_face)
6. MP-T vs RTMW3D-d5: gap 9.75 pt (> 2) -> no override
7. DECISION: RTMW3D-d5

## Throughput projection

| candidate | resource | measured | on resource | held tier-1 (49.7 M frames) | tier-1+2 envelope (200–300 M frames) |
|---|---|---|---|---|---|
| MP-L | CPU server, 48 cores | 7.0 fps/core | 336 fps | 1.7 d | 7–10 d |
| MP-T | CPU server, 48 cores | 11.6 fps/core | 555 fps | 1.0 d | 4–6 d |
| RTMW3D (det every frame) | one GPU (shared) | 20.1 fps/GPU | 20 fps | 28.7 d | 115–173 d |
| RTMW3D-d5 | one GPU (shared) | 48.8 fps/GPU | 49 fps | 11.8 d | 47–71 d |

- **MP throughput** is total frames divided by the summed per-clip process time, with 48 processes running at once on 48 cores. The sample's wall clock (119 / 173 fps) is limited by its few 2,047-frame LSE clips at the end of the run. A full-dataset pool stays saturated.
- **RTMW3D component timing** (7,481 frames, shared GPU): decode 1.4 ms, YOLOX-m detector 30.6 ms, RTMW3D-x pose 13.1 ms per frame. Both ONNX models have a fixed batch of 1.
- **Further RTMW3D speed-ups** are possible: a lighter detector, a fixed box for single-signer studio data, TensorRT, or several processes on one GPU. Additional cloud GPU capacity (spec §3) can absorb the tier-2 envelope.
- **Held tier-1 frames** are clip counts × sample mean frames, or the video inventory's frame counts. CISLR and LSA-T use assumed means of 60 and 150. The 200–300 M envelope is taken from the internal video acquisition plan (not included in this release).

## Failures and deviations

- **mediapipe pin.** `pip install mediapipe` gives 1.1.1, which no longer ships `solutions` (legacy holistic). The environment pins `mediapipe==0.10.21`, the last version with both APIs on Python 3.12. The pin brings `opencv-contrib-python 4.11` in place of `opencv-python-headless`. The CPU server had no `python3.12-venv` and no sudo, so the virtual environment was created with `--without-pip` and bootstrapped with get-pip.
- **MP-T crash.** In 0.10.21, `HolisticLandmarker.detect_for_video` aborts the process (`packet.cc Check failed: holder_ != nullptr`) whenever a hand stream is empty. It also discards pose and hands whenever no face is found. `MPTasks._detect` uses the same graph and packets but reads each stream only if it is non-empty. MediaPipe 0.10.35 was tried and fails on the CPU server (`libGLESv2.so.2` is missing).
- **RTMW3D.** Run through rtmlib's ONNX export of the official RTMW3D-x checkpoint, as the spec allows ("mmpose rtmpose3d / rtmlib"), so no mmcv build was needed. The first launch fell back silently to CPU because the pip cuDNN was not on the loader path. It was killed after about 1 min. The wrapper now calls `onnxruntime.preload_dlls()` and asserts that the CUDA EP is active.
- **RTMW3D x,y units.** RTMW3D's x,y are in crop pixels and its z is in root-relative metres. For the 3D metrics, image x,y were converted to metres per clip with an assumed 0.36 m shoulder width. Its bone-CV and depth-jump numbers therefore depend on that convention, while MediaPipe's are native.
- **How2Sign source.** The clips are from the raw `h2s_train_clips` archive (full frame), not from the Uni-Sign signer crop. iSign clips are from `iSign-videos_v1.1`.
- **No truncation.** LSE clips are long (mean 340, max 2,047 frames) and were processed in full.
- **ffmpeg.** The CPU server has no system ffmpeg. Decoding uses OpenCV's bundled FFmpeg on every host.
