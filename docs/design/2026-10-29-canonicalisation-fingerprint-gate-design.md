# Canonicalisation (canon-v1) and dataset-fingerprint gate — design

*Internal design/record document, lightly edited for the code release; dates are those of the original record.*

- Date: 2026-10-29
- Status: approved (2026-10-29) as the next step of
  `2026-10-25-unified-3d-pose-reextraction-design.md` §5–§6 (that spec fixes the constraints;
  this one fixes the concrete choices).
- Input: the MP-L-V4V0 re-extraction (all datasets verified). Datasets whose extraction outputs
  were temporarily unavailable at the time are added when available.

## 1. Input format (per clip, from `tools/extract/extract_dataset.py`)

`kp (T,553,4)` float16 = pose33 | face478 | lhand21 | rhand21, (x, y) normalised image
coordinates, z = MediaPipe relative depth, 4th = score; `world (T,33,4)` = pose world
landmarks (metres, origin between the hips, camera-aligned axes); `flags (T,3)` = pose found,
left/right hand source (0 none, 1 holistic, 2 crop step); attrs width, height, fps, split,
source. Missing parts are zeros.

## 2. canon-v1 output layout: C77-3D

Same 77-joint order as RESONA-77 (`foresign/groups.py`, so the 18 groups stay valid), but
**3D** (x, y, z) instead of (x, y, conf). "l_" = the signer's left (MediaPipe LEFT_*).

| joints | source |
|---|---|
| body9 0–8: nose, l/r shoulder, l/r elbow, l/r wrist, l/r hip | pose **world** landmarks 0, 11, 12, 13, 14, 15, 16, 23, 24 |
| lhand21 9–29, rhand21 30–50 | hand landmarks (image), re-attached at the body wrist (below) |
| mouth8 51–58 | face mesh outer lip ring 61, 40, 0, 270, 291, 321, 17, 91 |
| face18 59–76 | brows 70, 105, 107, 336, 334, 300; eyes 33, 133, 362, 263; nose 1, 168; contour 234, 454, 172, 397, 152, 10 |

**Re-attachment (hands, face).** Image landmarks are made isotropic in pixels:
`p = (x·W, y·H, z·W)`. A per-clip metres-per-pixel factor
`s = median_t |W_lsh − W_rsh| / median_t |P_lsh − P_rsh|` (world vs image pose shoulders,
frames with pose found) converts them. Hand joint j: `body_wrist_world + s·(p_j − p_handwrist)`.
Face joint j: `world_nose + s·(p_j − p_facenose(1))`. MediaPipe world axes are camera-aligned,
so no rotation is needed between the two.

**Clip frame.** Per clip, over frames with pose found: origin = median shoulder midpoint;
unit = median shoulder width; yaw rotation about the vertical (y) axis so the median shoulder
vector lies on +x in the x–z plane. Applied to all joints and frames. Removes camera distance,
framing, resolution and azimuth; keeps within-clip body motion.

**Time.** Resample to 25 fps (linear on coordinates; nearest on the mask). fps_src kept.

**Validity.** `mask (T,77)` uint8: body from pose found; hands from hand source > 0; face from
face found (any face landmark non-zero). Invalid joints are 0. Extractor scores are dropped
(not a model input — score scales are a known fingerprint).

**Storage.** One HDF5 per dataset (or YouTube batch), one group per clip: `x (T,77,3)` float16,
`mask (T,77)` uint8; attrs fps (25), fps_src, s, scale, split, source, canon_version
"canon-v1", code git sha. Written next to the raw shards (`canon_v1/<ds>.h5`).

## 3. Fingerprint gate (pre-registered before running it on canon-v1)

**Question.** Can a simple classifier tell *which ASL dataset* a clip comes from, using only
content-agnostic statistics? Language identity is held fixed (all ASL), so separability means
recording/extraction artefacts, which pretraining could exploit as a shortcut.

**Data.** ASL Citizen, WLASL, How2Sign (bake-off set; chance 1/3). Secondary: + YouTube-ASL
and OpenASL (5-way, chance 1/5). 300 clips per dataset, `numpy.random.default_rng(0)`,
train split only. From each clip one random **2-second window** (50 frames at 25 fps; clips
shorter than 2 s skipped) so clip length / isolated-vs-continuous cannot be used.

**Versions compared (same clips where the dataset exists in both).**
1. RESONA-77 old data (2D, RESONA-77 shoulder normalisation) — the baseline.
2. new raw MP-L-V4V0 (2D image coordinates, no canonicalisation).
3. canon-v1 (3D; also reported with z dropped, for a 2D-vs-2D comparison with 1).

**Probes (5-fold stratified, standardised features, L2 logistic regression C=1).**
- P1 summary statistics per window: per joint-group mean / std / range of each coordinate,
  mean |velocity|, mean |acceleration| (jitter), missing rate per group.
- P2 mean pose: the window-mean of all joint coordinates (body shape / framing).
Reported: accuracy ± fold std, confusion matrix, top features.

**Decision rule.** canon-v1 passes if, on the 3-way ASL task, P1 accuracy is
≤ RESONA-77's P1 accuracy − 0.15 **and** ≤ 0.70. Otherwise canon-v2 (candidates: per-frame
shoulder centring, hand-size normalisation, fps-matched jitter) is designed from the top
features and the gate is re-run; every version is reported. Some separability is genuine
(signer populations, recording conditions), so the gate asks for a clear drop, not chance.

## 4. Where and how it runs

A CPU server (32 cores), low priority, bounded memory (same limits as the extraction).
Canonicalising is a few minutes per million frames. The fingerprint uses ~1,500 windows and runs
in seconds. RESONA-77 baseline features are computed from the old H5 files on a separate
machine and the small feature tables are copied over.

## 5. Deliverables

`tools/canon/canon.py` (pure functions + unit tests), `tools/canon/build_canon.py`
(per dataset, resumable), `tools/canon/fingerprint.py` (features, probes, report),
report `docs/fingerprint_gate.md` (raw result JSON not included in this release).
Re-pretraining starts only after the gate passes.

## 6. canon-v2 (pre-registered 2026-10-30, after canon-v1 failed the gate)

canon-v1 result: 3-way P1 0.923 vs RESONA-77 0.993 (rule needs ≤ 0.70) → FAIL
(`docs/fingerprint_gate.md`). Top features: trunk depth (z of hips/nose), arm vertical range,
face-contour spread / sideways offset, mean pose (P2 0.943). Changes, each aimed at a feature
family; everything else as canon-v1:

- **v2a — full torso frame.** Per clip, from frames with pose found: x = median shoulder
  vector (l→r), y = median torso-down vector (shoulder midpoint → hip midpoint) made orthogonal
  to x, z = x × y. Rotating into this frame removes camera yaw, pitch (elevation) and roll, not
  only yaw. Origin and unit as canon-v1.
- **v2b — v2a + per-frame shoulder centring + hips as frame only.** Each frame is translated
  so its own shoulder midpoint is the origin; the two hip joints are used to build the frame but
  are masked out of the model input (MediaPipe extrapolates hips that are off-screen, and the
  datasets differ in whether the hips are in view).

Both are run through the unchanged gate (same clips, windows, probes, rule). The first that
passes, in the order v2a, v2b, is adopted; if both pass, the one with the lower 3-way P1. All
results are reported, including failures. Face joints are not changed (head movement relative
to the body carries linguistic information).

**Secondary analysis (reported, not part of the rule).** The same probes within task type:
isolated (asl_citizen vs wlasl, chance 0.5) and continuous (how2sign vs youtube_asl vs openasl,
chance 1/3), for RESONA-77, canon-v1, v2a and v2b. This separates task-type differences
(isolated clips start and end at rest) from recording artefacts.

**Amendment (2026-10-30, before any v2 result was seen).** Sign language
needs no lower body, and MediaPipe extrapolates hips that are off-screen (most datasets are
seated / cropped at the waist), so hips carry dataset-specific error.
- **v2c — no lower body at all.** The hip joints are neither used for the frame nor given to
  the model (slots 7, 8 kept as zeros with mask 0, so the 77-joint layout and the 18 groups are
  unchanged). Frame: x = median shoulder vector (l→r); up = median shoulder-midpoint→nose
  vector made orthogonal to x; z = x × up. Per-frame shoulder centring as v2b. Legs were never
  part of C77.
- Selection replaces the rule above: among v2a, v2b, v2c, the variants that pass the unchanged
  gate are candidates; if several pass, v2c is preferred (no extrapolated joints), otherwise the
  lowest 3-way P1. All three are reported.

## 7. canon-v3 — last geometric attempt (pre-registered 2026-10-30, after v2a/b/c failed)

v2 results: 3-way P1 0.920 / 0.927 / 0.931 (all FAIL). Remaining top features are recording
properties, not geometry: keypoint jitter (`*.acc`, asl_citizen 1.5–2× the others: webcam
compression), face size relative to the shoulders (`contour.std_x`) and the head's sideways
offset (`trunk.mean_x`, `contour.mean_x`). canon-v3 = canon-v2c plus:
- **Temporal smoothing** after resampling to 25 fps: Savitzky–Golay, window 7 frames (0.28 s),
  order 2, per joint and coordinate, over runs of valid frames only (masked frames untouched;
  runs shorter than 7 left as they are). Aim: equalise jitter levels.
- **Face in its own scale:** face joints (mouth8, face18) expressed relative to the face nose
  (index 1 of the mesh) and divided by the per-clip median inter-ocular distance (eye outer
  corners 33–263), then re-attached at the body nose with a fixed size of 0.35 shoulder widths
  per inter-ocular unit. Aim: remove face-size / distance cues; the face shape and its motion
  are kept.
The gate, clips, windows and probes are unchanged. If v3 fails, geometric canonicalisation stops
here: the remaining separability is reported as a property of the corpora, and the decision on
how to proceed (mitigation during pretraining, or a revised gate) is taken separately.

## 8. canon-v3.1 and the revised gate (decided 2026-10-30, before any v3.1 result)

**Why the gate is revised.** The calibration probes (`docs/canon_quality.md`) show that three
groups of disjoint signers *within one dataset* are already separable at P1 0.744–0.783
(ASL Citizen), 0.754–0.766 (AUTSL), 0.644–0.674 (WLASL); random halves give 0.48–0.52. Any
two corpora differ in their signers, so the old threshold (≤ 0.70) is at or below the
signer-identity level and cannot be met by canonicalisation. Replacing it was approved.

**Revised rule (3-way ASL P1, same clips/windows/probe):** pass if
P1 ≤ S + 0.10, where S = the highest within-dataset 3-signer-group P1 among ASL Citizen,
AUTSL and WLASL, measured on the **same canon version** with the same protocol
(300 clips/group, rng seed 0). The earlier rule's outcome (FAIL for v1–v3) stays reported.

**canon-v3.1** = canon-v3 with one correction found in the quality check: the up axis is the
median shoulder-midpoint → **ear-midpoint** vector (pose 7, 8) instead of → nose. The nose sits
far in front of the shoulders in MediaPipe world coordinates, which tilted the v2c/v3 frame
~65° backwards; the ears are roughly in the shoulders' coronal plane. Acceptance check before
use (besides the gate): median angle between the canonical up axis and the anatomical torso
direction < 15° in every dataset, natural front/side views in the figures, all v3 unit tests
(invariances, hips never read, smoothing, face scale) passing.

## 9. canon-v4 — final canonicalisation, chosen on quality (decided 2026-10-30)

Outcome so far: six variants (v1, v2a, v2b, v2c, v3, v3.1) give 3-way P1 0.920–0.931; all fail
the original rule and the signer-calibrated rule (P1 − S = 0.13–0.15). Geometry does not move
the probe, so the canonicalisation is now chosen on quality, and the remaining fingerprint is
reported as a limitation and handled during pretraining. Both gate failures stay in the report;
the threshold is not lowered again.

**canon-v4** = canon-v3 pipeline (hips never read and masked, per-frame shoulder-midpoint
centring, clip unit, face in its own scale, Savitzky–Golay 7/2 after the 25 fps resample) with
the **v1 orientation plus roll**: no pitch change (MediaPipe's camera-vertical is 5–9° from
the torso in every dataset, while head-based up axes are 47–67° off because MediaPipe places the
nose and ears 0.13–0.16 m in front of the shoulders); yaw about y so the median shoulder vector
lies in the x–y plane, then roll about z so it lies on +x.

**Acceptance (all required before use):** median up-axis–torso angle < 15° and median residual
shoulder tilt < 5° in every quality dataset; shoulder width 1.000; no far-out frames; figures
upright and natural (checked by eye); all unit tests pass. The fingerprint gate is run once more
for the record only.

**Mitigation in pretraining (Phase 2 plan):** dataset-balanced (temperature) sampling as before;
a dataset-ID linear probe on the frozen encoder at every checkpoint, reported next to the
downstream results (encoder vs input-level fingerprint); evaluation sets excluded by the
inductive filter rebuilt on the new clip ids.

## 10. canon-v5: reduce within-language separability (pre-registered 2026-10-30)

*The canon-v5 analysis code is not part of this release.*

canon-v4 is the pretraining input of the running pilot (unchanged). Work on normalisation
continued so that datasets of the **same language** are harder to tell apart.
Geometric frames did not move the probe; the remaining top cues are static per-clip properties
(sideways head offset, hand size, arm proportions, face depth). Candidates, all built on v4:

- **v5a — bone-length retargeting.** Replace every bone length (arms, hands, face-to-head link,
  shoulder line) with a fixed template (pooled median over the train clips of all datasets),
  keeping each bone's direction per frame (forward kinematics from the shoulder midpoint along
  the C77 kinematic tree). Removes body/hand proportions and residual scale cues; keeps motion.
- **v5b — v5a + per-clip static posture removal.** Subtract the clip-median offset of the head
  (nose, face block) relative to the shoulder midpoint, and of each wrist's rest position is
  NOT touched (handshape location is linguistic). Only the head's static offset is removed.
- **v5c — v5a + per-dataset standardisation.** Per dataset, subtract the mean pose and divide
  by the per-joint-coordinate std computed on that dataset's train clips (unsupervised; a new
  dataset gets its own statistics).

**Measurements (same clips/windows/probes):** 3-way ASL P1/P2, isolated pair, continuous
triple, 5-way, and S (signer-calibrated) per version; both earlier rules reported. **Content
preservation check (required):** frozen-input linear ISLR on AUTSL and LSA64 official splits
(features: the same P1 statistics plus a temporal-pyramid mean pose, 3 seeds) — a version is
acceptable only if its ISLR accuracy is ≥ v4's minus 2 points on both. Among acceptable
versions, the one with the lowest 3-way P1 is preferred; v5c is preferred only if it beats the
best of v5a/v5b by ≥ 0.05 (it needs per-dataset statistics at test time). If none improves on
v4 by ≥ 0.03 3-way P1, v4 stays.

## 11. Same sign, same motion? Cross-dataset gloss consistency (defined 2026-10-30, before results)

*The cross-dataset analysis code is not part of this release.*

Question: does the same word look like the same motion in different datasets after
canonicalisation? This measures what matters for pretraining (content alignment across
corpora) rather than dataset separability.

**Data.** ASL Citizen and WLASL glosses present in both (mapping by normalised English gloss;
reusing an existing cross-dataset gloss mapping if available; document the mapping and the
number of shared glosses). Official train/test splits of each.

**Measures**, for RESONA-77, raw MP-L-V4V0 (2D image), canon-v1, v4, v5a, v5c:
1. **Cross-dataset ISLR transfer.** Linear classifier (features: P1 statistics of the whole clip
   + temporal-pyramid mean pose, as in §10; StandardScaler + logistic regression, 3 seeds) on the
   shared glosses: train ASL Citizen train → test WLASL test, and WLASL train → ASL Citizen test;
   plus within-dataset (train/test of the same dataset, same gloss set) as the ceiling.
   Report top-1/top-5 and the **transfer gap** = within − cross.
2. **Cross-dataset retrieval.** For each test clip of one dataset, rank the other dataset's
   train clips by DTW distance on the canonical arm+hand trajectories (body wrists, elbows,
   hand wrists and fingertips; 25 fps; masked joints handled consistently) — report top-1/top-5
   gloss retrieval and mean reciprocal rank; within-dataset retrieval as the ceiling.
A version that makes the same sign more alike across datasets lowers the transfer gap and raises
cross-dataset retrieval, without lowering the within-dataset ceiling. No version is adopted on
this basis alone; the numbers are reported together with the gate results.

**Amendment (2026-10-30, before results): within-dataset, across signers.**
Same question inside one dataset: is the same word the same motion for *different signers*?
For ASL Citizen, WLASL, AUTSL and LSA64 (signer ids from their metadata), same versions:
(a) **cross-signer retrieval**: DTW on arm+hand trajectories, query and gallery always from
different signers (subset rule as in §11), top-1/top-5 and MRR; (b) **same-gloss vs
different-gloss distance ratio**: mean DTW distance between same-gloss clips of different
signers divided by the mean distance between different-gloss clips (lower = better), with
bootstrap 95% CIs. Normalisation that removes signer differences raises (a) and lowers (b).
