# Unified 3D Pose Re-extraction — Design

*Internal design/record document, lightly edited for the code release; dates are those of the original record.*

- Date: 2026-10-25
- Status: draft for approval
- Context: pilot (Phase 1a) showed a random-init encoder matches pretrained ones on frozen CSLR and WLASL; the existing RESONA-77 data is too heterogeneous (mixed extractors across datasets) and datasets are trivially distinguishable (dataset fingerprint). Post-hoc standardisation (RESONA-77) was insufficient. Decision: re-extract all poses from raw video with **one extractor, one setting, 3D keypoints**.

## 1. Goals

1. One pose extractor, one version, one configuration for every dataset (pretraining pool + evaluation sets).
2. **3D keypoints** (body, both hands, face) so the representation carries spatial depth and the JEPA can predict 3D structure.
3. Raw extractor output stored losslessly (all landmarks, x/y/z/score); every later canonicalisation is a separate, reversible step.
4. A measurable reduction of the dataset fingerprint, verified before re-pretraining.
5. Videos deleted after verified extraction (storage limits); a manifest keeps the dataset re-buildable.

Non-goals: SMPL-X mesh recovery, per-dataset tuned settings, keeping raw videos.

## 2. Extractor selection — sample bake-off (decides the extractor)

**Principle:** count, in the same unit, which method the datasets already use most, and align to it unless the bake-off shows it is clearly worse. The current majority is the MediaPipe family, but versions/settings differ, so a single exact configuration is chosen here.

**Candidates**
| ID | Extractor | 3D | Compute |
|---|---|---|---|
| MP-L | MediaPipe Holistic (legacy `solutions.holistic`, model_complexity 2, refine_face_landmarks, video mode) | pose world landmarks (metric, hip-centred) + hand/face relative z | CPU |
| MP-T | MediaPipe Tasks HolisticLandmarker (current API, video mode) | world landmarks for pose and hands | CPU |
| RTMW3D | RTMW3D-x whole-body 133-kpt 3D (mmpose `rtmpose3d` / rtmlib if available) | 3D for all 133 kpts | GPU (one GPU) |

(2D RTMW + separate lifting is dropped unless both 3D candidates fail: two-stage error and no 3D hands.)

**Sample:** 40 clips per dataset from 8 datasets spanning resolutions/cameras: PHOENIX-2014T (210×260 frames), AUTSL (512²), ASL Citizen (640×480 webcam), WLASL (YouTube mix), How2Sign (studio), INCLUDE, LSE (1280×720), iSign (web). Same clips for all candidates.

**Metrics (all label-free, computed per dataset and pooled)**
1. Hand detection rate: fraction of frames with each hand detected while the wrist is inside the frame.
2. Temporal stability: mean joint acceleration (jitter) on hands and face; left/right hand swap rate.
3. 3D plausibility: coefficient of variation of bone lengths (forearm, finger segments) over time; fraction of frames with implausible depth jumps.
4. Throughput: frames/s per CPU core (MP) or per GPU (RTMW3D).
5. **Fingerprint probe:** logistic-regression dataset-ID accuracy on per-clip summary statistics (coordinate range, jitter, missing rate, z spread, score distribution) — restricted to the **ASL datasets** (ASL Citizen, WLASL, How2Sign) so language identity cannot explain separability. Lower is better.
6. Spot visual check: 5 overlay videos per dataset × candidate.

**Pre-declared decision rule:** pick the candidate with the best hand detection rate unless another is within 2 points on it and better on fingerprint (5) and stability (2); ties → MediaPipe (majority principle, literature comparability with YouTube-ASL/SHuBERT-style baselines). Throughput only breaks remaining ties unless a candidate would take > 3 weeks for the tier-1 datasets.

## 3. Compute and storage

| Resource | Use | Limits |
|---|---|---|
| A shared CPU server (48 cores used) | MediaPipe extraction | low priority, bounded core and memory use; no GPU |
| One GPU | RTMW3D bake-off and (if selected) extraction; later re-pretraining | shared with other jobs |
| A workstation | extraction for datasets held locally | when idle |
| Optional cloud GPU | extra capacity for GPU extractors (RTMW3D) or overflow; only for datasets that can be fetched directly from the internet | time-limited jobs → shards small and resumable |

## 4. Streaming pipeline (download → extract → verify → delete)

Per dataset, resumable, one manifest row per video:
1. **Acquire** (if not local): download to a staging dir on the machine that will extract it.
2. **Extract** with the selected extractor; write per-dataset HDF5 `<ds>_<extractor>.h5`, one dataset per clip `(T, K, 4)` float16 = (x, y, z, score), attributes fps, width, height, extractor name+version, config hash.
3. **Verify** automatically: every manifest video has an output; output frame count == decoded frame count; no NaN; per-dataset detection-rate report within expected range; failures listed.
4. **Delete** the dataset's videos only after verification passes (and in an approved order of deletion); keep `manifest.csv` (video id, source URL/path, sha256, frames, fps, resolution, split).
5. Overlap: while dataset N is extracting, dataset N+1 downloads; at most ~2 datasets of video on disk.

**Order:** (1) datasets already on disk and small/medium (AUTSL, ASL Citizen, WLASL, INCLUDE, LSE, PHOENIX-2014/2014T, CISLR), (2) large ones (How2Sign, Sem-Lex, iSign, OpenASL-partial), (3) direct downloads not held (LSA64, Slovo, Sign2MINT, ChicagoFSWild, DGS-Korpus, KArSL full, LSA-T, Ghanaian lexicon), (4) request-form datasets (CSL-Daily, Sem-Lex missing parts, WLASL missing videos), (5) longest last: YouTube-ASL / OpenASL scraping (remove OpenASL-test overlap: 72.4% reported in YouTube-ASL), AI-Hub KSL (domestic access).

## 5. Canonicalisation (separate step, after extraction)

Designed after the bake-off on its output; constraints fixed now:
- Body in metric world coordinates (hip/shoulder-centred, shoulder-width scale); hands relative to wrists and face relative to head, re-attached to body frame.
- Resample to a common frame rate (fps recorded per clip).
- Drop extractor score as a model input (keep for masking/validity only) — score scales are a known fingerprint.
- Keep 2D projection available for literature comparison.

## 6. Acceptance gate before re-pretraining

On canonicalised data: ASL-only dataset-ID probe accuracy (metric 5) must drop clearly versus the old RESONA-77 data (report both; target set after the bake-off from the candidate's measured value). The random-init vs pretrained gap is then re-measured with the Phase 1a pipeline.

## 7. Deliverables

`tools/extract/` (extractor wrappers, manifest, verify, delete), bake-off report `docs/extractor_bakeoff.md`, per-dataset HDF5 + manifests, canonicalisation module, fingerprint report. Licences: derived poses kept for research use; CSL-Daily / restricted datasets never committed to git.
