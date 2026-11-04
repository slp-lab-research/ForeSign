# Pose extraction and canonicalisation: how to run

This page is the operational guide. The design and the studies behind the choices are in
`docs/design/2026-10-25-unified-3d-pose-reextraction-design.md`, `docs/extractor_bakeoff.md`,
`docs/mp_hand_improvement.md`, `docs/design/2026-10-29-canonicalisation-fingerprint-gate-design.md`,
`docs/fingerprint_gate.md` and `docs/canon_quality.md`.

No video, pose or annotation data is distributed with this code. Obtain every dataset from its
authors under its licence (links in the README).

## Environment

A separate environment (`requirements-extract.txt`): `mediapipe==0.10.21` (the last release with
the legacy `solutions.holistic` API), `opencv-contrib-python==4.11.0.86`, `numpy==1.26.4`,
`h5py`, `psutil`. The extractor configuration is frozen in `configs/extract/mp_l_v4v0.json`
(`MP-L-V4V0`, config hash `30d5b11c99ac`); `extract_dataset` refuses to run if the live extractor
configuration, the mediapipe version, or the sha256 of `tools/extract/extractors.py` /
`tools/extract/common.py` differ from it (`tools.extract.freeze_config.check_frozen`). Do not edit
those two files if you want outputs identical to ours. (`code_git_sha` in the frozen file refers to
the private development repository.)

## Environment variables

| variable | meaning | default |
|---|---|---|
| `REEXTRACT_SRC_ROOT` | root holding the source videos, one folder per dataset (layout below) | `./videos` |
| `REEXTRACT_ROOT` | work root: `manifest/`, `out/`, `reports/`, `scratch/`, `state/` | `~/reextract` |
| `REEXTRACT_SCRATCH` | where archive members are unpacked | `<REEXTRACT_ROOT>/scratch` |
| `YTASL_DEST` | YouTube-ASL download folder (`videos/`, `completed.tsv`) | `<REEXTRACT_SRC_ROOT>/youtube_asl` |
| `REEXTRACT_FILE_LIST` | optional list of `<size>\t<path relative to REEXTRACT_SRC_ROOT>` used instead of directory listings | unset |
| `REEXTRACT_QUEUE_SKIP` | YouTube-ASL clip ids already handled (one per line) | unset |
| `YTASL_DOWNLOADER_PATTERN` | `pgrep -f` pattern of a running downloader (stream driver only) | unset (= no downloader) |

## Source layout under `REEXTRACT_SRC_ROOT`

Clip ids depend only on paths *relative to each root* (and archive member names), so they match
ours as long as the inside of each dataset folder follows the official release. Defined in
`tools/extract/datasets.py`:

| dataset | path(s) |
|---|---|
| autsl | `autsl/{train,val,test}/*_color.mp4` |
| asl_citizen | `asl_citizen/ASL_Citizen/videos/*.mp4`, splits `asl_citizen/ASL_Citizen/splits/{train,val,test}.csv` |
| wlasl | `wlasl/wlasl_video_format.zip` (members `video_format/...`), `wlasl/WLASL_v0.3.json` |
| include | `include/*.zip` |
| lse | `lse/extracted/...` (LSE_UVIGO; MSSL / OSLWL folders) |
| phoenix14t | `phoenix14t/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/{train,dev,test}/<id>/` |
| phoenix14 | `phoenix14/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/{train,dev,test}/<id>/1/` |
| ghanaian_sl | `ghanaian_sl/GSL_openpose_data/<SIGN>/<SIGN>.mp4` |
| lsa64 | `lsa64/raw/all/<sign>_<signer>_<rep>.mp4` |
| sign2mint | `sign2mint/videos/*.mp4` |
| lsa_t | `lsa_t/cuts/*.mp4` (unpacked from `lsa_t/clips.7z`) |
| slovo | `slovo/{train,test}/*.mp4` |
| karsl | `karsl/{01,02,03}/{train,test}/<range>/<sign>/*_c.mp4` |
| chicagofswild | `chicagofswild/ChicagoFSWild/ChicagoFSWild/ChicagoFSWild-Frames.tgz` (+ `.csv`), `chicagofswild/ChicagoFSWildPlus/ChicagoFSWildPlus-Frames.tgz` (+ `.csv`) |
| how2sign | `how2sign/h2s_{train_clips,val,test}.tar.gz` (ZIP archives despite the name; members `raw_videos/...`) |
| openasl | `openasl/videos/<yid>.f<format>.{mp4,webm}` |
| isign | `isign/isign_videos/iSign-videos_v1.1/*.mp4`, `isign/isign_test_videos/test_mp4/*.mp4` (optional `isign/{val,test}.csv`) |
| youtube_asl | `$YTASL_DEST/videos/<yid>.<ext>` listed in `$YTASL_DEST/completed.tsv` (`yid<TAB>file<TAB>bytes<TAB>utc`) |

The YouTube-ASL video downloader is not part of this release; obtain the videos from the official
YouTube-ASL id list with a tool of your choice, respecting YouTube's terms.

## Steps (per dataset)

```bash
export REEXTRACT_SRC_ROOT=/path/to/videos REEXTRACT_ROOT=/path/to/work
python -m tools.extract.build_manifest autsl --workers 16      # manifest/autsl.csv
python -m tools.extract.extract_dataset autsl --workers 16 --mem-budget-gb 48   # out/autsl/shard_*.h5
python -m tools.extract.verify_dataset autsl --workers 16      # reports/autsl_verify.json
# optional: free disk space after a passing verification (only files we unpacked / downloaded)
python -m tools.extract.cleanup autsl --scratch
```
`tools.extract.stream_extract` runs the tier-2 datasets and YouTube-ASL batches on one long-lived
worker pool with the same per-clip extraction (identical outputs; only the order differs).

Canonicalisation (C77-3D, **canon-v4**, the version used for all Phase-2 data) and the index of a
canon directory:
```bash
python -m tools.canon.build_canon --version canon-v4 --src-root $REEXTRACT_ROOT \
    --ds autsl --out /path/to/canon_v4/autsl.h5 --workers 16
python -m tools.canon.build_canon --version canon-v4 --src-root $REEXTRACT_ROOT \
    --ds youtube_asl --batch b00001 --out /path/to/canon_v4/youtube_asl_b00001.h5
python -m tools.canon.index /path/to/canon_v4 --workers 8       # INDEX.tsv + SHA256SUMS
```
The canon directory is what `FS_CANON_ROOT` points to for pretraining and evaluation.

Pretraining exclusion list (`data_ann/canon_exclude.json`, shipped; rebuild only if your
manifests differ): `python -m tools.build_canon_exclude --manifest-dir $REEXTRACT_ROOT/manifest`.
The PHOENIX cross-release overlap needs `manifest/phoenix14{,t}_frame_md5.tsv.gz` (per-frame md5 of
the PHOENIX frame folders, written by `build_manifest`).

Fingerprint gate and quality checks (study tools): `python -m tools.canon.fingerprint --help`,
`python -m tools.canon.quality --help`.
