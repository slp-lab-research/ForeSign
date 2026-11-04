# Vendored code

`foresign/_vendor/resona/` holds the minimal modules of the lab's earlier code base (RESONA) that
ForeSign imports. They are copied verbatim (original headers kept); only their location changed
(imported as `foresign._vendor.resona...`).

| file | used for | origin / licence |
|---|---|---|
| `resona/models/v4_blocks.py` | `DSTformerBody`, `Attention`, `RMSNorm` (the encoder / predictor body of `foresign.jepa`; `foresign.fastops` subclasses `Attention` / `RMSNorm`) | lab code; the dual-stream spatio-temporal transformer is **ported with modifications from MotionBERT** (Zhu et al., ICCV 2023, `lib/model/DSTformer.py`, https://github.com/Walter0807/MotionBERT), which is released under the **Apache License 2.0**. The licence text is in `resona/LICENSE.MotionBERT`; the file header lists the changes (RMSNorm, SwiGLU FFN, token layout). |
| `resona/data/language_map.py` | dataset -> sign-language id (optional language-aware terms; `foresign.canon_data.lang_id`) | lab code |
| `resona/__init__.py`, `resona/models/__init__.py`, `resona/data/__init__.py` | package markers | lab code |

The lab code is distributed under this repository's licence (see `LICENSE`); the MotionBERT-derived
parts additionally remain subject to the Apache License 2.0.
