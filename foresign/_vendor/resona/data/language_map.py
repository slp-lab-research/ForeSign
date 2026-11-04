"""Single source of truth for dataset → sign-language mapping (RESONA v4).

Used by:
  - UnifiedSignDataset: attach a per-clip language id for language-aware
    contrastive learning (same-language cross-dataset = positive).
  - v4_probe: language clustering / cohesion metrics.

IMPORTANT: keys here are the H5 file stems (skeleton_resona77_<KEY>.h5), i.e.
the dataset names as they appear in data_p. Earlier the probe used a mapping
with names that did not match the real files (e.g. it lacked yt_asl and used
'phoenix'/'wlasl' where data is actually 'phoenix'/'wlasl'); always derive
language from THIS table keyed by the real H5 stem.
"""
from __future__ import annotations

# dataset (H5 stem) -> sign language code
DATASET_LANGUAGE: dict[str, str] = {
    # --- ASL (7 datasets) ---
    "asl_citizen": "ASL",
    "chicago_fswild": "ASL",
    "how2sign": "ASL",
    "openasl": "ASL",
    "semlex": "ASL",
    "wlasl": "ASL",
    "yt_asl": "ASL",
    # --- DGS / German (3) ---
    "dgs": "DGS",
    "phoenix": "DGS",
    "sign2mint": "DGS",
    # --- KSL / Korean (2) ---
    "ksl_sign103": "KSL",
    "ksl_sign636": "KSL",
    # --- LSA / Argentinian (2) ---
    "lsa64": "LSA",
    "lsa_t": "LSA",
    # --- single-dataset languages ---
    "autsl": "TR",          # Turkish
    "slovo": "RSL",         # Russian
    "include": "ISL",       # Indian
    "karsl": "KArSL",       # Arabic (KSU)
    "greek_sl": "GSL",      # Greek
    "ghanaian_sl": "GhSL",  # Ghanaian
    "lse": "LSE",           # Spanish
}

# Stable language → integer id (sorted for determinism across runs).
ALL_LANGUAGES: list[str] = sorted(set(DATASET_LANGUAGE.values()))
LANGUAGE_TO_ID: dict[str, int] = {l: i for i, l in enumerate(ALL_LANGUAGES)}
UNKNOWN_LANG_ID = -1


def dataset_stem_from_path(h5_path: str) -> str:
    """'/.../skeleton_resona77_yt_asl.h5' -> 'yt_asl'."""
    import os
    stem = os.path.basename(h5_path)
    if stem.endswith(".h5"):
        stem = stem[:-3]
    prefix = "skeleton_resona77_"
    if stem.startswith(prefix):
        stem = stem[len(prefix):]
    return stem


def language_for_dataset(name: str) -> str | None:
    return DATASET_LANGUAGE.get(name)


def language_id_for_path(h5_path: str) -> int:
    """Map an H5 file path to its language integer id (or UNKNOWN_LANG_ID)."""
    lang = DATASET_LANGUAGE.get(dataset_stem_from_path(h5_path))
    return LANGUAGE_TO_ID.get(lang, UNKNOWN_LANG_ID) if lang else UNKNOWN_LANG_ID
