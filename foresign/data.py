"""Batch collation shared by the pretraining data streams.

The Phase 0/1a loader for the earlier 2D "resona77" pose files is not part of this release;
pretraining reads C77-3D canon data through foresign.canon_data.CanonClips.
"""
from __future__ import annotations

import torch


def collate(batch):
    poses = torch.stack([b[0] for b in batch], 0)
    nvs = torch.tensor([b[1] for b in batch], dtype=torch.long)
    langs = torch.tensor([b[2] for b in batch], dtype=torch.long)
    dsids = torch.tensor([b[3] for b in batch], dtype=torch.long)
    return poses, nvs, langs, dsids
