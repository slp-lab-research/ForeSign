# code/tests/test_build.py
import torch

from foresign.build import build_model
from foresign.groups import GROUPS, GROUP_NAMES, shuffled_groups

CFG = dict(d_model=64, depth=2, n_heads=4, pred_dim=32, pred_depth=1, pred_heads=4,
           t_patch=2, ema_base=0.999, clip_len=64)


def test_shuffled_groups_preserve_sizes_and_joints():
    g = shuffled_groups(0)
    assert [len(x) for x in g] == [len(GROUPS[n]) for n in GROUP_NAMES]
    assert sorted(j for x in g for j in x) == sorted(j for n in GROUP_NAMES for j in GROUPS[n])
    assert g != [GROUPS[n] for n in GROUP_NAMES]
    assert g == shuffled_groups(0)


def test_build_model_layouts():
    a = build_model(CFG)
    s = build_model({**CFG, "group_layout": "shuffled"})
    assert a.tokenizer.group_idx == [GROUPS[n] for n in GROUP_NAMES]
    assert s.tokenizer.group_idx == shuffled_groups(0)
    assert s.target_tok.group_idx == s.tokenizer.group_idx
    x = torch.randn(2, 64, 234)
    assert s.tokenizer(x).shape == (2, 32, 18, 64)
