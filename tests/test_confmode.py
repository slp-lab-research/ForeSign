import torch

from foresign.confmode import resolve_drop_conf, zero_conf


def test_resolve(monkeypatch):
    monkeypatch.delenv("FS_DROP_CONF", raising=False)
    assert resolve_drop_conf({"drop_conf": True}) is True
    assert resolve_drop_conf({}) is False
    monkeypatch.setenv("FS_DROP_CONF", "1")
    assert resolve_drop_conf({}) is True
    monkeypatch.setenv("FS_DROP_CONF", "0")
    assert resolve_drop_conf({"drop_conf": True}) is False


def test_zero_conf_does_not_mutate_input():
    x = torch.ones(2, 4, 234)
    y = zero_conf(x)
    assert torch.all(y.view(2, 4, 78, 3)[..., 2] == 0)
    assert torch.all(x == 1)
