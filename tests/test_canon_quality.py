"""quality metrics, signer groups and the work / figure path on synthetic clips."""
import numpy as np
import pytest

from tests.test_canon import synth
from tools.canon import canon as C
from tools.canon import quality as Q


def test_metrics_canonical_clip():
    kp, world, flags, W, H = synth()
    for v in ("canon-v1", "canon-v2c", "canon-v3"):
        x, m, _ = C.canon_clip(kp, world, flags, W, H, 25.0, version=v)
        q = Q.clip_metrics(x, m)
        assert q["shoulder_width"] == pytest.approx(1.0, abs=0.1)
        assert q["shoulder_axis_deg"] < 10
        assert q["up_axis_deg"] < 40
        assert q["far_frame_pct"] == 0
        assert 0 < q["forearm_upperarm"] < 3 and q["hand_size"] > 0 and q["iod"] > 0


def test_metrics_v3_iod_fixed():
    kp, world, flags, W, H = synth()
    x, m, _ = C.canon_clip(kp, world, flags, W, H, 25.0, version="canon-v3")
    assert Q.clip_metrics(x, m)["iod"] == pytest.approx(C.FACE_SIZE, rel=0.1)


def test_far_frames_and_jitter():
    x = np.zeros((10, 77, 3), np.float32)
    x[:, 1, 0], x[:, 2, 0] = 0.5, -0.5
    m = np.ones((10, 77), np.uint8)
    x[3, 20] = 5.0
    q = Q.clip_metrics(x, m)
    assert q["far_frame_pct"] == pytest.approx(10.0)
    assert Q.jitter(x, m) > 0


def test_summarise():
    s = Q.summarise([1, 1, 1, 1, 1, 1, 1, 10, None, float("nan")])
    assert s["n"] == 8 and s["median"] == 1 and s["outlier_pct"] == pytest.approx(12.5)


def test_signer_groups_disjoint_and_balanced():
    sig = {f"c{i}": i % 10 for i in range(100)}
    g = Q.signer_groups(sig, 3, seed=0)
    by = {}
    for c, s in sig.items():
        by.setdefault(s, set()).add(g[c])
    assert all(len(v) == 1 for v in by.values())
    counts = np.bincount(list(g.values()))
    assert len(counts) == 3 and counts.min() >= 30


class _G:
    def __init__(self, kp, world, flags, W, H):
        self.attrs = dict(width=W, height=H, fps=30.0)
        self._d = dict(kp=kp, world=world, flags=flags)

    def __getitem__(self, k):
        return self._d[k]


def test_versions_and_figures(tmp_path):
    kp, world, flags, W, H = synth(T=90)
    vers = Q.versions_of(_G(kp, world, flags, W, H))
    assert set(vers) == {"raw", "canon-v1", "canon-v2c", "canon-v3", "canon-v3.1", "canon-v4", "_rawn",
                         "_axes"}
    axes = vers.pop("_axes")
    assert set(axes["canon-v3.1"]) == {"torso_up_deg", "up_spread_deg"}
    assert vers["canon-v3"][0].shape == (75, 77, 3) and vers["raw"][0].shape == (75, 77, 3)
    rawn = vers.pop("_rawn")
    assert rawn[0].shape == (75, 77, 2)
    r = dict(ds="asl_citizen", clip="asl_citizen__x", tags=["quality"],
             metrics={v: Q.clip_metrics(x, m) for v, (x, m) in vers.items()},
             viz={v: (x[:50], m[:50]) for v, (x, m) in vers.items()})
    paths = Q.figures([r], str(tmp_path))
    assert any(p.endswith("skeleton_asl_citizen.png") for p in paths)
    assert any(p.endswith(".gif") for p in paths)


def test_frame_axis_metrics_upright():
    from tests.test_canon_v3 import _upright
    kp, world, flags, W, H = _upright()
    out = {}
    for v in ("canon-v3", "canon-v3.1", "canon-v1", "canon-v4"):
        _, _, info = C.canon_clip(kp, world, flags, W, H, 25.0, version=v)
        out[v] = Q.frame_axis_metrics(info, world, flags, v)
    assert out["canon-v3.1"]["torso_up_deg"] < 5 and out["canon-v3.1"]["up_spread_deg"] < 5
    assert out["canon-v3"]["torso_up_deg"] > 30
    assert "up_spread_deg" not in out["canon-v1"]
    assert out["canon-v4"]["torso_up_deg"] < 5
