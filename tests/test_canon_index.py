"""INDEX.tsv / SHA256SUMS of a canon output directory."""
import hashlib
import os

from tests.test_canon_build import make_unit
from tools.canon import build_canon as B
from tools.canon import index as I


def test_index_and_sums(tmp_path):
    root = str(tmp_path / "src")
    make_unit(root)
    out_dir = tmp_path / "canon_v4"
    out_dir.mkdir()
    B.run(root, "toy", str(out_dir / "toy.h5"), workers=1, log=lambda s: None,
          version="canon-v4")
    rows = I.build(str(out_dir), workers=1)
    assert len(rows) == 1
    r = rows[0]
    assert r["dataset"] == "toy" and r["canon_version"] == "canon-v4" and r["clips"] == 6
    assert r["frames"] == 2 * (25 + 26 + 27) and r["no_pose_clips"] == 2
    assert r["splits"] == "test:2;train:4"
    data = (out_dir / "toy.h5").read_bytes()
    assert r["sha256"] == hashlib.sha256(data).hexdigest()
    lines = (out_dir / "INDEX.tsv").read_text().splitlines()
    assert lines[0].split("\t") == I.COLS and len(lines) == 2
    assert (out_dir / "SHA256SUMS").read_text() == f"{r['sha256']}  toy.h5\n"
    assert not any(p.endswith(".tmp") for p in os.listdir(out_dir))
