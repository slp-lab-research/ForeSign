"""The release tree imports on its own (no external code paths) and carries no machine paths."""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES = ["foresign.train", "foresign.jepa", "foresign.fastops", "foresign.canon_data",
           "evals.islr_canon", "evals.finetune_islr_canon", "evals.islr_std",
           "evals.cslr_phoenix", "evals.cslr_cosign", "evals.dsid_probe", "evals.phoenix_wer",
           "tools.run_queue", "tools.run_finetune_p2", "tools.run_sota_p2",
           "tools.make_random_ckpt", "tools.equiv_compile", "tools.profile_step",
           "tools.build_canon_exclude", "tools.build_autsl_labels", "tools.build_phoenix_ann",
           "tools.make_configs", "tools.canon.canon", "tools.canon.build_canon",
           "tools.canon.index", "tools.canon.fingerprint", "tools.extract.datasets",
           "tools.extract.build_manifest", "tools.extract.verify_dataset", "tools.extract.cleanup"]
BAD = re.compile(r"(/mnt/|/scratch/|[A-Z]:[\\/](Users|idc_jun))", re.I)


def test_modules_import_standalone():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    code = "import " + ", ".join(MODULES)
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr[-2000:]


def test_no_machine_paths_in_code():
    bad = []
    for dp, dns, fns in os.walk(ROOT):
        dns[:] = [d for d in dns if not d.startswith(".") and d != "__pycache__"]
        for fn in fns:
            if fn.endswith((".py", ".yaml", ".json")) and fn != os.path.basename(__file__):
                p = os.path.join(dp, fn)
                for n, line in enumerate(open(p, encoding="utf-8", errors="replace"), 1):
                    if BAD.search(line):
                        bad.append(f"{os.path.relpath(p, ROOT)}:{n}: {line.strip()[:120]}")
    assert not bad, "\n".join(bad)
