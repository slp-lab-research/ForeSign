from evals.cslr_phoenix import make_batch


def test_make_batch_skips_missing_and_returns_none_when_empty():
    store = [(None, [], ["A", "B"])]
    assert make_batch(store, "cpu", [0]) is None


def test_reference_full_by_default_and_oov_dropped_when_legacy():
    from evals.cslr_phoenix import _reference
    inv = {1: "A", 2: "B"}
    rec = (None, [1, 2], ["A", "OOV", "B"])
    assert _reference(rec, inv, False) == ["A", "OOV", "B"]
    assert _reference(rec, inv, True) == ["A", "B"]


def test_max_frames_per_dataset(monkeypatch):
    from evals.cslr_phoenix import max_frames
    monkeypatch.delenv("FS_CSLR_MAXLEN", raising=False)
    monkeypatch.delenv("FS_CSLR_DATASET", raising=False)
    assert max_frames() == 384
    monkeypatch.setenv("FS_CSLR_DATASET", "csl_daily")
    assert max_frames() is None
    monkeypatch.setenv("FS_CSLR_MAXLEN", "512")
    assert max_frames() == 512
