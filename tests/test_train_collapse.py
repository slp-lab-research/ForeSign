from foresign.train import COLLAPSE_EXIT_CODE, collapse_message


def test_collapse_contract():
    assert COLLAPSE_EXIT_CODE == 3
    m = collapse_message(120, 0.02, 300)
    assert m.startswith("[foresign] COLLAPSE at step 120")
