import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pytest
import numpy as np
import pandas as pd
from folds import make_split, check_split


def synth(n=4000, seed=0):
    r = np.random.default_rng(seed)
    ids = [f"S1-{i:06d}" for i in range(n)]
    has = r.random(n) > 0.06
    country = r.choice(["US", "IN"], n)
    m = [f"S2-{i:06d},S3-{i:06d}" if h else "" for i, h in enumerate(has)]
    return pd.DataFrame({"source1_entity_id": ids, "country": country,
                         "has_match": has, "matched_entity_ids": m})


def test_deterministic_disjoint_coverage():
    df = synth()
    a = make_split(df); b = make_split(df.sample(frac=1, random_state=3))  # order-independent
    pd.testing.assert_frame_equal(a, b)
    assert set(a.source1_entity_id) == set(df.source1_entity_id)      # full coverage
    assert a.source1_entity_id.is_unique                              # each anchor once
    tr = set(a[~a.is_val].source1_entity_id); va = set(a[a.is_val].source1_entity_id)
    assert not tr & va and len(tr) + len(va) == len(df)               # disjoint + complete
    assert 0.04 < a.is_val.mean() < 0.06
    check_split(a, df, tol=0.01)  # small synthetic data: rounding noise


def test_check_split_detects_shared_match_id():
    df = synth()
    sp = make_split(df)
    a_tr = sp[~sp.is_val].source1_entity_id.iloc[0]
    a_va = sp[sp.is_val].source1_entity_id.iloc[0]
    bad = df.copy()
    shared = "S2-SHARED"
    for a in (a_tr, a_va):
        bad.loc[bad.source1_entity_id == a, "matched_entity_ids"] = f"{shared},S3-{a}"
    with pytest.raises(AssertionError, match="both sides"):
        check_split(sp, bad, tol=0.01)
