import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pytest
import numpy as np
import pandas as pd
from metric import f_beta_score, per_anchor_scores, score_slices

def s(t, p, **k): return f_beta_score({"a": t}, {"a": p}, **k)

def test_singleton_correct_empty():   assert s("", "") == 1.0
def test_singleton_pred_nonempty():   assert s("", "S2-1") == 0.0
def test_nonsingleton_pred_empty():   assert s("S2-1", "") == 0.0
def test_perfect():                   assert s("S2-1,S3-2", "S3-2,S2-1") == 1.0
def test_precision_weighted():
    # T={1,2}, P={1}: TP=1,FP=0,FN=1 -> 1.25/(1.25+.25)=5/6
    assert s("S2-1,S2-2", "S2-1") == pytest.approx(5 / 6)
    # T={1}, P={1,2}: TP=1,FP=1,FN=0 -> 1.25/2.25=5/9  (extra id hurts more)
    assert s("S2-1", "S2-1,S2-2") == pytest.approx(5 / 9)
def test_disjoint():                  assert s("S2-1", "S2-9") == 0.0
def test_missing_anchor_is_empty():
    assert f_beta_score({"a": "", "b": "S2-1"}, {}) == 0.5
    assert f_beta_score({"a": "", "b": "S2-1"}, {"a": ""}, allow_missing=True) == 0.5
def test_macro_mean_and_nan():
    yt = {"a": "", "b": "S2-1,S2-2", "c": "S2-3"}
    yp = {"a": "", "b": "S2-1", "c": float("nan")}
    assert f_beta_score(yt, yp) == pytest.approx((1 + 5 / 6 + 0) / 3)
def test_micro():
    yt = {"a": "", "b": "S2-1,S2-2"}; yp = {"a": "", "b": "S2-1"}
    assert f_beta_score(yt, yp, averaging="micro") == pytest.approx(5 / 6)
def test_list_input():                assert s(["S2-1"], ["S2-1"]) == 1.0


# ---- 7 hand-computed cases from docs/PROBLEM_BRIEF.md sec. 4 (verified independently) ----
@pytest.mark.parametrize("t,p,exp", [
    ("a,b", "a,b,c", 2.5 / 3.5),        # PDF example: P=2/3,R=1 -> 0.714
    ("a,b", "a", 5 / 6),                # 0.833
    ("a,b", "a,b,c,d", 2.5 / 4.5),      # 0.556
    ("", "", 1.0),
    ("", "a", 0.0),
    ("a", "", 0.0),
    ("c", "a,b", 0.0),
])
def test_brief_table(t, p, exp):
    assert s(t, p) == pytest.approx(exp)
    assert s(t, p, strict=True) == pytest.approx(exp)
    assert s(t, p, strict=False) == pytest.approx(exp)

def test_pdf_example_ids():
    assert s("S2-00047,S3-00812", "S2-00047,S2-00193,S3-00812") == pytest.approx(0.714, abs=5e-4)

def test_duplicate_ids():
    # T={1}, P=[1,1]: default strict counts dup as FP -> TP=1,FP=1 -> 5/9; lenient collapses -> 1.0
    assert s("S2-1", "S2-1,S2-1") == pytest.approx(5 / 9)
    assert s("S2-1", "S2-1,S2-1", strict=False) == 1.0
    assert s("", ["S2-1", "S2-1"]) == 0.0

def test_whitespace_and_trailing_comma():
    assert s("S2-1,S2-2", " S2-2 , S2-1 ,") == 1.0
    assert s("", " , ,") == 1.0
    assert s("S2-1,", "S2-1") == 1.0

def test_pd_na_and_nan_in_list():
    assert s("", pd.NA) == 1.0
    assert s("S2-1", pd.NA) == 0.0
    assert f_beta_score({"a": pd.NA}, {"a": None}) == 1.0
    assert s("S2-1", ["S2-1", float("nan"), pd.NA, None]) == 1.0
    assert s("S2-1", np.array(["S2-1"])) == 1.0

def test_micro_singleton_false_positive():
    yt = {"a": "", "b": "S2-1"}; yp = {"a": "S2-9", "b": "S2-1"}
    assert f_beta_score(yt, yp) == pytest.approx(0.5)  # macro: (0+1)/2
    # micro pooled: TP=1, FP=1, FN=0 -> 1.25/2.25
    assert f_beta_score(yt, yp, averaging="micro") == pytest.approx(5 / 9)

def test_series_input_and_empty_ytrue():
    yt = pd.Series({"a": "", "b": "S2-1"}); yp = pd.Series({"b": "S2-1"})
    assert f_beta_score(yt, yp, allow_missing=True) == 1.0
    with pytest.raises(ValueError):
        f_beta_score({}, {})


def test_key_mismatch_raises_and_partial_warns():
    with pytest.raises(ValueError):
        f_beta_score({"S1-1": "S2-1"}, {1: "S2-1"})           # int vs str keys: zero overlap
    with pytest.warns(UserWarning):
        f_beta_score({"a": "S2-1", "b": "S2-2"}, {"a": "S2-1"})
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert f_beta_score({"a": "S2-1", "b": ""}, {"a": "S2-1"}, allow_missing=True) == 1.0
        assert f_beta_score({"a": "S2-1"}, {}) == 0.0          # fully empty y_pred is allowed

def test_score_slices_handles_empty():
    yt = {"a": "S2-1", "b": "", "c": "S2-3"}; yp = {"a": "S2-1", "b": "", "c": "S2-9"}
    g = {"a": "US", "b": "US", "c": "IN", "z": "FR"}          # FR anchor not in y_true -> no slice
    r = score_slices(yt, yp, g)
    assert r == {"US": 1.0, "IN": 0.0, "FR": None}


def test_score_slices_key_guard():
    with pytest.raises(ValueError):
        score_slices({"S1-1": "S2-1"}, {1: "S2-1"}, {"S1-1": "US"})   # int vs str keys
    yt = {"a": "S2-1", "b": "S2-2"}; g = {"a": "US", "b": "IN"}
    with pytest.warns(UserWarning) as w:                              # partial overlap warns, at caller
        r = score_slices(yt, {"a": "S2-1"}, g)
    assert r == {"US": 1.0, "IN": 0.0} and w[0].filename == __file__
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert score_slices(yt, {"a": "S2-1"}, g, allow_missing=True) == {"US": 1.0, "IN": 0.0}
        assert score_slices({"a": "S2-1"}, {}, {"a": "US"}) == {"US": 0.0}   # all-empty pred allowed
