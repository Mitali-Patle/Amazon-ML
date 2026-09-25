import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pytest
from metric import f_beta_score, per_anchor_scores

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
def test_macro_mean_and_nan():
    yt = {"a": "", "b": "S2-1,S2-2", "c": "S2-3"}
    yp = {"a": "", "b": "S2-1", "c": float("nan")}
    assert f_beta_score(yt, yp) == pytest.approx((1 + 5 / 6 + 0) / 3)
def test_micro():
    yt = {"a": "", "b": "S2-1,S2-2"}; yp = {"a": "", "b": "S2-1"}
    assert f_beta_score(yt, yp, averaging="micro") == pytest.approx(5 / 6)
def test_list_input():                assert s(["S2-1"], ["S2-1"]) == 1.0
