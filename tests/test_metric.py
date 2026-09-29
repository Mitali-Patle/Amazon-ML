"""Tests for the official F_0.5 metric.

Anchored on the worked example in the problem statement, plus the edge cases
that decide real score: singletons, empty predictions, and the precision/recall
asymmetry that should drive the decision threshold.
Run: .venv/bin/python -m tests.test_metric
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.prep.metric import entity_f_beta, macro_f_beta, score_breakdown  # noqa: E402

FAILS = []


def check(cond, msg):
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        FAILS.append(msg)


def close(a, b, tol=5e-4):
    return abs(a - b) < tol


print("== official worked example from the problem statement ==")
# predicted [S2-00047, S2-00193, S3-00812], truth [S2-00047, S3-00812]
# P = 2/3, R = 1.0 -> F_0.5 = 0.714
got = entity_f_beta(["S2-00047", "S2-00193", "S3-00812"], ["S2-00047", "S3-00812"])
check(close(got, 0.714), f"documented example == 0.714 (got {got:.4f})")

print("\n== singletons: 5.58% of the score, 0% of the pairs ==")
check(entity_f_beta([], []) == 1.0, "empty prediction on singleton -> 1.0")
check(entity_f_beta(["S2-1"], []) == 0.0, "any prediction on singleton -> 0.0")
check(entity_f_beta(["S2-1", "S2-2"], []) == 0.0, "two predictions on singleton -> still 0.0")

print("\n== degenerate cases ==")
check(entity_f_beta([], ["S2-1"]) == 0.0, "predicting nothing when truth exists -> 0.0")
check(entity_f_beta(["S2-9"], ["S2-1"]) == 0.0, "no overlap -> 0.0")
check(entity_f_beta(["S2-1"], ["S2-1"]) == 1.0, "exact single match -> 1.0")
check(entity_f_beta(["S2-1", "S2-2"], ["S2-1", "S2-2"]) == 1.0, "exact multi match -> 1.0")

print("\n== duplicates in a list must not be double-counted ==")
check(entity_f_beta(["S2-1", "S2-1"], ["S2-1"]) == 1.0, "duplicate prediction treated as a set")

print("\n== precision is weighted 2x recall ==")
# one false positive vs one false negative, from the same 2-truth entity
fp_case = entity_f_beta(["S2-1", "S2-2", "S2-3"], ["S2-1", "S2-2"])   # P=2/3 R=1
fn_case = entity_f_beta(["S2-1"], ["S2-1", "S2-2"])                    # P=1   R=1/2
check(fn_case > fp_case,
      f"missing one ({fn_case:.4f}) beats adding one wrong ({fp_case:.4f}) -- precision-heavy")

print("\n== the asymmetry that should set the threshold (4-truth entity) ==")
t4 = ["S2-1", "S2-2", "S2-3", "S2-4"]
a = entity_f_beta(["S2-1", "S2-2", "S2-3"], t4)                          # 3 right, 0 wrong
b = entity_f_beta(t4 + ["S2-9"], t4)                                      # 4 right, 1 wrong
c = entity_f_beta(["S2-1", "S2-2", "S2-3", "S2-8", "S2-9"], t4)          # 3 right, 2 wrong
print(f"     3 correct / 0 wrong = {a:.4f}")
print(f"     4 correct / 1 wrong = {b:.4f}")
print(f"     3 correct / 2 wrong = {c:.4f}")
check(close(a, 0.9375), f"3/0 == 0.9375 (got {a:.4f})")
check(a > b, "CONSERVATIVE WINS: 3-correct/0-wrong beats 4-correct/1-wrong")
check(c < a, "two false positives cost more than one missed match")

print("\n== where the miss-vs-add tradeoff flips (drives the threshold policy) ==")
# For an entity with `nt` true matches: is it better to miss one, or to find
# them all but add one false positive?
flip = {}
for nt in (1, 2, 3, 4, 6, 11):
    t = [f"S2-{i}" for i in range(nt)]
    f_miss = entity_f_beta(t[:-1], t)            # nt-1 correct, 0 wrong
    f_add = entity_f_beta(t + ["S2-BAD"], t)     # nt correct, 1 wrong
    flip[nt] = (f_miss, f_add)
    print(f"     truth={nt:2d}: miss={f_miss:.4f}  add={f_add:.4f}  -> "
          f"{'MISS' if f_miss > f_add else 'ADD'} better")
check(flip[1][1] > flip[1][0], "single-match entity: adding beats missing (missing scores 0)")
check(all(flip[n][0] > flip[n][1] for n in (2, 3, 4, 6, 11)),
      "every entity with >=2 truths: MISSING beats ADDING -- be conservative")

print("\n== macro average: denominator is len(truth), omissions score 0 ==")
truth = {"e1": ["S2-1"], "e2": ["S2-2"], "e3": []}
preds = {"e1": ["S2-1"]}                      # e2 omitted, e3 omitted (correct for a singleton)
m = macro_f_beta(preds, truth)
check(close(m, (1.0 + 0.0 + 1.0) / 3), f"omitted entity scores 0, singleton omission scores 1 (got {m:.4f})")

print("\n== strict mode catches a mis-joined submission ==")
try:
    macro_f_beta({"ghost": ["S2-1"]}, truth, strict=True)
    check(False, "extra predicted entity should raise")
except ValueError:
    check(True, "extra predicted entity raises in strict mode")
check(close(macro_f_beta({"e1": ["S2-1"], "ghost": ["S2-9"]}, truth, strict=False),
            (1.0 + 0.0 + 1.0) / 3), "strict=False ignores extras")

print("\n== perfect and worst-case submissions ==")
full = {"e1": ["S2-1"], "e2": ["S2-2"], "e3": []}
check(macro_f_beta(full, truth) == 1.0, "perfect submission -> 1.0")
check(macro_f_beta({k: ["S2-999"] for k in truth}, truth) == 0.0, "all-wrong submission -> 0.0")
# truth has 3 entities, exactly 1 of which is a singleton -> empty submission scores 1/3
check(close(macro_f_beta({}, truth), 1 / 3), "empty submission -> only the singleton scores")

print("\n== breakdown diagnostics ==")
b = score_breakdown(preds, truth, groups={"e1": "US", "e2": "India", "e3": "US"})
check(b["entities"] == 3, f"entities={b['entities']}")
check(b["singletons"] == 1 and b["singletons_correct"] == 1, "singleton accounting")
check(b["pair_tp"] == 1 and b["pair_fn"] == 1 and b["pair_fp"] == 0, "pair counts")
check(b["by_group"]["US"]["macro_f05"] == 1.0, "per-group macro computed")
check(b["zero_entities"] == 1, "zero-scoring entities counted")

print("\n== macro != micro (the trap) ==")
# one entity with many matches, many entities with one -- micro is dominated by the big one
t = {"big": [f"S2-{i}" for i in range(10)], **{f"s{i}": [f"S3-{i}"] for i in range(10)}}
p = {"big": [f"S2-{i}" for i in range(10)], **{f"s{i}": [] for i in range(10)}}
bd = score_breakdown(p, t)
check(bd["macro_f05"] < bd["micro_f05"],
      f"macro {bd['macro_f05']:.3f} < micro {bd['micro_f05']:.3f} -- optimising micro misleads")

print()
if FAILS:
    print(f"{len(FAILS)} FAILURE(S)")
    for f in FAILS:
        print("   -", f)
    sys.exit(1)
print("ALL METRIC TESTS PASSED")
