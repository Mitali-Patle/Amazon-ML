"""Tests for the decision layer.

The key property under test: the rule must reproduce the conservative behaviour
the metric demands, WITHOUT that behaviour being hard-coded anywhere. It should
fall out of maximising expected F_0.5.
Run: .venv/bin/python -m tests.test_decide
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.prep.decide import (choose_k, deconflict, decide_all, decide_entity,  # noqa: E402
                             expected_scores)
from src.prep.metric import entity_f_beta, macro_f_beta  # noqa: E402

FAILS = []


def check(cond, msg):
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        FAILS.append(msg)


print("== closed form matches the direct metric definition ==")
for tp, s, t in [(3, 5, 4), (2, 3, 2), (1, 1, 1), (4, 4, 6)]:
    pred = [f"x{i}" for i in range(tp)] + [f"bad{i}" for i in range(s - tp)]
    truth = [f"x{i}" for i in range(tp)] + [f"miss{i}" for i in range(t - tp)]
    direct = entity_f_beta(pred, truth)
    closed = 1.25 * tp / (0.25 * t + s)
    check(abs(direct - closed) < 1e-9, f"TP={tp} |S|={s} |T|={t}: {direct:.6f} == {closed:.6f}")

print("\n== confident candidates are all taken ==")
k, _ = choose_k([0.99, 0.98, 0.97])
check(k == 3, f"three confident -> k=3 (got {k})")

print("\n== all-weak candidates -> predict empty (singleton detection) ==")
k, _ = choose_k([0.05, 0.04, 0.02])
check(k == 0, f"weak candidates -> k=0 (got {k})")

print("\n== the conservative behaviour EMERGES, it is not hard-coded ==")
# one strong + one marginal: taking the marginal risks a false positive
k, _ = choose_k([0.95, 0.30])
print(f"     probs [0.95, 0.30] -> k={k}")
check(k in (1, 2), "marginal candidate decision is made, not crashed")
# clearly-below-even candidates must be dropped
k, _ = choose_k([0.95, 0.90, 0.10])
check(k == 2, f"drops the 0.10 tail (got k={k})")

print("\n== lone candidate: exact optimum is p > 0.5 ==")
# predicting it scores 1.0 w.p. p and 0 otherwise -> E = p
# predicting empty scores 1.0 w.p. (1-p) (the entity is a singleton) -> E = 1-p
for p, want in [(0.30, 0), (0.45, 0), (0.60, 1), (0.90, 1)]:
    k, _ = choose_k([p])
    check(k == want, f"single candidate p={p} -> k={k} (want {want})")

print("\n== expected score is a valid distribution over k ==")
e = expected_scores([0.9, 0.5, 0.1])
check(len(e) == 4, f"k=0..3 -> 4 values (got {len(e)})")
check(all(np.isfinite(e)), "all finite")
check(abs(e[0] - (0.1 * 0.5 * 0.9)) < 1e-9, f"k=0 == prod(1-p) = 0.045 (got {e[0]:.4f})")

print("\n== empty candidate list is safe ==")
check(decide_entity([], []) == [], "no candidates -> empty prediction")
k, s = choose_k([])
check(k == 0 and s == 1.0, "empty -> k=0, score 1.0 (it is a singleton)")

print("\n== max_k is respected ==")
k, _ = choose_k([0.99] * 10, max_k=3)
check(k == 3, f"capped at 3 (got {k})")

print("\n== deconfliction enforces many-to-one ==")
a = {"e1": [("S2-X", 0.9), ("S2-A", 0.8)], "e2": [("S2-X", 0.6), ("S2-B", 0.7)]}
d = deconflict(a)
check(d["e1"] == ["S2-X", "S2-A"], f"higher claimant keeps contested record: {d['e1']}")
check(d["e2"] == ["S2-B"], f"loser drops it: {d['e2']}")
allocated = [c for v in d.values() for c in v]
check(len(allocated) == len(set(allocated)), "no record allocated twice")

print("\n== deconfliction is deterministic on ties ==")
t1 = deconflict({"eB": [("S2-X", 0.5)], "eA": [("S2-X", 0.5)]})
t2 = deconflict({"eA": [("S2-X", 0.5)], "eB": [("S2-X", 0.5)]})
check(t1 == t2 or (t1["eA"] == t2["eA"] and t1["eB"] == t2["eB"]),
      "tie broken deterministically by entity id")

print("\n== deconfliction strictly removes false positives (ground truth check) ==")
truth = {"e1": ["S2-X"], "e2": ["S2-B"]}
raw = {"e1": [("S2-X", 0.9)], "e2": [("S2-X", 0.6), ("S2-B", 0.7)]}
before = macro_f_beta({e: [c for c, _ in v] for e, v in raw.items()}, truth)
after = macro_f_beta(deconflict(raw), truth)
print(f"     macro F_0.5 before={before:.4f}  after={after:.4f}")
check(after > before, "deconfliction improves the real score")

print("\n== decide_all end to end ==")
cands = {
    "e1": [("S2-1", 0.95), ("S2-2", 0.92), ("S3-9", 0.05)],   # take 2
    "e2": [("S2-3", 0.02), ("S3-4", 0.01)],                    # singleton
    "e3": [("S2-1", 0.60)],                                    # contests e1's S2-1, loses
    "e4": [],                                                  # no candidates at all
}
final, diag = decide_all(cands)
check(final["e1"] == ["S2-1", "S2-2"], f"e1 -> {final['e1']}")
check(final["e2"] == [], f"e2 predicted empty -> {final['e2']}")
check(final["e3"] == [], f"e3 lost the contested record -> {final['e3']}")
check(final["e4"] == [], "e4 with no candidates is still present in the output")
check(len(final) == 4, "every input entity appears in the output")
check(diag["removed_by_deconflict"] == 1, f"1 removal logged (got {diag['removed_by_deconflict']})")
allocated = [c for v in final.values() for c in v]
check(len(allocated) == len(set(allocated)), "global uniqueness holds")
print(f"     diag: {diag}")

def mixture(rng, n=800, calibrated=True):
    """Candidate sets with the real match-count distribution.

    calibrated=True: a candidate with probability p is true with probability p
    (what the rule assumes, and what the matcher's calibration must deliver).
    calibrated=False: negatives still carry sizeable probability mass but are
    never true -- the realistic failure mode.
    """
    truth, cands = {}, {}
    for i in range(n):
        n_true = int(rng.choice([0, 1, 2, 3, 4, 5],
                                p=[0.056, 0.054, 0.17, 0.24, 0.30, 0.18]))
        items, t = [], []
        for j in range(n_true):
            p = float(np.clip(rng.normal(0.85, 0.15), 0.01, 0.99))
            c = f"S2-{i}-{j}"
            items.append((c, p))
            if not calibrated or rng.random() < p:
                t.append(c)
        for j in range(8):
            p = float(np.clip(rng.normal(0.25, 0.18), 0.01, 0.99))
            c = f"BAD-{i}-{j}"
            items.append((c, p))
            if calibrated and rng.random() < p:
                t.append(c)
        truth[f"e{i}"], cands[f"e{i}"] = t, items
    return truth, cands


def best_fixed_threshold(cands, truth):
    best, tau_best = 0.0, None
    for tau in np.arange(0.05, 0.99, 0.05):
        s = macro_f_beta({e: [c for c, p in v if p >= tau] for e, v in cands.items()}, truth)
        if s > best:
            best, tau_best = s, tau
    return best, tau_best


print("\n== CALIBRATED probabilities: rule vs best hindsight-tuned threshold ==")
truth, cands = mixture(np.random.default_rng(0), calibrated=True)
rule_score = macro_f_beta(decide_all(cands, apply_deconflict=False)[0], truth)
fixed, tau = best_fixed_threshold(cands, truth)
print(f"     expected-F rule                      = {rule_score:.4f}")
print(f"     best fixed tau={tau:.2f} (WITH hindsight) = {fixed:.4f}")
check(rule_score >= fixed - 0.005,
      "under calibration the rule matches/beats a hindsight-tuned threshold")

print("\n== MISCALIBRATED probabilities: measured degradation ==")
# Negatives keep their probability mass but are never true, so sum(p)
# overstates E[|T|] and the rule becomes too permissive. Documented, not hidden:
# this is why calibration is a stated precondition of the decision layer.
truth_m, cands_m = mixture(np.random.default_rng(0), calibrated=False)
rule_m = macro_f_beta(decide_all(cands_m, apply_deconflict=False)[0], truth_m)
fixed_m, tau_m = best_fixed_threshold(cands_m, truth_m)
print(f"     expected-F rule                      = {rule_m:.4f}")
print(f"     best fixed tau={tau_m:.2f} (WITH hindsight) = {fixed_m:.4f}")
print(f"     degradation vs calibrated case       = {rule_score - rule_m:+.4f}")
check(rule_m < fixed_m,
      "miscalibration measurably hurts the rule -- calibrate the matcher before use")

print()
if FAILS:
    print(f"{len(FAILS)} FAILURE(S)")
    for f in FAILS:
        print("   -", f)
    sys.exit(1)
print("ALL DECISION TESTS PASSED")
