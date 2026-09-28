"""Their probabilities, OUR decision rule. No use of our model at all.

Every previous attempt combined our PREDICTIONS with theirs and lost score,
because our model is weaker and our disagreements were mostly our errors
(0.969, 0.974, 0.973 against their 0.974). This tries something structurally
different: keep their model entirely, and replace only the DECISION step.

Their rule is two hand-set constants -- a pair must clear 0.775 and the entity's
best pair must clear 0.875 or it is answered empty. Ours maximises expected
F_0.5 per entity from the closed form F = 1.25*TP/(0.25|T| + |S|), choosing the
prefix length that maximises

    E[F(top-k)] ~= 1.25 * sum_{i<k} p_i / (0.25 * sum_all p_i + k)
    E[F(empty)]  = prod_i (1 - p_i)

The evidence that their thresholds cost them: their submission has 5,689,603 ids
against a training-implied ~6.0M, and a 5.94% empty rate against a true 5.58%.
Both say under-prediction, which is what a too-high anchor threshold produces.

CAVEAT, stated plainly: this cannot be validated. Their probabilities cover test
only. The single available signal is whether the output lands CLOSER to the
training distribution than their thresholds do -- if it does not, do not ship it.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np
import polars as pl

from .config import DATA, PARQUET, REPORTS, ROOT
from .decide import choose_k, deconflict
from .submission import (MATCHING_HEADER, OUTPUT, preflight, required_entities,
                         run_official_validator, write_tsv)

THEIRS = DATA / "external" / "v3_pair_probabilities.tsv"
TRUE_MEAN, TRUE_EMPTY = 3.461, 5.58     # training ground truth


def main(write: bool = False, max_k: int | None = None) -> dict:
    t = pl.read_csv(THEIRS, separator="\t", quote_char=None)
    t = t.rename({c: c.strip().strip('"') for c in t.columns})
    cand: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for e, c, p in zip(t["source1_entity_id"].to_list(),
                       t["candidate_entity_id"].to_list(), t["prob"].to_list()):
        cand[e].append((c, float(p)))
    del t
    print(f"their stored pairs: {sum(len(v) for v in cand.values()):,} "
          f"over {len(cand):,} entities")

    entities = required_entities("test")
    for e in entities:
        cand.setdefault(e, [])

    chosen: dict[str, list[tuple[str, float]]] = {}
    k_hist: Counter[int] = Counter()
    for e, items in cand.items():
        if not items:
            chosen[e] = []
            k_hist[0] += 1
            continue
        pr = np.fromiter((x[1] for x in items), dtype=np.float64, count=len(items))
        order = np.argsort(-pr)
        k, _ = choose_k(pr, max_k=max_k)
        k_hist[k] += 1
        chosen[e] = [items[i] for i in order[:k]] if k else []
    final = deconflict(chosen)

    n = len(entities)
    ids = sum(len(v) for v in final.values())
    empty = sum(1 for v in final.values() if not v)
    mean_pred, empty_pct = ids / n, 100 * empty / n

    # the only available signal: distance from the training distribution
    d_mean = abs(mean_pred - TRUE_MEAN)
    d_empty = abs(empty_pct - TRUE_EMPTY)
    their_mean, their_empty = 5_689_603 / n, 5.94
    print(f"\n{'':22}{'ids':>12} {'mean/entity':>12} {'empty %':>9}")
    print(f"{'their thresholds':22}{5_689_603:>12,d} {their_mean:>12.3f} {their_empty:>9.2f}")
    print(f"{'our expected-F rule':22}{ids:>12,d} {mean_pred:>12.3f} {empty_pct:>9.2f}")
    print(f"{'TRAIN TRUTH':22}{'~6,000,000':>12} {TRUE_MEAN:>12.3f} {TRUE_EMPTY:>9.2f}")
    print(f"\ndistance from truth   mean: theirs {abs(their_mean-TRUE_MEAN):.3f} -> ours {d_mean:.3f}"
          f"   |   empty: theirs {abs(their_empty-TRUE_EMPTY):.2f} -> ours {d_empty:.2f}")
    closer = (d_mean < abs(their_mean - TRUE_MEAN)) and (d_empty < abs(their_empty - TRUE_EMPTY))
    print(f"VERDICT: {'CLOSER to truth on both -- worth shipping' if closer else 'NOT closer on both -- do NOT ship'}")

    out = {"ids": ids, "mean_predictions": round(mean_pred, 3),
           "empty_pct": round(empty_pct, 2), "closer_to_truth": bool(closer),
           "k_histogram": dict(sorted(k_hist.items())[:10])}

    if write:
        pre = preflight(entities, final, candidates=None)
        if not pre["ok"]:
            print("PREFLIGHT_FAILED", pre["problems"])
            return {**out, "ok": False}
        out["matching_file"] = write_tsv(OUTPUT / "matching_results.tsv",
                                         MATCHING_HEADER, entities, final)
        td = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "test"
        cp = OUTPUT / "candidate_pairs.tsv"
        v = run_official_validator(OUTPUT / "matching_results.tsv",
                                   cp if cp.exists() else None, td)
        out["validator_passed"] = v.get("passed")
        print(f"VALIDATOR passed={v.get('passed')}")
        if v.get("passed"):
            print("THEIR_PROBS_OUR_RULE_OK")
    (REPORTS / "their_probs_our_rule.json").write_text(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--max-k", type=int)
    a = ap.parse_args()
    main(write=a.write, max_k=a.max_k)
