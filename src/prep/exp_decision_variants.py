"""READ-ONLY: why does the decision rule leave 0.0457 on the table, and what fixes it?

The error analysis showed that with the SAME model probabilities, choosing k
with hindsight scores 0.9745 against our 0.9288. So the matcher is close to
right and the cut is wrong. This tests competing cuts on the frozen
predictions -- nothing is retrained.

HYPOTHESIS. The rule uses E[|T|] = sum of ALL candidate probabilities. With 100
candidates per entity, ~97 are negatives. If the calibrator maps them to ~0.05
while their true rate is ~0.001, they contribute ~4.8 phantom expected matches
on top of a true ~3.3, inflating E[|T|] and distorting every prefix score.

Variants tested:
  current      - E[|T|] = sum of all probabilities (what shipped)
  topM         - E[|T|] = sum of the M largest probabilities only
  power        - sharpen probabilities p -> p**gamma before the rule
  floor        - zero out probabilities below a floor, then sum
  fixed_tau    - plain threshold, swept (the baseline the rule should beat)
  expected_true- E[|T|] fixed to the known training mean (3.46)
"""
from __future__ import annotations

import json
import pickle
from collections import defaultdict

import numpy as np
import polars as pl

from .audit import load_pairs
from .config import DATA, REPORTS, VALIDATION
from .metric import entity_f_beta
from .train_matcher import encode, load_features

MODEL_PATH = DATA / "models" / "matcher.pkl"
CACHE = DATA / "reports" / "val_predictions.parquet"
B2, OPB2 = 0.25, 1.25


def get_predictions() -> pl.DataFrame:
    if CACHE.exists():
        return pl.read_parquet(CACHE)
    with MODEL_PATH.open("rb") as fh:
        b = pickle.load(fh)
    val = load_features(DATA / "features" / "pairs" / "val")
    X, _ = encode(val, b["features"], b["cat_cols"], cat_levels=b["cat_levels"])
    p = b["model"].predict_proba(X)[:, 1]
    out = val.select(["source1_entity_id", "candidate_entity_id", "label"]).with_columns(
        pl.Series("p", p.astype(np.float32)))
    out.write_parquet(CACHE)
    return out


def choose(probs: np.ndarray, expected_truth: float) -> int:
    """Best prefix length under E[F] = 1.25*cum_p / (0.25*E[T] + k), vs empty."""
    if probs.size == 0:
        return 0
    empty = float(np.prod(1.0 - probs))
    cum = np.cumsum(probs)
    ks = np.arange(1, probs.size + 1, dtype=np.float64)
    scores = OPB2 * cum / (B2 * expected_truth + ks)
    k = int(np.argmax(scores))
    return 0 if empty >= scores[k] else k + 1


def main() -> dict:
    pred = get_predictions()
    print(f"cached predictions: {pred.height:,} pairs")

    cand: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for e, c, p in zip(pred["source1_entity_id"].to_list(),
                       pred["candidate_entity_id"].to_list(), pred["p"].to_list()):
        cand[e].append((c, p))

    val_ids = set(pl.read_parquet(VALIDATION / "split_assignment.parquet")
                  .filter(pl.col("fold") == "val")["entity_id"].to_list())
    truth = {e: set() for e in val_ids}
    for a, c in load_pairs().filter(pl.col("source1_entity_id").is_in(val_ids)).iter_rows():
        truth[a].add(c)
    for e in val_ids:
        cand.setdefault(e, [])

    # pre-sort once
    arr: dict[str, tuple[list[str], np.ndarray]] = {}
    for e, v in cand.items():
        if not v:
            arr[e] = ([], np.array([]))
            continue
        p = np.array([x[1] for x in v], dtype=np.float64)
        o = np.argsort(-p)
        arr[e] = ([v[i][0] for i in o], p[o])

    def evaluate(kfunc, label):
        tot = 0.0
        for e, t in truth.items():
            ids, p = arr[e]
            k = kfunc(p)
            tot += entity_f_beta(ids[:k], t)
        s = tot / len(truth)
        print(f"   {label:34s} {s:.4f}")
        return s

    res: dict[str, float] = {}
    print("\n=== baseline ===")
    res["current (E[T]=sum all p)"] = evaluate(lambda p: choose(p, float(p.sum())), "current")

    print("\n=== fixed threshold sweep (rule must beat this) ===")
    best_tau, best_tau_s = None, 0.0
    for tau in np.arange(0.05, 0.96, 0.05):
        s = evaluate(lambda p, t=tau: int((p >= t).sum()), f"tau={tau:.2f}")
        if s > best_tau_s:
            best_tau, best_tau_s = tau, s
    res[f"fixed threshold (best tau={best_tau:.2f})"] = best_tau_s

    print("\n=== E[T] from top-M probabilities only ===")
    for M in (3, 5, 8, 12, 20):
        s = evaluate(lambda p, m=M: choose(p, float(p[:m].sum())), f"topM M={M}")
        res[f"topM M={M}"] = s

    print("\n=== E[T] with a probability floor ===")
    for fl in (0.02, 0.05, 0.1, 0.2, 0.3):
        s = evaluate(lambda p, f=fl: choose(p, float(p[p >= f].sum())), f"floor={fl}")
        res[f"floor={fl}"] = s

    print("\n=== sharpen probabilities p**gamma ===")
    for g in (1.5, 2.0, 3.0, 5.0):
        s = evaluate(lambda p, gg=g: choose(p ** gg, float((p ** gg).sum())), f"gamma={g}")
        res[f"gamma={g}"] = s

    print("\n=== E[T] fixed at the training mean (3.46) ===")
    for et in (2.0, 3.0, 3.46, 5.0):
        s = evaluate(lambda p, e=et: choose(p, e), f"E[T]={et}")
        res[f"E[T] fixed={et}"] = s

    print("\n=== hindsight-optimal k (ceiling for any cut) ===")
    tot = 0.0
    for e, t in truth.items():
        ids, p = arr[e]
        best = entity_f_beta([], t)
        for k in range(1, len(ids) + 1):
            sc = entity_f_beta(ids[:k], t)
            if sc > best:
                best = sc
        tot += best
    res["ORACLE k (ceiling)"] = tot / len(truth)
    print(f"   {'oracle k':34s} {res['ORACLE k (ceiling)']:.4f}")

    ranked = sorted(res.items(), key=lambda x: -x[1])
    print("\n=== RANKED ===")
    for k, v in ranked[:12]:
        print(f"   {v:.4f}  {k}")

    (REPORTS / "decision_variants.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
