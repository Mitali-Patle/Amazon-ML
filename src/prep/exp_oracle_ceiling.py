"""What is the BEST F_0.5 our candidate generation permits?

Recall@K is a proxy. The competition scores macro F_0.5 per entity, and the two
do not translate linearly -- singletons contribute 0% of pairs but 5.58% of the
score, and low-match entities are weighted far above their pair share.

This computes the ORACLE: for each validation entity, predict exactly the true
matches that appear in its candidate list (a perfect matcher, no false
positives, no missed retrievable matches). The resulting macro F_0.5 is a hard
ceiling on the entire pipeline -- no model can exceed it without better
candidate generation.

Also reports several reference points below the ceiling so the gap the matcher
has to close is visible:
  * oracle            -- perfect matcher on current candidates
  * predict-all       -- keep every candidate (max recall, terrible precision)
  * predict-nothing   -- empty everywhere (scores exactly the singleton rate)
  * oracle-no-singles -- oracle but never predicting empty, isolating how much
                         of the ceiling depends on singleton detection
"""
from __future__ import annotations

import json

import polars as pl

from .audit import load_pairs
from .config import CANDIDATES, REPORTS, REPR, VALIDATION
from .metric import macro_f_beta, score_breakdown


def main(candidates_file: str = "val_candidate_pairs_per_source_50.tsv") -> dict:
    split = pl.read_parquet(VALIDATION / "split_assignment.parquet")
    val_ids = set(split.filter(pl.col("fold") == "val")["entity_id"].to_list())

    truth: dict[str, set[str]] = {e: set() for e in val_ids}
    for a, b in load_pairs().filter(pl.col("source1_entity_id").is_in(val_ids)).iter_rows():
        truth[a].add(b)

    wide = pl.read_csv(CANDIDATES / candidates_file, separator="\t", quote_char=None)
    cands = {
        r["source1_entity_id"]: set(r["candidate_entity_ids"].split(","))
        if r["candidate_entity_ids"] else set()
        for r in wide.iter_rows(named=True)
    }

    country = dict(
        pl.read_parquet(REPR / "train_source1.parquet", columns=["entity_id", "country"])
        .filter(pl.col("entity_id").is_in(val_ids)).iter_rows())

    oracle = {e: (truth[e] & cands.get(e, set())) for e in truth}
    predict_all = {e: cands.get(e, set()) for e in truth}
    predict_none: dict[str, set[str]] = {e: set() for e in truth}
    # oracle that is forbidden from predicting empty: falls back to its single
    # best candidate, isolating the value of singleton detection
    oracle_no_singles = {
        e: (v if v else (set(list(cands.get(e, set()))[:1]))) for e, v in oracle.items()
    }

    res = {
        "candidates_file": candidates_file,
        "entities": len(truth),
        "true_pairs": sum(len(v) for v in truth.values()),
        "singletons": sum(1 for v in truth.values() if not v),
    }
    for name, pred in (("oracle", oracle), ("predict_all", predict_all),
                       ("predict_nothing", predict_none),
                       ("oracle_no_singleton_detection", oracle_no_singles)):
        res[name] = score_breakdown(pred, truth, groups=country)

    # how much of the truth was retrievable at all
    retrievable = sum(len(truth[e] & cands.get(e, set())) for e in truth)
    total = sum(len(v) for v in truth.values())
    res["pair_recall_of_candidates"] = round(100 * retrievable / total, 3)

    (REPORTS / "oracle_ceiling.json").write_text(json.dumps(res, indent=2, default=str))
    return res


if __name__ == "__main__":
    r = main()
    print(f"entities={r['entities']:,}  true pairs={r['true_pairs']:,}  "
          f"singletons={r['singletons']:,}")
    print(f"candidate pair recall = {r['pair_recall_of_candidates']}%\n")
    print(f"{'scenario':34s} {'macro F_0.5':>12s} {'micro F_0.5':>12s} {'singleton acc':>14s}")
    for k in ("oracle", "oracle_no_singleton_detection", "predict_all", "predict_nothing"):
        b = r[k]
        sa = b["singleton_accuracy"]
        print(f"{k:34s} {b['macro_f05']:>12.4f} {b['micro_f05']:>12.4f} "
              f"{(f'{sa:.3f}' if sa is not None else '-'):>14s}")
    print(f"\noracle by country: "
          f"{ {g: v['macro_f05'] for g, v in r['oracle']['by_group'].items()} }")
    o = r["oracle"]
    print(f"\nceiling detail: perfect entities={o['perfect_entities']:,} "
          f"({100*o['perfect_entities']/o['entities']:.1f}%)  "
          f"zero-scoring={o['zero_entities']:,} ({100*o['zero_entities']/o['entities']:.1f}%)")
