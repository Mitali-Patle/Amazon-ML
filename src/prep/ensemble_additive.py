"""Additive ensemble: their v3 as the base, plus only our confident unique finds.

WHY NOT A SYMMETRIC BLEND -- measured, not theorised. The 50/50 union_pess blend
scored 0.969 against their v3's 0.974. The diff explains it exactly:

    ids they had that we dropped : 249,750
    ids we added they lacked     :  51,034
    net                          : -198,716

Averaging pulled their correct predictions below the selection threshold wherever
our weaker model disagreed. Their model validates higher (0.974 vs our 0.966), so
our disagreements are more often the wrong ones, and giving them equal weight is
actively harmful.

THIS VERSION CANNOT DROP ANYTHING THEY HAD. Their predictions are reproduced
exactly from their own probabilities and stated rule (pair >= 0.775, anchor best
>= 0.875, France +0.05 on both). We only ADD pairs where:

  * our probability is >= ADD_MIN (high confidence), and
  * the pair is not already predicted by them, and
  * the record is not already claimed by another entity (mutual exclusivity --
    verified over all 7,638,365 training pairs, so a contested record means one
    side is definitely a false merge).

Under F_0.5 a false positive costs ~2x a false negative, so ADD_MIN is set high
on purpose: additions must clear a much higher bar than their own threshold.
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from .config import DATA, PARQUET, REPORTS, ROOT
from .submission import (MATCHING_HEADER, OUTPUT, preflight, required_entities,
                         run_official_validator, write_tsv)

SCORES = DATA / "scores" / "test"
THEIRS = DATA / "external" / "v3_pair_probabilities.tsv"
PAIR_T, FRANCE_BUMP = 0.775, 0.05
# Their anchor rule (best pair must clear 0.875 or the entity is answered empty)
# forces 13,411 entities to empty DESPITE having candidate pairs, producing a
# 5.94% empty rate against a true training rate of 5.58%. Lowering it to 0.5
# yields 5.60% -- almost exactly the truth. Under F_0.5 a wrongly-empty
# non-singleton scores 0 where it could score ~0.8+, so this is real headroom.
ANCHOR_T = 0.875   # reverted: their validated value; lowering it gained only 1,748 ids


def their_predictions() -> tuple[dict[str, list[str]], set[tuple[str, str]]]:
    """Reproduce their v3 submission from their probabilities and stated rule."""
    t = pl.read_csv(THEIRS, separator="\t", quote_char=None)
    t = t.rename({c: c.strip().strip('"') for c in t.columns})
    fr = set(pl.read_parquet(PARQUET / "test_source1.parquet",
                             columns=["entity_id", "country"])
             .filter(pl.col("country") == "France")["entity_id"].to_list())
    e_col = t["source1_entity_id"].to_list()
    c_col = t["candidate_entity_id"].to_list()
    p_col = t["prob"].to_list()
    best: dict[str, float] = {}
    for e, p in zip(e_col, p_col):
        if p > best.get(e, 0.0):
            best[e] = p
    # Survivors of their thresholds, then THEIR deconfliction rule: "if a pool
    # record is claimed by several anchors, it goes to the one with the higher
    # prob". Omitting this step left 71,281 records double-claimed, which
    # preflight correctly rejected -- the many-to-one structure is verified over
    # all 7,638,365 training pairs, so a contested record is always a false merge.
    survivors: list[tuple[str, str, float]] = []
    stored: set[tuple[str, str]] = set()
    for e, c, p in zip(e_col, c_col, p_col):
        stored.add((e, c))
        bump = FRANCE_BUMP if e in fr else 0.0
        if best.get(e, 0.0) >= ANCHOR_T + bump and p >= PAIR_T + bump:
            survivors.append((e, c, p))

    owner: dict[str, tuple[str, float]] = {}
    for e, c, p in survivors:
        cur = owner.get(c)
        if cur is None or p > cur[1] or (p == cur[1] and e < cur[0]):
            owner[c] = (e, p)
    pred: dict[str, list[str]] = defaultdict(list)
    for e, c, _ in survivors:
        if owner[c][0] == e:
            pred[e].append(c)
    return dict(pred), stored


def main(add_min: float = 0.90, write: bool = True) -> dict:
    print("reconstructing their v3 predictions...")
    theirs, stored = their_predictions()
    base_ids = sum(len(v) for v in theirs.values())
    print(f"  {len(theirs):,} entities, {base_ids:,} ids  (their 0.974 submission)")

    ce_map: dict[tuple[str, str], float] = {}
    ce_path = DATA / "scores" / "test_band_ce.parquet"
    if ce_path.exists():
        ce = pl.read_parquet(ce_path)
        for a, c, p in zip(ce["source1_entity_id"].to_list(),
                           ce["candidate_entity_id"].to_list(), ce["p_ce"].to_list()):
            ce_map[(a, c)] = float(p)
        del ce
        gc.collect()

    jc = None
    jp = DATA / "models" / "joint_calibrator.pkl"
    if jp.exists():
        with jp.open("rb") as fh:
            jc = pickle.load(fh)["iso"]

    # records already claimed by their predictions -- never contest these
    claimed: dict[str, str] = {}
    for e, ids in theirs.items():
        for c in ids:
            claimed[c] = e

    final = {e: list(v) for e, v in theirs.items()}
    stats = Counter()
    for f in sorted(SCORES.glob("scores_*.parquet")):
        df = pl.read_parquet(f).filter(pl.col("p") > 0.5)
        for e, c, p in zip(df["source1_entity_id"].to_list(),
                           df["candidate_entity_id"].to_list(), df["p"].to_list()):
            ours = ce_map.get((e, c), p)
            if jc is not None:
                ours = float(jc.predict([ours])[0])
            if ours < add_min:
                continue
            if c in claimed:
                stats["skip_claimed"] += 1
                continue
            cur = final.setdefault(e, [])
            if c in cur:
                continue
            cur.append(c)
            claimed[c] = e
            stats["added"] += 1
        del df
        gc.collect()

    entities = required_entities("test")
    for e in entities:
        final.setdefault(e, [])
    added = sum(len(final[e]) for e in entities) - base_ids
    mean_pred = sum(len(v) for v in final.values()) / len(entities)
    empty = sum(1 for v in final.values() if not v)
    out = {"add_min": add_min, "their_base_ids": base_ids,
           "added_ids": int(stats["added"]), "skipped_claimed": int(stats["skip_claimed"]),
           "total_ids": base_ids + added, "mean_predictions": round(mean_pred, 3),
           "empty_predictions": empty, "empty_pct": round(100 * empty / len(entities), 2)}
    print(f"\n  base (theirs)  : {base_ids:,} ids")
    print(f"  added (ours)   : {stats['added']:,}  (skipped {stats['skip_claimed']:,} already claimed)")
    print(f"  total          : {base_ids + added:,} ids")
    print(f"  mean/entity    : {mean_pred:.3f}   (train truth 3.461)")
    print(f"  empty          : {empty:,} = {100*empty/len(entities):.2f}%   (train 5.58%)")
    print("  NOTE: nothing from their submission was removed -- additions only.")

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
        print(f"\nVALIDATOR passed={v.get('passed')}")
        if v.get("passed"):
            print("ADDITIVE_SUBMISSION_OK")
    (REPORTS / f"ensemble_additive_{add_min}.json").write_text(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--add-min", type=float, default=0.97)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    main(add_min=a.add_min, write=not a.dry_run)
