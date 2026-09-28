"""Memory-safe ensemble: our cascade + the teammate's v3, blended per shard.

The straightforward version loads every scored pair from both models and joins.
That is 519M pairs on our side -- tens of GB in polars -- and it OOMs, which is
what killed pass C. This streams instead.

TWO FACTS MAKE STREAMING SAFE
  1. Each Source-1 entity lives wholly inside ONE score shard (shards are chunks
     of entities, not of pairs), so the per-entity decision can run inside the
     shard loop.
  2. Only pairs either model considers plausible can affect the outcome. Ours
     below PRUNE never survive the decision rule, and theirs are already cut at
     0.10 by their own pipeline. Everything else is discarded on read.

Deconfliction is the one genuinely global step (two entities can contest the
same record from different shards), so it runs once at the end over the compact
selected set -- a few million rows, not hundreds of millions.
"""
from __future__ import annotations

import argparse
import gc
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from .config import DATA, PARQUET, REPORTS, ROOT
from .decide import choose_k, deconflict
from .submission import (CANDIDATE_HEADER, MATCHING_HEADER, OUTPUT, preflight,
                         required_entities, run_official_validator, write_tsv)

SCORES = DATA / "scores" / "test"
THEIRS = DATA / "external" / "v3_pair_probabilities.tsv"
PRUNE = 0.01          # our pairs below this cannot survive the decision rule
ASSUMED_LOW = 0.05    # stand-in for "the other model did not store this pair"


def load_theirs() -> dict[tuple[str, str], float]:
    t = pl.read_csv(THEIRS, separator="\t", quote_char=None)
    t = t.rename({c: c.strip().strip('"') for c in t.columns})
    d: dict[tuple[str, str], float] = {}
    for a, c, p in zip(t["source1_entity_id"].to_list(),
                       t["candidate_entity_id"].to_list(), t["prob"].to_list()):
        d[(a, c)] = float(p)
    return d


def main(policy: str = "union_pess", w: float = 0.5, write: bool = True) -> dict:
    import pickle
    print("loading teammate probabilities...")
    theirs = load_theirs()
    print(f"  {len(theirs):,} pairs")

    ce_path = DATA / "scores" / "test_band_ce.parquet"
    ce_map: dict[tuple[str, str], float] = {}
    if ce_path.exists():
        ce = pl.read_parquet(ce_path)
        for a, c, p in zip(ce["source1_entity_id"].to_list(),
                           ce["candidate_entity_id"].to_list(), ce["p_ce"].to_list()):
            ce_map[(a, c)] = float(p)
        del ce
        gc.collect()
        print(f"  cross-encoder scores: {len(ce_map):,}")

    jc = None
    jp = DATA / "models" / "joint_calibrator.pkl"
    if jp.exists():
        with jp.open("rb") as fh:
            jc = pickle.load(fh)["iso"]
        print("  joint calibrator loaded")

    selected: dict[str, list[tuple[str, float]]] = {}
    seen_pairs: set[tuple[str, str]] = set()
    stats = Counter()

    for f in sorted(SCORES.glob("scores_*.parquet")):
        df = pl.read_parquet(f).filter(pl.col("p") > PRUNE)
        by_ent: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for e, c, p in zip(df["source1_entity_id"].to_list(),
                           df["candidate_entity_id"].to_list(), df["p"].to_list()):
            key = (e, c)
            ours = ce_map.get(key, p)          # cross-encoder overrides inside the band
            if jc is not None:
                ours = float(jc.predict([ours])[0])
            th = theirs.get(key)
            if th is None:
                blended = w * ours + (1 - w) * ASSUMED_LOW if policy == "union_pess" else (
                    ours if policy == "union_mean" else 0.0)
                stats["ours_only"] += 1
            else:
                blended = w * ours + (1 - w) * th
                stats["both"] += 1
            if blended > 0:
                by_ent[e].append((c, blended))
            seen_pairs.add(key)
        del df

        for e, items in by_ent.items():
            pr = np.fromiter((x[1] for x in items), dtype=np.float64, count=len(items))
            order = np.argsort(-pr)
            k, _ = choose_k(pr)
            selected[e] = [items[i] for i in order[:k]] if k else []
        del by_ent
        gc.collect()
        print(f"  {f.name}: {len(selected):,} entities decided", flush=True)

    # pairs only THEY retrieved -- our blocking never saw them, so they are pure
    # added recall. Under union policies they get a discounted score and compete
    # on equal terms in a second decision pass for the affected entities.
    if policy != "intersect":
        extra: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for (e, c), th in theirs.items():
            if (e, c) not in seen_pairs:
                v = w * ASSUMED_LOW + (1 - w) * th if policy == "union_pess" else th
                extra[e].append((c, v))
                stats["theirs_only"] += 1
        for e, items in extra.items():
            merged = {c: p for c, p in selected.get(e, [])}
            for c, p in items:
                merged[c] = max(merged.get(c, 0.0), p)
            arr = list(merged.items())
            pr = np.fromiter((x[1] for x in arr), dtype=np.float64, count=len(arr))
            order = np.argsort(-pr)
            k, _ = choose_k(pr)
            selected[e] = [arr[i] for i in order[:k]] if k else []
        del extra
    print(f"  pair provenance: {dict(stats)}")

    entities = required_entities("test")
    for e in entities:
        selected.setdefault(e, [])
    matches = deconflict(selected)

    mean_pred = sum(len(v) for v in matches.values()) / len(entities)
    empty = sum(1 for v in matches.values() if not v)
    out = {"policy": policy, "w": w, "provenance": dict(stats),
           "mean_predictions": round(mean_pred, 3),
           "empty_predictions": empty,
           "empty_pct": round(100 * empty / len(entities), 2)}
    print(f"\n  mean predictions/entity = {mean_pred:.3f}   (train truth 3.46)")
    print(f"  empty = {empty:,} = {100*empty/len(entities):.2f}%   (train singleton 5.58%)")

    if write:
        pre = preflight(entities, matches, candidates=None)
        if not pre["ok"]:
            print("PREFLIGHT_FAILED", pre["problems"])
            return {**out, "ok": False}
        info = write_tsv(OUTPUT / "matching_results.tsv", MATCHING_HEADER, entities, matches)
        out["matching_file"] = info
        test_dir = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "test"
        cp = OUTPUT / "candidate_pairs.tsv"
        v = run_official_validator(OUTPUT / "matching_results.tsv",
                                   cp if cp.exists() else None, test_dir)
        out["validator_passed"] = v.get("passed")
        print(f"\nVALIDATOR passed={v.get('passed')}")
        print(v.get("stdout", "")[-400:])
        if v.get("passed"):
            print("ENSEMBLE_SUBMISSION_OK")
    (REPORTS / f"ensemble_stream_{policy}.json").write_text(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="union_pess",
                    choices=["intersect", "union_mean", "union_pess"])
    ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    main(policy=a.policy, w=a.w, write=not a.dry_run)
