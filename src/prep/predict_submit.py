"""Test features -> matcher -> decision layer -> validated submission.

MEMORY DESIGN (the thing this module exists to get right).

The naive version accumulates every scored pair before deciding: 1.73M entities
x 50 candidates = 86M (id, probability) tuples, which measured 11.1GB RSS at
shard 8 of 18 and drove the machine 18GB into swap. Don't do that.

Instead, exploit a verified property of the candidate shards: each Source-1
entity lives WHOLLY inside one shard (checked: zero id overlap between shards,
because shards are chunks of entities, not of pairs). So the per-entity decision
-- which is independent across entities -- can run inside the shard loop, and
only the ~3 SELECTED candidates per entity survive it. That is ~5M tuples
instead of 86M, roughly 18x less.

Deconfliction is the one genuinely global step (two entities contesting the same
record can sit in different shards), so it runs once at the end -- but over the
compact selected set, not the full candidate space.

`candidate_pairs.tsv` is streamed straight from the candidate shards to disk and
never materialised in memory, for the same reason.

No labels exist for test. The model and the decision rule arrive already fixed
from the internal-train split; nothing here is tuned on test.
"""
from __future__ import annotations

import csv
import json
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

from .config import CANDIDATES, DATA, PARQUET, REPORTS, ROOT
from .decide import choose_k, deconflict
from .pair_features import add_pair_features, join_query_pool, load_pool_table, load_query_table
from .submission import (CANDIDATE_HEADER, MATCHING_HEADER, OUTPUT, preflight,
                         required_entities, run_official_validator, write_tsv)
from .train_matcher import encode

MODEL_PATH = DATA / "models" / "matcher.pkl"


def stream_candidate_file(shard_dir: Path, entities: list[str], out: Path) -> dict:
    """Write candidate_pairs.tsv straight from shards, never holding it in RAM."""
    have: dict[str, str] = {}
    for sh in sorted(shard_dir.glob("*.parquet")):
        df = pl.read_parquet(sh, columns=["source1_entity_id", "candidate_entity_ids"])
        for eid, ids in zip(df["source1_entity_id"].to_list(),
                            df["candidate_entity_ids"].to_list()):
            have[eid] = ids or ""
        del df
    out.parent.mkdir(parents=True, exist_ok=True)
    n_empty = 0
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n",
                       quoting=csv.QUOTE_NONE, quotechar=None, escapechar=None)
        w.writerow(CANDIDATE_HEADER)
        for eid in entities:
            v = have.get(eid, "")
            if not v:
                n_empty += 1
            w.writerow([eid, v])
    return {"rows": len(entities), "empty": n_empty,
            "mb": round(out.stat().st_size / 2**20, 2)}


def main(max_k: int | None = None) -> dict:
    t0 = time.time()
    with MODEL_PATH.open("rb") as fh:
        bundle = pickle.load(fh)
    model, feats, cat_cols, levels = (bundle["model"], bundle["features"],
                                      bundle["cat_cols"], bundle["cat_levels"])
    print(f"matcher loaded: {len(feats)} numeric + {sum(len(v) for v in levels.values())} one-hot")

    shard_dir = CANDIDATES / "test" / "merged"
    shards = sorted(shard_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError("no test candidate shards")
    print(f"test candidate shards: {len(shards)}")

    q_tab = load_query_table("test")
    pool_tab = load_pool_table("test")
    print(f"representations: queries={q_tab.height:,} pool={pool_tab.height:,}")

    # Only SELECTED candidates survive the shard loop -- see module docstring.
    selected: dict[str, list[tuple[str, float]]] = {}
    n_pairs = 0
    k_hist: Counter[int] = Counter()

    for i, sh in enumerate(shards, 1):
        wide = pl.read_parquet(sh, columns=["source1_entity_id", "candidate_entity_ids"])
        ex = (wide.with_columns(pl.col("candidate_entity_ids").str.split(","))
              .explode("candidate_entity_ids")
              .rename({"candidate_entity_ids": "candidate_entity_id"})
              .filter(pl.col("candidate_entity_id").is_not_null()
                      & (pl.col("candidate_entity_id") != "")))
        del wide
        if ex.height == 0:
            continue

        feat = add_pair_features(join_query_pool(ex, q_tab, pool_tab))
        del ex
        X, _ = encode(feat, feats, cat_cols, cat_levels=levels)
        p = model.predict_proba(X)[:, 1].astype(np.float32)
        del X
        n_pairs += feat.height

        # decide per entity INSIDE the shard; keep only the chosen prefix
        eids = feat["source1_entity_id"].to_list()
        cids = feat["candidate_entity_id"].to_list()
        del feat
        by_ent: dict[str, list[tuple[str, float]]] = {}
        for e, c, pr in zip(eids, cids, p.tolist()):
            by_ent.setdefault(e, []).append((c, pr))
        del eids, cids, p
        for e, items in by_ent.items():
            probs = np.fromiter((x[1] for x in items), dtype=np.float64, count=len(items))
            order = np.argsort(-probs)
            k, _ = choose_k(probs, max_k=max_k)
            k_hist[k] += 1
            if k:
                selected[e] = [items[j] for j in order[:k]]
            else:
                selected[e] = []
        del by_ent
        print(f"  shard {i}/{len(shards)}: {n_pairs:,} pairs scored, "
              f"{sum(len(v) for v in selected.values()):,} kept  ({time.time()-t0:.0f}s)",
              flush=True)

    print(f"\nscored {n_pairs:,} pairs in {time.time()-t0:.0f}s; "
          f"kept {sum(len(v) for v in selected.values()):,} after per-entity decision")

    entities = required_entities("test")
    for e in entities:
        selected.setdefault(e, [])

    # global step: mutual exclusivity across the whole submission
    t_d = time.time()
    matches = deconflict(selected)
    removed = sum(len(v) for v in selected.values()) - sum(len(v) for v in matches.values())
    print(f"deconflict removed {removed:,} contested ids in {time.time()-t_d:.0f}s")

    pre = preflight(entities, matches, candidates=None)
    print(f"preflight: {  {k: v for k, v in pre.items() if k != 'problems'} }")
    if not pre["ok"]:
        print("PREFLIGHT_FAILED", pre["problems"])
        return {"ok": False, "preflight": pre}

    m_info = write_tsv(OUTPUT / "matching_results.tsv", MATCHING_HEADER, entities, matches)
    c_info = stream_candidate_file(shard_dir, entities, OUTPUT / "candidate_pairs.tsv")
    print(f"matching_results.tsv: {m_info}")
    print(f"candidate_pairs.tsv:  {c_info}")

    test_dir = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "test"
    val = run_official_validator(OUTPUT / "matching_results.tsv",
                                 OUTPUT / "candidate_pairs.tsv", test_dir)

    report = {
        "test_pairs_scored": n_pairs,
        "kept_after_decision": sum(len(v) for v in selected.values()),
        "deconflict_removed": removed,
        "k_histogram": dict(sorted(k_hist.items())),
        "prediction_size_histogram": dict(Counter(len(v) for v in matches.values())),
        "empty_predictions": sum(1 for v in matches.values() if not v),
        "mean_predictions": round(sum(len(v) for v in matches.values()) / len(entities), 3),
        "matching_file": m_info, "candidate_file": c_info,
        "preflight": pre, "validator_passed": val.get("passed"),
        "total_seconds": round(time.time() - t0, 1),
    }
    (REPORTS / "predict_submit.json").write_text(json.dumps(report, indent=2, default=str))

    print(f"\nVALIDATOR passed={val.get('passed')} exit={val.get('exit_code')}")
    print(val.get("stdout", "")[-900:])
    if val.get("passed"):
        print(f"SUBMISSION_OK -> output/matching_results.tsv ({report['total_seconds']}s)")
    else:
        print("SUBMISSION_FAILED")
    return report


if __name__ == "__main__":
    main()
