"""PHASE 8 -- run candidate generation on the internal validation set and measure recall.

This is the number that matters: candidate generation is a hard recall ceiling.
A true match never retrieved here cannot be recovered by any downstream matcher.

Validation entities are searched against the test-ratio validation pool built in
phase 4/5, per country, exactly as inference will work.
"""
from __future__ import annotations

import json
import time

import numpy as np
import polars as pl
import scipy.sparse as sp

from .audit import load_pairs
from .candidates import build_vocab, evaluate, to_matrix, topk_batch, write_report
from .config import (ADDR_WEIGHT, CANDIDATES, INDEXES, K_GRID, REPR, REPORTS,
                     TOP_K, VALIDATION, config_hash)


def load_pool_records() -> pl.DataFrame:
    cols = ["entity_id", "country", "name_tokens", "addr_tokens", "name_script"]
    return pl.concat([
        pl.read_parquet(REPR / "train_source2.parquet", columns=cols),
        pl.read_parquet(REPR / "train_source3.parquet", columns=cols),
    ])


def run(addr_weight: float = ADDR_WEIGHT, top_k: int = TOP_K,
        save_candidates: bool = True, pool_mode: str = "full",
        query_limit: int | None = None, tag: str = "phase8_candidate_recall") -> dict:
    """pool_mode:
        "full"  -- search the whole train S2/S3 corpus (10.32M records).
                   Test searches 9.97M, so this matches test DIFFICULTY.
        "ratio" -- the 1:5.75 ratio-matched pool from phase 5 (634k records).
                   Matches test *composition* but is ~15x smaller in absolute
                   size, which is what actually determines a true match's rank.
    """
    t_all = time.time()
    split = pl.read_parquet(VALIDATION / "split_assignment.parquet")

    val_ids = set(split.filter(pl.col("fold") == "val")["entity_id"].to_list())
    s1 = pl.read_parquet(
        REPR / "train_source1.parquet",
        columns=["entity_id", "country", "name_tokens", "addr_tokens", "name_script"],
    ).filter(pl.col("entity_id").is_in(val_ids))
    if query_limit:
        s1 = s1.head(query_limit) if s1.height <= query_limit else pl.concat(
            [s1.filter(pl.col("country") == c).head(query_limit // 2)
             for c in sorted(s1["country"].unique().to_list())])

    pool_all = load_pool_records()
    if pool_mode == "ratio":
        pool_ids_wanted = set(
            pl.read_parquet(VALIDATION / "val_pool.parquet")["entity_id"].to_list())
        pool_all = pool_all.filter(pl.col("entity_id").is_in(pool_ids_wanted))

    truth_pairs = load_pairs().filter(pl.col("source1_entity_id").is_in(val_ids))
    truth: dict[str, set[str]] = {}
    for a, b in truth_pairs.iter_rows():
        truth.setdefault(a, set()).add(b)

    overall = {
        "config_hash": config_hash(),
        "addr_weight": addr_weight,
        "top_k": top_k,
        "pool_mode": pool_mode,
        "query_limit": query_limit,
        "k_grid": list(K_GRID),
        "by_country": {},
    }
    all_rows = []

    for country in sorted(s1["country"].unique().to_list()):
        q = s1.filter(pl.col("country") == country)
        p = pool_all.filter(pl.col("country") == country)
        t0 = time.time()

        vocab, idf, vstats = build_vocab(p)
        t_vocab = time.time() - t0

        t1 = time.time()
        P = to_matrix(p, vocab, idf, addr_weight, side="pool")
        Q = to_matrix(q, vocab, idf, addr_weight, side="query")
        t_mat = time.time() - t1

        t2 = time.time()
        top_idx, top_score = topk_batch(Q, P.T.tocsc(), max(K_GRID))
        t_ret = time.time() - t2

        pool_ids = p["entity_id"].to_list()
        pool_script = p["name_script"].to_list()
        query_ids = q["entity_id"].to_list()
        ev = evaluate(top_idx, query_ids, pool_ids, truth, pool_script)

        full_space = q.height * p.height
        kept = q.height * top_k
        ev.update({
            "pool_records": p.height,
            "query_entities": q.height,
            "vocab": vstats,
            "pair_space_full": full_space,
            "pair_space_topk": kept,
            "reduction_ratio": round(full_space / kept, 1) if kept else None,
            "seconds": {"vocab": round(t_vocab, 1), "matrix": round(t_mat, 1),
                        "retrieval": round(t_ret, 1)},
            "nnz": {"pool": int(P.nnz), "query": int(Q.nnz)},
        })
        overall["by_country"][country] = ev

        print(f"\n  [{country}] pool={p.height:,} queries={q.height:,} "
              f"vocab={vstats['indexed_keys']:,} ({t_vocab:.0f}s vocab, {t_mat:.0f}s matrix, {t_ret:.0f}s retrieval)")
        print(f"      pair recall   " + "  ".join(f"@{k}={ev['pair_recall'][f'@{k}']:>6.2f}%" for k in K_GRID))
        print(f"      entity recall " + "  ".join(f"@{k}={ev['entity_recall'][f'@{k}']:>6.2f}%" for k in K_GRID))
        print(f"      by script @{top_k}: " + " | ".join(
            f"{s}:{v:.1f}%({ev['script_pair_counts'][s]:,})"
            for s, v in sorted(ev["by_script"][f"@{top_k}"].items(),
                               key=lambda x: -ev["script_pair_counts"][x[0]])))
        print(f"      mean candidates/query={ev['mean_candidates']:,.0f}  reduction={ev['reduction_ratio']:,.0f}x")

        if save_candidates:
            keep = top_idx[:, :top_k]
            for r, qid in enumerate(query_ids):
                row = keep[r][keep[r] >= 0]
                all_rows.append((qid, ",".join(pool_ids[j] for j in row)))

        del P, Q, top_idx, top_score, vocab, idf

    # aggregate across countries, weighted by true pairs
    tot = sum(c["true_pairs"] for c in overall["by_country"].values())
    overall["aggregate"] = {
        "true_pairs": tot,
        "pair_recall": {
            f"@{k}": round(sum(c["pair_recall"][f"@{k}"] * c["true_pairs"]
                               for c in overall["by_country"].values()) / tot, 2)
            for k in K_GRID
        },
        "entity_recall": {
            f"@{k}": round(sum(c["entity_recall"][f"@{k}"] * c["queries"]
                               for c in overall["by_country"].values())
                           / sum(c["queries"] for c in overall["by_country"].values()), 2)
            for k in K_GRID
        },
    }
    overall["total_seconds"] = round(time.time() - t_all, 1)

    if save_candidates:
        pl.DataFrame(
            {"source1_entity_id": [r[0] for r in all_rows],
             "candidate_entity_ids": [r[1] for r in all_rows]}
        ).write_csv(CANDIDATES / "val_candidate_pairs.tsv", separator="\t")

    write_report(tag, overall)
    return overall


if __name__ == "__main__":
    print("PHASE 8 -- candidate generation + recall on internal validation")
    r = run()
    a = r["aggregate"]
    print(f"\n  AGGREGATE over {a['true_pairs']:,} true pairs")
    print(f"    pair recall   " + "  ".join(f"@{k}={a['pair_recall'][f'@{k}']:>6.2f}%" for k in K_GRID))
    print(f"    entity recall " + "  ".join(f"@{k}={a['entity_recall'][f'@{k}']:>6.2f}%" for k in K_GRID))
    print(f"\n  total {r['total_seconds']}s")
