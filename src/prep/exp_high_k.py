"""What does a larger K actually buy us?

A teammate reached 0.96 on the leaderboard with LightGBM at K up to 150 and
country partitioning. Our cascade scores 0.9581 on validation but at K=50 per
source, whose oracle ceiling is only 0.9833 -- so we may be losing to a weaker
matcher simply because it searched a larger candidate set.

This measures, on the internal validation set, for K in {25, 50, 100, 150, 200}
per source:
  * pair recall
  * the ORACLE macro F_0.5 -- the ceiling a perfect matcher could reach
  * candidate volume (what the downstream cost would be)

Retrieval runs ONCE at the largest K; smaller K are prefixes of the same ranking,
so every row of the table comes from one pass.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict

import numpy as np
import polars as pl

from .audit import load_pairs
from .config import ADDR_WEIGHT, REPORTS, REPR, VALIDATION
from .metric import entity_f_beta
from .retrieval import attach_country_key, build_partitions, retrieve_source

COLS = ["entity_id", "country", "name_tokens", "addr_tokens", "name_script"]
K_MAX = 200
GRID = (25, 50, 100, 150, 200)


def main() -> dict:
    t0 = time.time()
    split = pl.read_parquet(VALIDATION / "split_assignment.parquet")
    val_ids = set(split.filter(pl.col("fold") == "val")["entity_id"].to_list())
    q = attach_country_key(
        pl.read_parquet(REPR / "train_source1.parquet", columns=COLS)
        .filter(pl.col("entity_id").is_in(val_ids)))
    print(f"queries={q.height:,}")

    truth: dict[str, set[str]] = {e: set() for e in val_ids}
    for a, c in load_pairs().filter(pl.col("source1_entity_id").is_in(val_ids)).iter_rows():
        truth[a].add(c)

    # retrieve once at K_MAX per source; smaller K are prefixes of this ranking
    ranked: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for src in ("source2", "source3"):
        tag = "S2" if src == "source2" else "S3"
        pool = attach_country_key(pl.read_parquet(REPR / f"train_{src}.parquet", columns=COLS))
        print(f"  retrieving {tag} top-{K_MAX} over {pool.height:,} records...")
        r = retrieve_source(q, pool, tag, k=K_MAX, addr_weight=ADDR_WEIGHT)
        for e, cands, scores in zip(r.query_ids, r.candidates, r.scores):
            ranked[e].extend(zip(cands, scores))
        del pool, r
        print(f"    done ({time.time()-t0:.0f}s)")

    rows = []
    for k in GRID:
        # top-k PER SOURCE, matching how the pipeline retrieves
        per_entity: dict[str, set[str]] = {}
        n_cand = 0
        for e, items in ranked.items():
            s2 = [c for c, _ in sorted([x for x in items if x[0].startswith("S2-")],
                                       key=lambda x: -x[1])[:k]]
            s3 = [c for c, _ in sorted([x for x in items if x[0].startswith("S3-")],
                                       key=lambda x: -x[1])[:k]]
            per_entity[e] = set(s2) | set(s3)
            n_cand += len(per_entity[e])
        for e in val_ids:
            per_entity.setdefault(e, set())

        tot = sum(len(v) for v in truth.values())
        hit = sum(len(truth[e] & per_entity[e]) for e in truth)
        oracle = np.mean([entity_f_beta(truth[e] & per_entity[e], truth[e]) for e in truth])
        mean_c = n_cand / len(truth)
        rows.append({
            "k_per_source": k,
            "pair_recall": round(100 * hit / tot, 3),
            "oracle_macro_f05": round(float(oracle), 4),
            "mean_candidates": round(mean_c, 1),
            "test_pairs_projected_M": round(1_732_544 * mean_c / 1e6, 1),
        })
        print(f"  K={k:<4} recall={rows[-1]['pair_recall']:>6.2f}%  "
              f"oracle={rows[-1]['oracle_macro_f05']:.4f}  "
              f"cands/entity={mean_c:.0f}  test pairs≈{rows[-1]['test_pairs_projected_M']}M")

    out = {"grid": rows, "seconds": round(time.time() - t0, 1)}
    (REPORTS / "high_k_ceiling.json").write_text(json.dumps(out, indent=2))
    print(f"\n{'K/src':>6} {'recall':>9} {'ORACLE':>9} {'cands':>7} {'test pairs':>12}")
    for r in rows:
        print(f"{r['k_per_source']:>6} {r['pair_recall']:>8.2f}% {r['oracle_macro_f05']:>9.4f} "
              f"{r['mean_candidates']:>7.0f} {r['test_pairs_projected_M']:>11.1f}M")
    return out


if __name__ == "__main__":
    main()
