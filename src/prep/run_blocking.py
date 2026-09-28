"""Country x source blocking on the internal validation set, with recall audit.

Also A/B-tests the one genuinely open design question: is it better to pool S2+S3
into one index and take a global top-K, or to index each source separately and
take top-K from each?

  combined_50   : one S2+S3 index, global top-50          (previous behaviour)
  per_source_25 : S2 top-25 + S3 top-25  -> same 50 budget
  per_source_50 : S2 top-50 + S3 top-50  -> 100 budget

80.48% of entities match in BOTH sources, so a global cut can be monopolised by
one source; the comparison settles whether that actually costs recall.
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter

import numpy as np
import polars as pl

from .audit import load_pairs
from .config import ADDR_WEIGHT, CANDIDATES, K_GRID, REPORTS, REPR, VALIDATION, config_hash
from .retrieval import attach_country_key, merge_sources, retrieve_source

logging.basicConfig(level=logging.INFO, format="    %(message)s")
COLS = ["entity_id", "country", "name_tokens", "addr_tokens", "name_script"]


def load_queries() -> pl.DataFrame:
    split = pl.read_parquet(VALIDATION / "split_assignment.parquet")
    val_ids = set(split.filter(pl.col("fold") == "val")["entity_id"].to_list())
    q = pl.read_parquet(REPR / "train_source1.parquet", columns=COLS).filter(
        pl.col("entity_id").is_in(val_ids))
    return attach_country_key(q)


def load_source(src: str) -> pl.DataFrame:
    return attach_country_key(pl.read_parquet(REPR / f"train_{src}.parquet", columns=COLS))


def truth_for(query_ids: list[str]) -> dict[str, set[str]]:
    want = set(query_ids)
    t: dict[str, set[str]] = {}
    for a, b in load_pairs().filter(pl.col("source1_entity_id").is_in(want)).iter_rows():
        t.setdefault(a, set()).add(b)
    return t


def score_recall(query_ids, cands, truth, meta, q_country, paths) -> dict:
    """Recall@K overall and broken down by country, source and script."""
    ks = list(K_GRID)
    hit = {k: 0 for k in ks}
    ent = {k: [] for k in ks}
    tot = 0
    by = {dim: {k: Counter() for k in ks} for dim in ("country", "source", "script", "path")}
    den = {dim: Counter() for dim in ("country", "source", "script", "path")}

    for i, qid in enumerate(query_ids):
        t = truth.get(qid)
        if not t:
            continue
        tot += len(t)
        c_lbl, p_lbl = q_country[i], paths[i]
        for m in t:
            den["country"][c_lbl] += 1
            den["source"][m[:2]] += 1
            den["script"][meta.get(m, "?")] += 1
            den["path"][p_lbl] += 1
        for k in ks:
            got = set(cands[i][:k])
            h = t & got
            hit[k] += len(h)
            ent[k].append(len(h) / len(t))
            for m in h:
                by["country"][k][c_lbl] += 1
                by["source"][k][m[:2]] += 1
                by["script"][k][meta.get(m, "?")] += 1
                by["path"][k][p_lbl] += 1

    def pct(num, dnm):
        return round(100 * num / dnm, 2) if dnm else None

    return {
        "true_pairs": tot,
        "queries_with_truth": sum(1 for q in query_ids if truth.get(q)),
        "pair_recall": {f"@{k}": pct(hit[k], tot) for k in ks},
        "entity_recall": {f"@{k}": round(100 * float(np.mean(ent[k])), 2) for k in ks},
        "mean_candidates": round(float(np.mean([len(c) for c in cands])), 1),
        "by_country": {k: {c: pct(by["country"][k][c], den["country"][c])
                           for c in sorted(den["country"])} for k in ks},
        "by_source": {k: {s: pct(by["source"][k][s], den["source"][s])
                          for s in sorted(den["source"])} for k in ks},
        "by_script": {k: {s: pct(by["script"][k][s], den["script"][s])
                          for s in sorted(den["script"])} for k in ks},
        "by_path": {k: {p: pct(by["path"][k][p], den["path"][p])
                        for p in sorted(den["path"])} for k in ks},
        "denominators": {d: dict(den[d]) for d in den},
    }


def run_mode(mode: str, q: pl.DataFrame, s2: pl.DataFrame, s3: pl.DataFrame,
             truth: dict, meta: dict) -> dict:
    t0 = time.time()
    print(f"\n{'=' * 64}\nMODE: {mode}\n{'=' * 64}")
    if mode == "combined_50":
        pooled = pl.concat([s2, s3])
        r = retrieve_source(q, pooled, "S23", k=50)
        qids, cands, _, paths = r.query_ids, r.candidates, r.scores, r.path
        diags = [r.diagnostics]
    else:
        k = 25 if mode == "per_source_25" else 50
        r2 = retrieve_source(q, s2, "S2", k=k)
        r3 = retrieve_source(q, s3, "S3", k=k)
        qids, cands, _, paths = merge_sources([r2, r3])
        diags = [r2.diagnostics, r3.diagnostics]

    res = score_recall(qids, cands, truth, meta, q["country_key"].to_list(), paths)
    res["mode"] = mode
    res["seconds"] = round(time.time() - t0, 1)
    res["diagnostics"] = diags
    res["path_counts"] = dict(Counter(paths))
    res["_qids"], res["_cands"] = qids, cands

    print(f"  pair recall   " + "  ".join(f"@{k}={res['pair_recall'][f'@{k}']:>6.2f}%" for k in K_GRID))
    print(f"  entity recall " + "  ".join(f"@{k}={res['entity_recall'][f'@{k}']:>6.2f}%" for k in K_GRID))
    print(f"  by country @50: {res['by_country'][50]}")
    print(f"  by source  @50: {res['by_source'][50]}")
    print(f"  by script  @50: {res['by_script'][50]}")
    print(f"  paths: {res['path_counts']}   mean candidates={res['mean_candidates']}")
    print(f"  {res['seconds']}s")
    return res


def save_candidates(mode: str, qids: list[str], cands: list[list[str]], path: str) -> dict:
    """Write candidate_pairs.tsv in the official schema and self-verify it."""
    out = CANDIDATES / f"val_candidate_pairs_{mode}.tsv"
    pl.DataFrame({"source1_entity_id": qids,
                  "candidate_entity_ids": [",".join(c) for c in cands]}
                 ).write_csv(out, separator="\t")
    checks = {
        "rows": len(qids),
        "unique_source1_ids": len(set(qids)),
        "rows_with_s1_in_list": sum(1 for c in cands if any(x.startswith("S1-") for x in c)),
        "rows_with_duplicate_ids": sum(1 for c in cands if len(c) != len(set(c))),
        "rows_with_zero_candidates": sum(1 for c in cands if not c),
        "mean_candidates": round(sum(len(c) for c in cands) / len(cands), 2),
        "max_candidates": max(len(c) for c in cands),
        "file": str(out.relative_to(out.parents[2])),
        "mb": round(out.stat().st_size / 2**20, 1),
    }
    checks["ok"] = (checks["rows"] == checks["unique_source1_ids"]
                    and checks["rows_with_s1_in_list"] == 0
                    and checks["rows_with_duplicate_ids"] == 0)
    return checks


def main(modes=("combined_50", "per_source_25", "per_source_50"),
         save_mode: str | None = None) -> dict:
    q = load_queries()
    print(f"queries: {q.height:,}   country routing: "
          f"{dict(Counter(q['country_status'].to_list()))}")
    s2, s3 = load_source("source2"), load_source("source3")
    print(f"pools: S2={s2.height:,}  S3={s3.height:,}")

    truth = truth_for(q["entity_id"].to_list())
    tmeta_ids = {m for v in truth.values() for m in v}
    meta = {}
    for src in (s2, s3):
        for eid, sc in zip(src["entity_id"].to_list(), src["name_script"].to_list()):
            if eid in tmeta_ids:
                meta[eid] = sc

    out = {"config_hash": config_hash(), "addr_weight": ADDR_WEIGHT, "modes": {}}
    for m in modes:
        r = run_mode(m, q, s2, s3, truth, meta)
        if save_mode == m:
            r["output_check"] = save_candidates(m, r.pop("_qids"), r.pop("_cands"), m)
            print(f"  saved -> {r['output_check']}")
        r.pop("_qids", None); r.pop("_cands", None)
        out["modes"][m] = r

    print(f"\n{'=' * 64}\nCOMPARISON (pair recall)\n{'=' * 64}")
    print(f"  {'mode':<16} " + " ".join(f"{'@'+str(k):>8}" for k in K_GRID) + "   mean cands")
    for m in modes:
        r = out["modes"][m]
        print(f"  {m:<16} " + " ".join(f"{r['pair_recall'][f'@{k}']:>7.2f}%" for k in K_GRID)
              + f"   {r['mean_candidates']:>6.1f}")

    (REPORTS / "phase8b_country_blocking.json").write_text(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(modes=(sys.argv[1],), save_mode=sys.argv[1])
    else:
        main()
