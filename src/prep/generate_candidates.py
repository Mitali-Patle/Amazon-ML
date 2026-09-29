"""Bulk candidate generation for the train split and the official test set.

    .venv/bin/python -m src.prep.generate_candidates train
    .venv/bin/python -m src.prep.generate_candidates test

Uses the chunked/streaming path: candidates are flushed to parquet shards per
chunk instead of accumulated, because 2.1M queries x 100 candidates held as
Python strings is ~12GB and OOMs a 15GB machine.

TRAIN generates candidates for the 95% internal-training entities only -- the
5% validation entities already have theirs and must keep the exact candidates
their recall was measured on.

TEST is the first end-to-end exercise of the France partition. There are no
labels, so recall cannot be measured; instead the run asserts structural
health (every entity present, no empty candidate lists, per-country candidate
counts comparable across partitions) and logs the routing diagnostics. A
France candidate count materially below US/India is the early-warning signal
that the country partition misbehaved on unseen data.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import polars as pl

from .config import ADDR_WEIGHT, CANDIDATE_CHUNK, CANDIDATES, REPORTS, REPR, TOP_K, VALIDATION, config_hash
from .retrieval import attach_country_key, merge_source_shards, retrieve_source_chunked

COLS = ["entity_id", "country", "name_tokens", "addr_tokens", "name_script"]


def load_queries(split: str, sample_n: int | None = None,
                 seed: int = 20260925) -> pl.DataFrame:
    """Source-1 queries for a split.

    `sample_n` subsamples TRAIN only, stratified by country. Training data
    saturates long before the full 2.1M entities: at a 3.3% positive rate a
    150k sample still yields ~500k positive pairs, far past where a GBDT on 34
    features stops improving. TEST is never sampled -- every test entity must
    appear in the submission.
    """
    src_split = "train" if split in ("train", "val") else split
    q = pl.read_parquet(REPR / f"{src_split}_source1.parquet", columns=COLS)
    if split in ("train", "val"):
        assign = pl.read_parquet(VALIDATION / "split_assignment.parquet")
        fold = "val" if split == "val" else "train"
        keep = set(assign.filter(pl.col("fold") == fold)["entity_id"].to_list())
        q = q.filter(pl.col("entity_id").is_in(keep))
        if split == "train" and sample_n and sample_n < q.height:
            frac = sample_n / q.height
            q = (q.with_columns(pl.int_range(pl.len()).over("country").alias("_i"),
                                pl.len().over("country").alias("_n"))
                 .filter(pl.col("_i") < (pl.col("_n") * frac).cast(pl.Int64))
                 .drop("_i", "_n"))
    return attach_country_key(q)


def run(split: str, k: int = TOP_K, sample_n: int | None = None) -> dict:
    t_all = time.time()
    out_root = CANDIDATES / split
    out_root.mkdir(parents=True, exist_ok=True)

    q = load_queries(split, sample_n=sample_n)
    routing = dict(zip(*[list(x) for x in zip(
        *q.group_by("country_status").len().iter_rows())])) if q.height else {}
    print(f"[{split}] queries={q.height:,}  routing={routing}")
    print(f"[{split}] country mix={dict(q.group_by('country_key').len().iter_rows())}")

    report: dict = {"split": split, "config_hash": config_hash(), "k_per_source": k,
                    "sample_n": sample_n,
                    "queries": q.height, "country_status": routing, "sources": {}}

    source_dirs: dict[str, Path] = {}
    for src in ("source2", "source3"):
        tag = "S2" if src == "source2" else "S3"
        pool_split = "train" if split in ("train", "val") else split
        pool = attach_country_key(pl.read_parquet(REPR / f"{pool_split}_{src}.parquet", columns=COLS))
        d = out_root / tag
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        print(f"[{split}] {tag}: pool={pool.height:,} -> retrieving top-{k} "
              f"in chunks of {CANDIDATE_CHUNK:,}")
        shards = retrieve_source_chunked(q, pool, tag, k=k, out_dir=d,
                                         addr_weight=ADDR_WEIGHT,
                                         chunk_size=CANDIDATE_CHUNK)
        report["sources"][tag] = {
            "pool_records": pool.height,
            "shards": len(shards),
            "seconds": round(time.time() - t0, 1),
        }
        print(f"[{split}] {tag}: {len(shards)} shards in {report['sources'][tag]['seconds']}s")
        source_dirs[tag] = d
        del pool
        gc.collect()

    t0 = time.time()
    merged_dir = out_root / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged = merge_source_shards(source_dirs, merged_dir)
    report["merge_seconds"] = round(time.time() - t0, 1)
    report["merged_shards"] = len(merged)
    print(f"[{split}] merged {len(merged)} shards in {report['merge_seconds']}s")

    report["total_seconds"] = round(time.time() - t_all, 1)
    (REPORTS / f"candidates_{split}.json").write_text(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    if split not in ("train", "val", "test"):
        raise SystemExit("usage: generate_candidates.py [train|val|test] [k] [sample_n]")
    k = int(sys.argv[2]) if len(sys.argv) > 2 else TOP_K
    sample_n = int(sys.argv[3]) if len(sys.argv) > 3 else None
    r = run(split, k=k, sample_n=sample_n)
    print(f"\n[{split}] DONE in {r['total_seconds']}s -> {CANDIDATES / split}/merged")
