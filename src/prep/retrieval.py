"""Country x source partitioned retrieval with a bounded country-agnostic fallback.

ARCHITECTURE

    S1 record -> normalize_country -> resolve against partitions PRESENT IN THE POOL
                       |
         +-------------+--------------------+
        |                                   |
   partition exists                  missing / unknown
        |                                   |
   country x source index          country-AGNOSTIC index (same machinery,
   (S2 and S3 separately)          union of all partitions for that source)
        |                                   |
        +-------------+--------------------+
                      |
                 top-K per source -> merged candidate list

Country is a COMPUTATIONAL FILTER, never a silent discard: a record whose
country cannot be resolved still retrieves, just from a wider pool.

Partitions are discovered from the data at build time -- there is no hard-coded
country list anywhere, so an unseen country such as France gets its own index
automatically as long as the pool contains it.

Scoring (unchanged, from the approved playbook):
    score = SUM idf(shared name tokens) + ADDR_WEIGHT * SUM idf(shared addr tokens)
IDF weights are applied on the POOL side only; applying them on both sides of
``Q @ P.T`` silently squares them (see tests/test_scoring.py).
"""
from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

from .candidates import build_vocab, to_matrix, topk_batch
from .config import ADDR_WEIGHT, BATCH_QUERIES, CANDIDATE_CHUNK, DF_STOPWORD
from .country import CountryStatus, needs_fallback, normalize_country

log = logging.getLogger("retrieval")


@dataclass
class Partition:
    """One (country, source) retrieval index."""
    country: str
    source: str
    ids: list[str]
    script: list[str]
    vocab: dict[str, int]
    idf: np.ndarray
    matrix: sp.csr_matrix
    stats: dict = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str]:
        return (self.country, self.source)

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Partition({self.country}x{self.source}: {len(self.ids):,} records, "
                f"{len(self.vocab):,} keys, {self.matrix.nnz:,} nnz)")


def build_partitions(pool: pl.DataFrame, source: str,
                     addr_weight: float = ADDR_WEIGHT) -> dict[str, Partition]:
    """Build one index per country present in `pool`. Countries are DISCOVERED."""
    parts: dict[str, Partition] = {}
    countries = sorted(pool["country_key"].unique().to_list())
    for c in countries:
        sub = pool.filter(pl.col("country_key") == c)
        if sub.height == 0:
            continue
        t0 = time.time()
        vocab, idf, vstats = build_vocab(sub)
        mat = to_matrix(sub, vocab, idf, addr_weight, side="pool")
        parts[c] = Partition(
            country=c, source=source,
            ids=sub["entity_id"].to_list(),
            script=sub["name_script"].to_list(),
            vocab=vocab, idf=idf, matrix=mat,
            stats={**vstats, "build_seconds": round(time.time() - t0, 1),
                   "nnz": int(mat.nnz)},
        )
        log.info("built %s x %s: %d records, %d keys", c, source, sub.height, len(vocab))
    return parts


def build_agnostic(pool: pl.DataFrame, source: str,
                   addr_weight: float = ADDR_WEIGHT) -> Partition:
    """Country-agnostic index over the WHOLE source pool, for fallback queries.

    Built lazily by the caller only when at least one query needs it, because it
    costs roughly as much as all country partitions combined.
    """
    t0 = time.time()
    vocab, idf, vstats = build_vocab(pool)
    mat = to_matrix(pool, vocab, idf, addr_weight, side="pool")
    return Partition(
        country="*", source=source,
        ids=pool["entity_id"].to_list(),
        script=pool["name_script"].to_list(),
        vocab=vocab, idf=idf, matrix=mat,
        stats={**vstats, "build_seconds": round(time.time() - t0, 1), "nnz": int(mat.nnz)},
    )


def _retrieve_against(part: Partition, queries: pl.DataFrame, k: int,
                      addr_weight: float) -> tuple[np.ndarray, np.ndarray]:
    Q = to_matrix(queries, part.vocab, part.idf, addr_weight, side="query")
    return topk_batch(Q, part.matrix.T.tocsc(), k, batch=BATCH_QUERIES)


@dataclass
class RetrievalResult:
    """Per-query candidate ids and scores, plus which path produced them."""
    query_ids: list[str]
    candidates: list[list[str]]
    scores: list[list[float]]
    path: list[str]                    # "partition" | "fallback"
    partition_used: list[str]
    diagnostics: dict = field(default_factory=dict)


def retrieve_source(queries: pl.DataFrame, pool: pl.DataFrame, source: str,
                    k: int, addr_weight: float = ADDR_WEIGHT,
                    allow_fallback: bool = True,
                    parts: dict[str, "Partition"] | None = None,
                    agnostic_cache: dict | None = None) -> RetrievalResult:
    """Retrieve top-k from one source, routing each query by country.

    `queries` and `pool` must already carry `country_key` (from normalize_country).

    `parts` and `agnostic_cache` are optional reuse hooks for chunked/streaming
    callers (see `retrieve_source_chunked` below) that invoke this function once
    per query chunk against the SAME pool: they let the (expensive) country
    partitions and the lazily-built country-agnostic fallback index be built
    ONCE and reused across chunks, instead of once per call. Routing, scoring
    and fallback semantics are completely unchanged either way -- when both are
    left as None (the default, used by every existing caller) this behaves
    exactly as before.
    """
    available = set(pool["country_key"].unique().to_list())
    if parts is None:
        parts = build_partitions(pool, source, addr_weight)

    n = queries.height
    cand: list[list[str]] = [[] for _ in range(n)]
    scor: list[list[float]] = [[] for _ in range(n)]
    path = ["none"] * n
    part_used = [""] * n

    qkeys = queries["country_key"].to_list()
    qstatus = queries["country_status"].to_list()

    # route
    by_partition: dict[str, list[int]] = {}
    fallback_rows: list[int] = []
    for i, (kc, st) in enumerate(zip(qkeys, qstatus)):
        status = CountryStatus(st)
        if needs_fallback(status) or kc not in available:
            fallback_rows.append(i)
        else:
            by_partition.setdefault(kc, []).append(i)

    diag = {
        "source": source,
        "queries": n,
        "available_partitions": sorted(available),
        "routed_to_partition": {c: len(v) for c, v in sorted(by_partition.items())},
        "routed_to_fallback": len(fallback_rows),
        "partition_stats": {},
        "timings": {},
    }

    for c, rows in sorted(by_partition.items()):
        part = parts[c]
        t0 = time.time()
        sub = queries[rows]
        idx, sc = _retrieve_against(part, sub, k, addr_weight)
        for r, qi in enumerate(rows):
            keep = idx[r][idx[r] >= 0]
            cand[qi] = [part.ids[j] for j in keep]
            scor[qi] = sc[r][: len(keep)].tolist()
            path[qi] = "partition"
            part_used[qi] = f"{c}x{source}"
        diag["partition_stats"][f"{c}x{source}"] = {
            **part.stats, "pool_records": len(part.ids), "queries": len(rows)}
        diag["timings"][f"{c}x{source}"] = round(time.time() - t0, 1)

    if fallback_rows:
        if not allow_fallback:
            log.warning("%d queries need fallback but it is disabled; they get 0 candidates",
                        len(fallback_rows))
        else:
            t0 = time.time()
            if agnostic_cache is not None and "agnostic" in agnostic_cache:
                ag = agnostic_cache["agnostic"]
            else:
                ag = build_agnostic(pool, source, addr_weight)
                if agnostic_cache is not None:
                    agnostic_cache["agnostic"] = ag
            sub = queries[fallback_rows]
            idx, sc = _retrieve_against(ag, sub, k, addr_weight)
            for r, qi in enumerate(fallback_rows):
                keep = idx[r][idx[r] >= 0]
                cand[qi] = [ag.ids[j] for j in keep]
                scor[qi] = sc[r][: len(keep)].tolist()
                path[qi] = "fallback"
                part_used[qi] = f"*x{source}"
            diag["fallback_stats"] = {**ag.stats, "pool_records": len(ag.ids),
                                      "queries": len(fallback_rows)}
            diag["timings"][f"*x{source}"] = round(time.time() - t0, 1)
            log.info("fallback retrieved for %d queries over %d records",
                     len(fallback_rows), len(ag.ids))

    return RetrievalResult(queries["entity_id"].to_list(), cand, scor, path, part_used, diag)


def merge_sources(results: list[RetrievalResult], k_total: int | None = None
                  ) -> tuple[list[str], list[list[str]], list[list[float]], list[str]]:
    """Merge per-source candidate lists for the same query order.

    Per-source top-K is retained rather than re-truncating to a global top-K:
    80.48% of entities match in BOTH S2 and S3, so a single global cut can be
    monopolised by one source and starve the other.
    """
    qids = results[0].query_ids
    for r in results[1:]:
        if r.query_ids != qids:
            raise ValueError("merge_sources requires identical query order")
    out_c, out_s, out_p = [], [], []
    for i in range(len(qids)):
        pairs: list[tuple[str, float]] = []
        for r in results:
            pairs.extend(zip(r.candidates[i], r.scores[i]))
        seen: dict[str, float] = {}
        for cid, sc in pairs:
            if cid not in seen or sc > seen[cid]:
                seen[cid] = sc
        ranked = sorted(seen.items(), key=lambda x: -x[1])
        if k_total:
            ranked = ranked[:k_total]
        out_c.append([c for c, _ in ranked])
        out_s.append([s for _, s in ranked])
        paths = {r.path[i] for r in results}
        out_p.append("fallback" if "fallback" in paths else "partition")
    return qids, out_c, out_s, out_p


# --------------------------------------------------------------------------
# chunked / streaming candidate generation
#
# `retrieve_source` (above) is correct but, called once over the FULL query
# set, accumulates `cand`/`scor` as Python lists of strings for every query
# up front: 2.1M train queries x 100 candidates is ~210M Python strings
# (~12GB), which OOMs on a 15GB machine (1.73M test queries is comparable).
# The functions below process queries in bounded-size chunks, flush each
# chunk to a parquet shard on disk, and free it before starting the next --
# ROUTING, SCORING AND FALLBACK LOGIC ARE NOT DUPLICATED OR CHANGED: each
# chunk is handled by calling `retrieve_source` (and, for the merge helper,
# the existing `merge_sources`) exactly as before, just on a slice of queries
# at a time. Partitions and the lazy fallback index are built ONCE per source
# and passed in via `parts=`/`agnostic_cache=` so a chunked run does not pay
# to rebuild the CSR index once per chunk.
# --------------------------------------------------------------------------
def _shard_frame(r: RetrievalResult) -> pl.DataFrame:
    return pl.DataFrame({
        "source1_entity_id": r.query_ids,
        "candidate_entity_ids": [",".join(c) for c in r.candidates],
        "candidate_scores": [",".join(f"{s:.6f}" for s in sc) for sc in r.scores],
        "path": r.path,
        "partition_used": r.partition_used,
    })


def _shard_to_result(df: pl.DataFrame) -> RetrievalResult:
    qids = df["source1_entity_id"].to_list()
    cands = [c.split(",") if c else [] for c in df["candidate_entity_ids"].to_list()]
    scores = [[float(x) for x in s.split(",")] if s else []
              for s in df["candidate_scores"].to_list()]
    paths = df["path"].to_list()
    part_used = df["partition_used"].to_list() if "partition_used" in df.columns else list(paths)
    return RetrievalResult(qids, cands, scores, paths, part_used)


def retrieve_source_chunked(queries: pl.DataFrame, pool: pl.DataFrame, source: str,
                            k: int, out_dir: Path, addr_weight: float = ADDR_WEIGHT,
                            allow_fallback: bool = True,
                            chunk_size: int = CANDIDATE_CHUNK) -> list[Path]:
    """Memory-bounded version of `retrieve_source`: same routing/scoring, but
    queries are processed `chunk_size` at a time and each chunk's candidates
    are written to a parquet shard under `out_dir` and freed immediately,
    instead of being accumulated for the whole query set.

    Country partitions (and the country-agnostic fallback index, if needed at
    all) are built ONCE, up front, and reused for every chunk -- rebuilding
    them per chunk would be correct but wastes the ~30-250s partition-build
    cost once per chunk instead of once per source.

    Returns the list of shard paths, in order. Read them back with
    `read_candidate_shards(out_dir)`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = build_partitions(pool, source, addr_weight)
    agnostic_cache: dict = {}

    n = queries.height
    n_chunks = max(1, (n + chunk_size - 1) // chunk_size)
    shard_paths: list[Path] = []
    t0 = time.time()
    for ci, start in enumerate(range(0, n, chunk_size)):
        stop = min(start + chunk_size, n)
        qchunk = queries[start:stop]
        r = retrieve_source(qchunk, pool, source, k, addr_weight=addr_weight,
                            allow_fallback=allow_fallback, parts=parts,
                            agnostic_cache=agnostic_cache)
        shard = _shard_frame(r)
        shard_path = out_dir / f"shard_{ci:04d}.parquet"
        shard.write_parquet(shard_path)
        shard_paths.append(shard_path)
        log.info("%s chunk %d/%d (%d queries) -> %s  [%.1fs elapsed]",
                 source, ci + 1, n_chunks, stop - start, shard_path.name, time.time() - t0)
        del qchunk, r, shard
        gc.collect()
    return shard_paths


def merge_source_shards(source_dirs: dict[str, Path], out_dir: Path,
                        k_total: int | None = None) -> list[Path]:
    """Merge per-source chunked shards (written by `retrieve_source_chunked`,
    one directory per source, same `chunk_size` and query order) into final
    merged shards, chunk by chunk.

    Reuses `merge_sources` UNCHANGED: this only reconstructs lightweight
    `RetrievalResult` objects from each on-disk shard's strings, one chunk at
    a time, so peak memory is one chunk's worth of candidates rather than the
    whole query set.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_names = sorted(source_dirs)
    shard_lists = {s: sorted(Path(source_dirs[s]).glob("shard_*.parquet")) for s in src_names}
    n_shards = len(shard_lists[src_names[0]])
    for s in src_names:
        if len(shard_lists[s]) != n_shards:
            raise ValueError(f"shard count mismatch for source {s!r}: "
                             f"{len(shard_lists[s])} vs {n_shards} (chunking must match)")

    out_paths: list[Path] = []
    for ci in range(n_shards):
        results = [_shard_to_result(pl.read_parquet(shard_lists[s][ci])) for s in src_names]
        qids, cands, scores, paths = merge_sources(results, k_total=k_total)
        merged = pl.DataFrame({
            "source1_entity_id": qids,
            "candidate_entity_ids": [",".join(c) for c in cands],
            "candidate_scores": [",".join(f"{s:.6f}" for s in sc) for sc in scores],
            "path": paths,
        })
        out_path = out_dir / f"shard_{ci:04d}.parquet"
        merged.write_parquet(out_path)
        out_paths.append(out_path)
        del results, qids, cands, scores, paths, merged
        gc.collect()
    return out_paths


def read_candidate_shards(shard_dir: Path, pattern: str = "shard_*.parquet") -> pl.DataFrame:
    """Read back all shards written by `retrieve_source_chunked` /
    `merge_source_shards`, concatenated in shard order."""
    shard_dir = Path(shard_dir)
    paths = sorted(shard_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no shards matching {pattern!r} under {shard_dir}")
    return pl.concat([pl.read_parquet(p) for p in paths])


def attach_country_key(df: pl.DataFrame) -> pl.DataFrame:
    """Add `country_key` + `country_status`, preserving the raw `country` column."""
    keys, stats = [], []
    for raw in df["country"].to_list():
        k, s = normalize_country(raw)
        keys.append(k)
        stats.append(s.value)
    return df.with_columns([
        pl.Series("country_key", keys),
        pl.Series("country_status", stats),
    ])
