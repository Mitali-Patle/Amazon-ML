"""PHASE 9/10 -- pair-feature table: the final preprocessing handoff to modelling.

Given candidate pairs (source1_entity_id, candidate_entity_id) produced by
candidate generation (candidates.py / retrieval.py) plus the per-record
representations built in Phase 3 (represent.py), this module computes a
feature table describing how well each pair matches.

CRITICAL RULE (never relax this): every feature must be computable from the
two record representations (plus which source the candidate came from and,
if supplied, the retrieval score/rank) ALONE. Nothing is ever built from the
ground-truth match sets -- e.g. "this S2/S3 record is already claimed by
another Source-1 entity" is uncomputable at inference time and would encode
the answer. The optional `label` column is joined from ground truth for
train/validation only, is appended in a separate step AFTER all features are
finalised, and is never read back into any feature computation. Running the
pipeline with `include_label=False` must produce byte-identical feature
columns to running it with `include_label=True` -- tested in
tests/test_pair_features.py.

MEMORY-SAFETY BOUNDARY (stated explicitly because there are two very
different scales in play): the per-record representation tables (source1:
<=2.2M rows, source2+source3: <=10.3M rows) are loaded ONCE per run and kept
in memory -- this mirrors what candidates.py/retrieval.py already do when
building the CSR retrieval index over the same pools, so it is proven
feasible in this environment (20 cores / 15 GB RAM). What is NEVER
materialised in full is the O(pairs) table: 110,341 x 100 = 11M pairs for
validation, ~210M for train. That table is only ever alive one CHUNK at a
time (default 10,000 Source-1 rows x <=100 candidates = <=1M pairs) and is
written to its own parquet shard before the next chunk is built, so peak
extra memory is bounded by chunk size, not corpus size.

Chunk size is also a real speed knob, not just a memory one: each chunk
re-runs a hash join against the pool table (join_query_pool), and polars
rebuilds that hash table on every call rather than caching it across chunks.
Measured on a 5,000-entity / 500k-pair sample: chunk_size=500 (10 chunks)
took 50s total vs chunk_size=5000 (1 chunk) at 27s for the IDENTICAL
500k pairs -- the extra chunks alone cost ~23s of repeated hash-table
construction over the 10.3M-row train pool. 10,000 keeps the chunk count for
a full train run (~2.1M Source-1 rows) around ~210, a reasonable trade
against the ~1M-pair peak memory each chunk implies.

Vectorisation: token/number/suffix Jaccard, containment and exact-match
features are pure polars expressions (``list.set_intersection`` etc.),
computed on the whole chunk at once -- no Python loop. rapidfuzz has no
vectorised elementwise (as opposed to all-pairs ``cdist``) API, so
token_set_ratio/partial_ratio and the char-3gram Jaccard run in a single
Python pass per chunk with a per-chunk memoised n-gram cache (a query name
is repeated once per candidate, so caching avoids re-tokenising it up to
100x). This is the one deliberately-not-fully-vectorised part of the module;
it is benchmarked in the sample run below.

Retrieval score/rank: the current candidate file
(data/candidates/val_candidate_pairs_per_source_50.tsv) is a plain
``source1_entity_id -> comma-joined candidate ids`` table with no score or
rank attached, so those two feature columns are simply OMITTED for it (per
task spec: "if present in the input, else omitted"). The loader also accepts
a "long" pre-exploded parquet (one row per pair) with optional
``retrieval_score`` / ``retrieval_rank`` columns, for forward compatibility
with whatever candidates.py/retrieval.py emit next -- untested against a real
file today because none exists yet, but exercised by a synthetic fixture in
the test suite.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz

from .config import DATA, REPORTS, REPR
from .normalize import char_ngrams

FEATURES_DIR = DATA / "features" / "pairs"

# Columns read from a representation parquet (source1/source2/source3 are all
# the same schema, per represent.py). business_address / name_core are not
# needed by any feature in the spec, so they are dropped at load time to keep
# the once-loaded tables lean.
REPR_COLS = [
    "entity_id", "country", "business_name", "name_norm", "addr_norm",
    "name_script", "addr_script", "name_tokens", "addr_tokens",
    "legal_suffix", "addr_pin", "addr_numbers",
]

ID_COLS = ["source1_entity_id", "candidate_entity_id"]

FEATURE_COLS = [
    # name
    "name_tok_jaccard", "name_tok_contain_q_in_c", "name_tok_contain_c_in_q",
    "name_raw_exact", "name_norm_exact", "name_char3_jaccard",
    "name_token_set_ratio", "name_partial_ratio", "name_len_ratio",
    # address
    "addr_tok_jaccard", "addr_tok_contain_q_in_c", "addr_tok_contain_c_in_q",
    "addr_norm_exact", "addr_num_shared_count", "addr_num_shared_frac",
    "addr_pin_both_present", "addr_pin_equal",
    "addr_token_set_ratio", "addr_partial_ratio",
    # legal suffix
    "suffix_sets_equal", "suffix_sets_intersect", "suffix_jaccard",
    "suffix_count_q", "suffix_count_c",
    # script
    "name_script_pair", "addr_script_pair", "name_cross_script", "addr_cross_script",
    # structural
    "candidate_source", "country_equal", "q_addr_empty", "c_addr_empty", "either_addr_empty",
]

RETRIEVAL_COLS = ["retrieval_score", "retrieval_rank"]

_FLOAT_FEATURES = {
    "name_tok_jaccard", "name_tok_contain_q_in_c", "name_tok_contain_c_in_q",
    "name_char3_jaccard", "name_token_set_ratio", "name_partial_ratio", "name_len_ratio",
    "addr_tok_jaccard", "addr_tok_contain_q_in_c", "addr_tok_contain_c_in_q",
    "addr_num_shared_frac", "addr_token_set_ratio", "addr_partial_ratio", "suffix_jaccard",
}


# --------------------------------------------------------------------------
# loading representation tables (ONCE per run -- see memory-safety note above)
# --------------------------------------------------------------------------
def _load_side(path: Path, key_name: str, feat_prefix: str) -> pl.DataFrame:
    df = pl.read_parquet(path, columns=REPR_COLS)
    rename = {"entity_id": key_name}
    rename.update({c: f"{feat_prefix}_{c}" for c in REPR_COLS if c != "entity_id"})
    return df.rename(rename)


def load_query_table(split: str, repr_dir: Path = REPR) -> pl.DataFrame:
    """Source-1 representations, keyed by ``source1_entity_id``, ``q_`` prefixed.

    ``repr_dir`` defaults to the project's ``data/representations`` (via
    config.REPR) but is overridable so tests can point at tiny fixture
    parquet files instead.
    """
    return _load_side(repr_dir / f"{split}_source1.parquet", "source1_entity_id", "q")


def load_pool_table(split: str, repr_dir: Path = REPR) -> pl.DataFrame:
    """Source-2 + Source-3 representations, keyed by ``candidate_entity_id``, ``c_`` prefixed."""
    parts = [
        _load_side(repr_dir / f"{split}_source2.parquet", "candidate_entity_id", "c"),
        _load_side(repr_dir / f"{split}_source3.parquet", "candidate_entity_id", "c"),
    ]
    return pl.concat(parts, how="vertical")


def load_ground_truth(path: Path) -> dict[str, frozenset[str]]:
    """{source1_entity_id -> frozenset(matched ids)}. LABEL-ONLY use -- see module docstring."""
    gt = pl.read_parquet(path)
    out: dict[str, frozenset[str]] = {}
    for sid, matched in zip(gt["source1_entity_id"].to_list(), gt["matched_entity_ids"].to_list()):
        out[sid] = frozenset(matched.split(",")) if matched else frozenset()
    return out


# --------------------------------------------------------------------------
# candidate-pair loading (wide TSV: id -> comma-joined candidate ids)
# --------------------------------------------------------------------------
def read_wide_candidates(path: Path) -> pl.DataFrame:
    """Read the official ``source1_entity_id \t candidate_entity_ids`` TSV.

    Kept un-exploded (one row per Source-1 entity) -- this table is small
    (110,341 or ~2.1M rows of short strings) even though the pair space it
    encodes is not. Chunking happens by slicing THIS frame's rows, then
    exploding one slice at a time (see ``iter_pair_chunks``).

    Accepts either the official TSV or a DIRECTORY of parquet shards written by
    ``retrieval.merge_source_shards`` (how bulk train/test candidates are
    produced -- 2.1M x 100 candidates is streamed to shards rather than held in
    memory). Shard output carries extra ``candidate_scores``/``path`` columns;
    only the two official columns are kept so both inputs yield an identical
    frame and the downstream code cannot tell them apart.
    """
    if path.is_dir():
        shards = sorted(path.glob("*.parquet"))
        if not shards:
            raise FileNotFoundError(f"no parquet shards in {path}")
        return pl.concat([
            pl.read_parquet(s, columns=["source1_entity_id", "candidate_entity_ids"])
            for s in shards
        ])
    return pl.read_csv(path, separator="\t")


def iter_pair_chunks(wide: pl.DataFrame, chunk_size: int):
    """Yield exploded (source1_entity_id, candidate_entity_id) frames, one
    chunk of ``chunk_size`` SOURCE-1 rows at a time -- at most
    ``chunk_size * max_candidates`` pair rows alive at once."""
    n = wide.height
    for start in range(0, n, chunk_size):
        sl = wide.slice(start, chunk_size)
        yield (
            sl.with_columns(pl.col("candidate_entity_ids").str.split(","))
            .rename({"candidate_entity_ids": "candidate_entity_id"})
            .explode("candidate_entity_id")
        )


# --------------------------------------------------------------------------
# feature computation -- pure, no I/O, safe to unit-test directly
# --------------------------------------------------------------------------
def join_query_pool(pairs: pl.DataFrame, query: pl.DataFrame, pool: pl.DataFrame) -> pl.DataFrame:
    """Join a (source1_entity_id, candidate_entity_id) pair frame against the
    query/pool representation tables. Asserts every id resolves -- a silent
    join miss would otherwise show up downstream as null features, not a
    loud failure."""
    out = pairs.join(query, on="source1_entity_id", how="left")
    out = out.join(pool, on="candidate_entity_id", how="left")
    missing_q = out.filter(pl.col("q_country").is_null()).height
    missing_c = out.filter(pl.col("c_country").is_null()).height
    if missing_q or missing_c:
        raise AssertionError(
            f"pair join produced {missing_q} unresolved source1_entity_id and "
            f"{missing_c} unresolved candidate_entity_id -- representation table "
            f"does not cover every id in the candidate file"
        )
    return out


def _row_level_features(q_name: list[str], c_name: list[str],
                         q_addr: list[str], c_addr: list[str]) -> dict[str, np.ndarray]:
    """char-3gram Jaccard (name) + rapidfuzz token_set/partial ratios (name, addr).

    rapidfuzz has no vectorised elementwise API (only all-pairs ``cdist``), so
    this is a single Python pass per chunk. A query's name/address is
    repeated once per candidate (up to 100x for this competition's K), so a
    per-chunk memoised n-gram cache avoids re-tokenising the same string
    up to 100 times.
    """
    n = len(q_name)
    name_char3_jaccard = np.empty(n, dtype=np.float32)
    name_token_set_ratio = np.empty(n, dtype=np.float32)
    name_partial_ratio = np.empty(n, dtype=np.float32)
    addr_token_set_ratio = np.empty(n, dtype=np.float32)
    addr_partial_ratio = np.empty(n, dtype=np.float32)

    ngram_cache: dict[str, frozenset[str]] = {}

    def grams(s: str) -> frozenset[str]:
        g = ngram_cache.get(s)
        if g is None:
            g = frozenset(char_ngrams(s))
            ngram_cache[s] = g
        return g

    for i in range(n):
        qn, cn, qa, ca = q_name[i], c_name[i], q_addr[i], c_addr[i]
        gq, gc = grams(qn), grams(cn)
        union = gq | gc
        name_char3_jaccard[i] = (len(gq & gc) / len(union)) if union else 0.0
        name_token_set_ratio[i] = fuzz.token_set_ratio(qn, cn) / 100.0
        name_partial_ratio[i] = fuzz.partial_ratio(qn, cn) / 100.0
        addr_token_set_ratio[i] = fuzz.token_set_ratio(qa, ca) / 100.0
        addr_partial_ratio[i] = fuzz.partial_ratio(qa, ca) / 100.0

    return {
        "name_char3_jaccard": name_char3_jaccard,
        "name_token_set_ratio": name_token_set_ratio,
        "name_partial_ratio": name_partial_ratio,
        "addr_token_set_ratio": addr_token_set_ratio,
        "addr_partial_ratio": addr_partial_ratio,
    }


def add_pair_features(joined: pl.DataFrame) -> pl.DataFrame:
    """``joined`` must already carry q_*/c_* columns (see ``join_query_pool``).

    Returns a frame with exactly ID_COLS + FEATURE_COLS (+ RETRIEVAL_COLS if
    present on the input). No label, no I/O, no fitted statistic -- pure
    per-row computation, identical code path for train/val/test.
    """
    has_retrieval = all(c in joined.columns for c in RETRIEVAL_COLS)

    # ---- vectorised polars set/exact/structural features ----
    df = joined.with_columns(
        # name token sets
        pl.col("q_name_tokens").list.set_intersection("c_name_tokens").list.len().alias("_n_inter"),
        pl.col("q_name_tokens").list.set_union("c_name_tokens").list.len().alias("_n_union"),
        pl.col("q_name_tokens").list.len().alias("_n_qlen"),
        pl.col("c_name_tokens").list.len().alias("_n_clen"),
        # addr token sets
        pl.col("q_addr_tokens").list.set_intersection("c_addr_tokens").list.len().alias("_a_inter"),
        pl.col("q_addr_tokens").list.set_union("c_addr_tokens").list.len().alias("_a_union"),
        pl.col("q_addr_tokens").list.len().alias("_a_qlen"),
        pl.col("c_addr_tokens").list.len().alias("_a_clen"),
        # addr numbers
        pl.col("q_addr_numbers").list.set_intersection("c_addr_numbers").list.len().alias("_num_inter"),
        pl.col("q_addr_numbers").list.set_union("c_addr_numbers").list.len().alias("_num_union"),
        # legal suffix
        pl.col("q_legal_suffix").list.set_intersection("c_legal_suffix").list.len().alias("_suf_inter"),
        pl.col("q_legal_suffix").list.set_union("c_legal_suffix").list.len().alias("_suf_union"),
        pl.col("q_legal_suffix").list.len().alias("suffix_count_q"),
        pl.col("c_legal_suffix").list.len().alias("suffix_count_c"),
        (pl.col("q_legal_suffix").list.sort() == pl.col("c_legal_suffix").list.sort()).alias("suffix_sets_equal"),
        # exact matches
        (pl.col("q_business_name") == pl.col("c_business_name")).alias("name_raw_exact"),
        (pl.col("q_name_norm") == pl.col("c_name_norm")).alias("name_norm_exact"),
        (pl.col("q_addr_norm") == pl.col("c_addr_norm")).alias("addr_norm_exact"),
        # name length (chars)
        pl.col("q_name_norm").str.len_chars().alias("_n_qchars"),
        pl.col("c_name_norm").str.len_chars().alias("_n_cchars"),
        # addr_pin
        (pl.col("q_addr_pin") != "").alias("_q_pin_present"),
        (pl.col("c_addr_pin") != "").alias("_c_pin_present"),
        # script
        (pl.col("q_name_script") + "__" + pl.col("c_name_script")).alias("name_script_pair"),
        (pl.col("q_addr_script") + "__" + pl.col("c_addr_script")).alias("addr_script_pair"),
        (pl.col("q_name_script") != pl.col("c_name_script")).alias("name_cross_script"),
        (pl.col("q_addr_script") != pl.col("c_addr_script")).alias("addr_cross_script"),
        # structural
        (pl.col("q_country") == pl.col("c_country")).alias("country_equal"),
        (pl.col("q_addr_norm") == "").alias("q_addr_empty"),
        (pl.col("c_addr_norm") == "").alias("c_addr_empty"),
        pl.col("candidate_entity_id").str.split("-").list.first().alias("candidate_source"),
    )

    zero = lambda num, den: pl.when(den == 0).then(0.0).otherwise(num / den)  # noqa: E731

    df = df.with_columns(
        zero(pl.col("_n_inter"), pl.col("_n_union")).alias("name_tok_jaccard"),
        zero(pl.col("_n_inter"), pl.col("_n_qlen")).alias("name_tok_contain_q_in_c"),
        zero(pl.col("_n_inter"), pl.col("_n_clen")).alias("name_tok_contain_c_in_q"),
        zero(pl.col("_a_inter"), pl.col("_a_union")).alias("addr_tok_jaccard"),
        zero(pl.col("_a_inter"), pl.col("_a_qlen")).alias("addr_tok_contain_q_in_c"),
        zero(pl.col("_a_inter"), pl.col("_a_clen")).alias("addr_tok_contain_c_in_q"),
        pl.col("_num_inter").alias("addr_num_shared_count"),
        zero(pl.col("_num_inter"), pl.col("_num_union")).alias("addr_num_shared_frac"),
        zero(pl.col("_suf_inter"), pl.col("_suf_union")).alias("suffix_jaccard"),
        (pl.col("_suf_inter") > 0).alias("suffix_sets_intersect"),
        (pl.col("_q_pin_present") & pl.col("_c_pin_present")).alias("addr_pin_both_present"),
        (pl.col("_q_pin_present") & pl.col("_c_pin_present") & (pl.col("q_addr_pin") == pl.col("c_addr_pin"))).alias("addr_pin_equal"),
        pl.when(pl.max_horizontal("_n_qchars", "_n_cchars") == 0)
          .then(1.0)
          .otherwise(pl.min_horizontal("_n_qchars", "_n_cchars") / pl.max_horizontal("_n_qchars", "_n_cchars"))
          .alias("name_len_ratio"),
    )
    df = df.with_columns(
        (pl.col("q_addr_empty") | pl.col("c_addr_empty")).alias("either_addr_empty"),
    )

    # ---- row-level (Python) features: char-3gram jaccard + rapidfuzz ratios ----
    row_feats = _row_level_features(
        df["q_name_norm"].to_list(), df["c_name_norm"].to_list(),
        df["q_addr_norm"].to_list(), df["c_addr_norm"].to_list(),
    )
    df = df.with_columns([pl.Series(k, v) for k, v in row_feats.items()])

    # ---- cast booleans to compact ints for a GBDT-ready numeric table ----
    bool_cols = [
        "name_raw_exact", "name_norm_exact", "addr_norm_exact",
        "suffix_sets_equal", "suffix_sets_intersect",
        "addr_pin_both_present", "addr_pin_equal",
        "name_cross_script", "addr_cross_script",
        "country_equal", "q_addr_empty", "c_addr_empty", "either_addr_empty",
    ]
    df = df.with_columns([pl.col(c).cast(pl.Int8) for c in bool_cols])
    df = df.with_columns([
        pl.col(c).cast(pl.Float32) for c in _FLOAT_FEATURES
        if df.schema[c] != pl.Float32
    ])

    keep = list(ID_COLS) + list(FEATURE_COLS)
    if has_retrieval:
        keep += RETRIEVAL_COLS
    return df.select(keep)


def assert_features_sane(df: pl.DataFrame) -> None:
    """No NaN/inf in any float feature column; no null anywhere in the feature set."""
    float_cols = [c for c in FEATURE_COLS if c in _FLOAT_FEATURES]
    for c in float_cols:
        s = df[c]
        if s.is_nan().any():
            raise AssertionError(f"feature {c!r} contains NaN")
        if s.is_infinite().any():
            raise AssertionError(f"feature {c!r} contains inf")
    for c in ID_COLS + FEATURE_COLS:
        if df[c].null_count() > 0:
            raise AssertionError(f"feature/id column {c!r} contains {df[c].null_count()} nulls")


def attach_labels(features: pl.DataFrame, gt_map: dict[str, frozenset[str]]) -> pl.DataFrame:
    """Append `label` in a separate, final step. NEVER feed this column back
    into feature computation -- it is joined from ground truth, which is
    unavailable at test time by definition."""
    s1 = features["source1_entity_id"].to_list()
    cid = features["candidate_entity_id"].to_list()
    label = np.fromiter(
        (1 if c in gt_map.get(s, ()) else 0 for s, c in zip(s1, cid)),
        dtype=np.int8, count=len(s1),
    )
    return features.with_columns(pl.Series("label", label))


# --------------------------------------------------------------------------
# orchestration: chunked, sharded, memory-bounded
# --------------------------------------------------------------------------
def run(
    candidates_path: Path,
    split: str,
    out_dir: Path,
    chunk_size: int = 10_000,
    ground_truth_path: Path | None = None,
    include_label: bool = True,
    sample_n: int | None = None,
    report_name: str | None = None,
    repr_dir: Path = REPR,
) -> dict:
    """Compute the pair-feature table and write it as parquet shards.

    ``split`` selects which representation files back the query/pool joins
    ("train" for the validation candidates -- validation entities are
    searched against the full TRAIN pool per the retrieval report -- or
    "test" for real inference). ``sample_n`` truncates to the first N
    Source-1 rows of the candidate file, for smoke tests.
    """
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob(f"{split}_shard_*.parquet"):
        p.unlink()

    wide = read_wide_candidates(candidates_path)
    if sample_n is not None:
        wide = wide.head(sample_n)

    t_load0 = time.time()
    query = load_query_table(split, repr_dir)
    pool = load_pool_table(split, repr_dir)
    t_load = time.time() - t_load0

    gt_map = None
    if include_label and ground_truth_path is not None:
        gt_map = load_ground_truth(ground_truth_path)

    n_shards = 0
    n_pairs = 0
    label_pos = 0
    schema: list[str] | None = None
    t_features = 0.0

    for chunk in iter_pair_chunks(wide, chunk_size):
        joined = join_query_pool(chunk, query, pool)
        tf0 = time.time()
        feats = add_pair_features(joined)
        t_features += time.time() - tf0
        assert_features_sane(feats)
        if gt_map is not None:
            feats = attach_labels(feats, gt_map)
            label_pos += int(feats["label"].sum())
        if schema is None:
            schema = feats.columns
        feats.write_parquet(out_dir / f"{split}_shard_{n_shards:05d}.parquet")
        n_shards += 1
        n_pairs += feats.height

    elapsed = time.time() - t0
    report = {
        "split": split,
        "candidates_path": str(candidates_path),
        "source1_rows": wide.height,
        "n_pairs": n_pairs,
        "n_shards": n_shards,
        "schema": schema,
        "label_positive_pairs": label_pos if gt_map is not None else None,
        "label_positive_rate": round(label_pos / n_pairs, 6) if gt_map is not None and n_pairs else None,
        "seconds_loading_repr_tables": round(t_load, 2),
        "seconds_feature_compute": round(t_features, 2),
        "seconds_total": round(elapsed, 2),
        "pairs_per_second": round(n_pairs / t_features, 1) if t_features > 0 else None,
    }
    if report_name:
        REPORTS.mkdir(parents=True, exist_ok=True)
        (REPORTS / f"{report_name}.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 9/10 pair-feature builder")
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--out-dir", type=Path, default=FEATURES_DIR)
    ap.add_argument("--chunk-size", type=int, default=10_000)
    ap.add_argument("--ground-truth", type=Path, default=None)
    ap.add_argument("--no-label", action="store_true")
    ap.add_argument("--sample-n", type=int, default=None)
    ap.add_argument("--debug", action="store_true", help="smoke test: ~200 source1 rows")
    ap.add_argument("--report-name", type=str, default=None)
    ap.add_argument("--repr-dir", type=Path, default=REPR)
    args = ap.parse_args()

    sample_n = 200 if args.debug else args.sample_n
    out_dir = args.out_dir / (args.split + ("_debug" if args.debug else ""))
    report = run(
        candidates_path=args.candidates,
        split=args.split,
        out_dir=out_dir,
        chunk_size=args.chunk_size,
        ground_truth_path=args.ground_truth,
        include_label=not args.no_label,
        sample_n=sample_n,
        report_name=args.report_name,
        repr_dir=args.repr_dir,
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
