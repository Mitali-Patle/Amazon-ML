"""PHASE 6/7/8 -- corpus statistics, sparse index, top-K retrieval, recall@K.

Design follows the measured decisions in docs/PREPROCESSING_PLAYBOOK.md:

  * retrieval is per-country TOP-K, not threshold-union: union blocking at
    df<50k produced 78e9 candidate pairs (H6);
  * score = sum(idf over shared name tokens) + ADDR_WEIGHT * sum(idf over shared
    address tokens), ADDR_WEIGHT=0.7 from a measured sweep (D9);
  * transliterated tokens are already folded into name_tokens/addr_tokens by
    phase 3 -- they are NOT a separate scored channel (D7);
  * tokens with df >= DF_STOPWORD are dropped from the index. They carry ~0 idf
    (``limited`` is 11.65% of the corpus) and dominate cost.

LEAKAGE POLICY (report section 7): document frequency is LABEL-FREE and is
computed over the pool actually being searched -- the validation pool for
validation, the test pool at inference. This mirrors what is available at
inference. Anything label-derived (thresholds, priors, calibration) is
train-only and is NOT computed here.

Implementation note: scoring is a sparse matmul Q @ P.T. Queries are processed
in batches because the product is dense-ish -- a query touches ~90k pool records,
so one batch materialises BATCH_QUERIES * ~90k nonzeros.
"""
from __future__ import annotations

import json
import time
from collections import Counter

import numpy as np
import polars as pl
import scipy.sparse as sp

from .config import ADDR_WEIGHT, BATCH_QUERIES, DF_STOPWORD, K_GRID, REPORTS


# --------------------------------------------------------------------------
# vocabulary + document frequency  (PHASE 6, label-free)
# --------------------------------------------------------------------------
def build_vocab(pool: pl.DataFrame) -> tuple[dict[str, int], np.ndarray, dict]:
    """Return (token->col index, idf vector, stats).

    Name and address occupy SEPARATE key spaces: a token means something
    different in a name than in an address, and they are weighted differently.
    """
    df: Counter[str] = Counter()
    for col, prefix in (("name_tokens", "n\x00"), ("addr_tokens", "a\x00")):
        for toks in pool[col]:
            if toks is None:
                continue
            for t in set(toks):
                df[prefix + t] += 1

    n_docs = pool.height
    kept = {k: v for k, v in df.items() if v < DF_STOPWORD}
    vocab = {k: i for i, k in enumerate(kept)}
    # Clamped at 0: a raw log(N/(1+df)) turns NEGATIVE once a token appears in
    # more than half the corpus, which would make sharing a token *penalise* a
    # candidate. The DF_STOPWORD cut keeps df/N <= ~1.5% in production so this
    # never binds there, but the clamp removes the trap for small pools.
    idf = np.zeros(len(vocab), dtype=np.float32)
    for k, i in vocab.items():
        idf[i] = max(0.0, float(np.log(n_docs / (1.0 + kept[k]))))

    stats = {
        "pool_records": n_docs,
        "distinct_keys": len(df),
        "indexed_keys": len(vocab),
        "dropped_stopword_keys": len(df) - len(vocab),
        "df_stopword_threshold": DF_STOPWORD,
        "top_dropped": [(k.replace("n\x00", "name:").replace("a\x00", "addr:"), v)
                        for k, v in df.most_common(10) if v >= DF_STOPWORD],
    }
    return vocab, idf, stats


def to_matrix(frame: pl.DataFrame, vocab: dict[str, int], idf: np.ndarray,
              addr_weight: float, side: str) -> sp.csr_matrix:
    """Rows = records, cols = vocabulary.

    The scoring target is

        score(q, p) = SUM_{shared name tok} idf  +  addr_weight * SUM_{shared addr tok} idf

    which is LINEAR in idf. Since the score is computed as ``Q @ P.T``, each
    weight must be applied on exactly ONE side, otherwise the product silently
    becomes ``idf**2`` and the address weight becomes ``addr_weight**2``. So:

        side="pool"  -> idf values
        side="query" -> 1.0 for name tokens, addr_weight for address tokens

    VECTORIZED via polars explode+join instead of a per-row Python loop (the
    row loop measured ~21 min for 110k queries against the full corpus,
    extrapolating to ~5.5-6h for the full train/test runs). Name and address
    tokens occupy disjoint column ranges in `vocab` (keys are prefixed
    ``n\x00``/``a\x00``), so the two channels can never collide on the same
    (row, col) cell -- exactly like the original per-row ``seen`` dict, which
    also could not collide across channels for the same reason. Within a
    channel, a token repeated in one row's list is deduped by (row, token)
    before the vocab lookup, matching the original ``set(tokens)`` per row.
    """
    if side not in ("pool", "query"):
        raise ValueError(f"side must be 'pool' or 'query', got {side!r}")

    n = frame.height
    n_cols = len(vocab)
    if n == 0 or n_cols == 0:
        return sp.csr_matrix((n, n_cols), dtype=np.float32)

    vocab_df = pl.DataFrame({
        "key": pl.Series(list(vocab.keys()), dtype=pl.Utf8),
        "col": np.fromiter(vocab.values(), dtype=np.int32, count=n_cols),
    })
    row_idx = np.arange(n, dtype=np.int64)

    def channel_hits(col_name: str, prefix: str, query_weight: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sub = (
            pl.DataFrame({"row": row_idx, "toks": frame[col_name]})
            .explode("toks")
            .filter(pl.col("toks").is_not_null())
        )
        if sub.height == 0:
            empty = np.empty(0, dtype=np.int32)
            return empty, empty, np.empty(0, dtype=np.float32)
        sub = sub.unique(subset=["row", "toks"], maintain_order=False)
        sub = sub.with_columns((pl.lit(prefix) + pl.col("toks")).alias("key"))
        sub = sub.join(vocab_df, on="key", how="inner")
        if sub.height == 0:
            empty = np.empty(0, dtype=np.int32)
            return empty, empty, np.empty(0, dtype=np.float32)
        rows = sub["row"].to_numpy().astype(np.int32)
        cols = sub["col"].to_numpy().astype(np.int32)
        if side == "pool":
            vals = idf[cols]
        else:
            vals = np.full(sub.height, query_weight, dtype=np.float32)
        return rows, cols, vals

    n_rows, n_cols_idx, n_vals = channel_hits("name_tokens", "n\x00", 1.0)
    a_rows, a_cols_idx, a_vals = channel_hits("addr_tokens", "a\x00", addr_weight)

    rows = np.concatenate([n_rows, a_rows])
    cols = np.concatenate([n_cols_idx, a_cols_idx])
    data = np.concatenate([n_vals, a_vals]).astype(np.float32)

    # Name/addr column ranges are disjoint (distinct key prefixes), and each
    # channel was already deduped by (row, token) above, so no (row, col)
    # pair repeats -- csr_matrix does not need to sum any duplicates here.
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n_cols))


# --------------------------------------------------------------------------
# top-K retrieval  (PHASE 7/8)
# --------------------------------------------------------------------------
def topk_batch(Q: sp.csr_matrix, P_T: sp.csc_matrix, k: int,
               batch: int = BATCH_QUERIES) -> tuple[np.ndarray, np.ndarray]:
    """Top-k pool indices and scores for each query row."""
    n = Q.shape[0]
    out_idx = np.full((n, k), -1, dtype=np.int32)
    out_score = np.zeros((n, k), dtype=np.float32)
    for start in range(0, n, batch):
        stop = min(start + batch, n)
        S = (Q[start:stop] @ P_T).tocsr()
        for r in range(stop - start):
            lo, hi = S.indptr[r], S.indptr[r + 1]
            if lo == hi:
                continue
            cols, vals = S.indices[lo:hi], S.data[lo:hi]
            if len(vals) > k:
                part = np.argpartition(-vals, k - 1)[:k]
                cols, vals = cols[part], vals[part]
            order = np.argsort(-vals)
            cols, vals = cols[order], vals[order]
            m = len(cols)
            out_idx[start + r, :m] = cols
            out_score[start + r, :m] = vals
    return out_idx, out_score


# --------------------------------------------------------------------------
# recall@K evaluation
# --------------------------------------------------------------------------
def evaluate(top_idx: np.ndarray, query_ids: list[str], pool_ids: list[str],
             truth: dict[str, set[str]], pool_script: list[str],
             k_grid=K_GRID) -> dict:
    """Pair-level recall@K plus per-script breakdown, and per-entity recall@K.

    Both are reported because the competition metric is macro-averaged PER
    ENTITY: entities with few matches carry weight far above their share of
    pairs, so pair-level recall alone would be the wrong number to optimise.
    """
    pool_pos = {pid: i for i, pid in enumerate(pool_ids)}
    res = {k: 0 for k in k_grid}
    ent_recall = {k: [] for k in k_grid}
    script_hit = {k: Counter() for k in k_grid}
    script_tot: Counter[str] = Counter()
    total_pairs = 0
    cand_counts = []

    for qi, qid in enumerate(query_ids):
        t = truth.get(qid)
        if not t:
            continue
        row = top_idx[qi]
        row = row[row >= 0]
        cand_counts.append(len(row))
        total_pairs += len(t)

        # pool positions of this entity's true matches (some may be outside the pool)
        truth_pos = {m: pool_pos.get(m) for m in t}
        for m, j in truth_pos.items():
            if j is not None:
                script_tot[pool_script[j]] += 1

        for k in k_grid:
            got = set(row[:k].tolist())
            hit = sum(1 for j in truth_pos.values() if j is not None and j in got)
            res[k] += hit
            ent_recall[k].append(hit / len(t))
            for m, j in truth_pos.items():
                if j is not None and j in got:
                    script_hit[k][pool_script[j]] += 1

    out = {
        "true_pairs": total_pairs,
        "queries": len([q for q in query_ids if truth.get(q)]),
        "mean_candidates": round(float(np.mean(cand_counts)), 1) if cand_counts else 0.0,
        "pair_recall": {f"@{k}": round(100 * res[k] / total_pairs, 2) for k in k_grid},
        "entity_recall": {f"@{k}": round(100 * float(np.mean(ent_recall[k])), 2) for k in k_grid},
        "by_script": {
            f"@{k}": {s: round(100 * script_hit[k][s] / script_tot[s], 2)
                      for s in sorted(script_tot)}
            for k in k_grid
        },
        "script_pair_counts": dict(script_tot),
    }
    return out


def write_report(name: str, payload: dict) -> None:
    (REPORTS / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str))
