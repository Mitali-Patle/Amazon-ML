"""Pins the retrieval scoring formula.

score(q,p) = SUM_{shared name tok} idf + addr_weight * SUM_{shared addr tok} idf

Applying idf on both sides of ``Q @ P.T`` silently squares it and turns the
address weight into addr_weight**2. That bug costs ~2pp recall and produces no
error, so it is asserted numerically here.
Run: .venv/bin/python -m tests.test_scoring
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.prep.candidates import build_vocab, to_matrix, topk_batch  # noqa: E402

FAILS = []


def check(cond, msg):
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        FAILS.append(msg)


ADDR_W = 0.7
# Fixture must be big enough that idf stays strictly positive after the
# clamp in build_vocab; a 3-doc corpus makes every idf collapse to 0.
FILLER = [[f"filler{i}"] for i in range(12)]
pool = pl.DataFrame({
    "entity_id": ["P1", "P2", "P3"] + [f"F{i}" for i in range(12)],
    "name_tokens": [["alpha", "beta"], ["alpha"], ["gamma"]] + FILLER,
    "addr_tokens": [["main", "street"], ["main"], ["main"]] + FILLER,
})
query = pl.DataFrame({
    "entity_id": ["Q1"],
    "name_tokens": [["alpha", "gamma"]],
    "addr_tokens": [["main"]],
})

vocab, idf, stats = build_vocab(pool)
P = to_matrix(pool, vocab, idf, ADDR_W, side="pool")
Q = to_matrix(query, vocab, idf, ADDR_W, side="query")
scores = np.asarray((Q @ P.T).todense()).ravel()

get = lambda k: idf[vocab[k]]  # noqa: E731
expect = np.array([
    get("n\x00alpha") + ADDR_W * get("a\x00main"),                      # P1: alpha, main
    get("n\x00alpha") + ADDR_W * get("a\x00main"),                      # P2: alpha, main
    get("n\x00gamma") + ADDR_W * get("a\x00main"),                      # P3: gamma, main
])

print("== score is LINEAR in idf, address weight applied exactly once ==")
for i in range(3):
    check(abs(scores[i] - expect[i]) < 1e-5,
          f"P{i+1}: got {scores[i]:.6f}, expected {expect[i]:.6f}")

print("\n== the squared-idf bug would be detectable ==")
bad = to_matrix(query, vocab, idf, ADDR_W, side="pool")
bad_scores = np.asarray((bad @ P.T).todense()).ravel()
check(abs(bad_scores[0] - scores[0]) > 1e-3,
      f"idf-on-both-sides gives {bad_scores[0]:.4f} != correct {scores[0]:.4f}")

print("\n== rarer token outranks commoner one ==")
# 'gamma' appears in 1 of 3 pool docs, 'alpha' in 2 -> gamma has higher idf
check(get("n\x00gamma") > get("n\x00alpha"), "idf(gamma) > idf(alpha)")
check(scores[2] > scores[0], "P3 (rare gamma match) outranks P1 (common alpha match)")

print("\n== side argument is validated ==")
try:
    to_matrix(query, vocab, idf, ADDR_W, side="typo")
    check(False, "invalid side should raise")
except ValueError:
    check(True, "invalid side raises ValueError")

print("\n== top-k returns descending scores and pads with -1 ==")
idx, sc = topk_batch(Q, P.T.tocsc(), k=5)
check(idx.shape == (1, 5), f"shape {idx.shape}")
check((idx[0] >= 0).sum() == 3, f"3 real hits, rest padded: {idx[0]}")
check(sc[0, 0] >= sc[0, 1] >= sc[0, 2], "scores descending")
check(idx[0, 0] == 2, "rarest match ranked first")

print("\n== empty query retrieves nothing, does not crash ==")
empty = pl.DataFrame({"entity_id": ["E"], "name_tokens": [[]], "addr_tokens": [[]]})
E = to_matrix(empty, vocab, idf, ADDR_W, side="query")
ei, _ = topk_batch(E, P.T.tocsc(), k=5)
check((ei[0] == -1).all(), "empty query -> all padding, no exception")

print()
if FAILS:
    print(f"{len(FAILS)} FAILURE(S)")
    for f in FAILS:
        print("   -", f)
    sys.exit(1)
print("ALL SCORING TESTS PASSED")
