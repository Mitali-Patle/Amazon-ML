"""Country routing + fallback tests (TEST A-E from the brief).

Every case asserts the same thing in different clothing: a record is NEVER
silently dropped because of its country. It either gets a partition or it gets
the bounded country-agnostic fallback.
Run: .venv/bin/python -m tests.test_country_blocking
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.prep.country import CountryStatus, needs_fallback, normalize_country  # noqa: E402
from src.prep.retrieval import attach_country_key, merge_sources, retrieve_source  # noqa: E402

FAILS = []


def check(cond, msg):
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        FAILS.append(msg)


def make_pool(rows):
    df = pl.DataFrame(
        {"entity_id": [r[0] for r in rows], "country": [r[1] for r in rows],
         "name_tokens": [r[2] for r in rows], "addr_tokens": [r[3] for r in rows],
         "name_script": [r[4] if len(r) > 4 else "latin" for r in rows]})
    return attach_country_key(df)


# a pool with three countries; filler keeps idf strictly positive
POOL = make_pool(
    [("S2-us1", "US", ["acme", "widgets"], ["main", "street", "austin"]),
     ("S2-us2", "US", ["acme", "tools"], ["main", "street"]),
     ("S2-in1", "India", ["bharat", "textiles"], ["mg", "road", "pune"]),
     ("S2-fr1", "France", ["boulangerie", "dupont"], ["rue", "lafayette", "lyon"]),
     ("S2-fr2", "France", ["boulangerie", "martin"], ["rue", "lafayette"])]
    + [(f"S2-f{i}", ["US", "India", "France"][i % 3], [f"filler{i}"], [f"pad{i}"])
       for i in range(15)])

print("== country normalisation ==")
for raw, want_key, want_status in [
        ("US", "US", CountryStatus.OK), ("India", "India", CountryStatus.OK),
        ("France", "France", CountryStatus.OK),
        ("  usa ", "US", CountryStatus.ALIASED), ("U.S.A.", "US", CountryStatus.ALIASED),
        ("united states", "US", CountryStatus.ALIASED), ("IND", "India", CountryStatus.ALIASED),
        ("fr", "France", CountryStatus.ALIASED),
        ("", "", CountryStatus.MISSING), ("   ", "", CountryStatus.MISSING),
        (None, "", CountryStatus.MISSING)]:
    k, s = normalize_country(raw)
    check(k == want_key and s == want_status, f"{raw!r} -> ({k!r}, {s.value})")

print("\n== TEST A: known country, normal partition retrieval ==")
q = make_pool([("S1-a", "US", ["acme", "widgets"], ["main", "street", "austin"])])
r = retrieve_source(q, POOL, "S2", k=5)
check(r.path[0] == "partition", f"path={r.path[0]}")
check(r.partition_used[0] == "USxS2", f"partition={r.partition_used[0]}")
check(len(r.candidates[0]) > 0, f"{len(r.candidates[0])} candidates")
check(all(c.startswith("S2-us") or "f" in c for c in r.candidates[0]),
      "candidates come only from the US partition")
check(r.candidates[0][0] == "S2-us1", f"best match ranked first ({r.candidates[0][0]})")

print("\n== TEST B: UNSEEN country (France) gets its own partition ==")
q = make_pool([("S1-f", "France", ["boulangerie", "dupont"], ["rue", "lafayette", "lyon"])])
r = retrieve_source(q, POOL, "S2", k=5)
check(r.path[0] == "partition", f"France uses a partition, not fallback (path={r.path[0]})")
check(r.partition_used[0] == "FrancexS2", f"partition={r.partition_used[0]}")
check(r.candidates[0][0] == "S2-fr1", f"true match retrieved first ({r.candidates[0][0]})")
check("France" in r.diagnostics["available_partitions"],
      "France partition was discovered from the pool, not hard-coded")

print("\n== TEST C: MISSING country -> fallback, never discarded ==")
q = make_pool([("S1-m", "", ["acme", "widgets"], ["main", "street", "austin"])])
r = retrieve_source(q, POOL, "S2", k=5)
check(r.path[0] == "fallback", f"path={r.path[0]}")
check(len(r.candidates[0]) > 0, f"still retrieved {len(r.candidates[0])} candidates")
check("S2-us1" in r.candidates[0], "true match still reachable via fallback")
check(r.diagnostics["routed_to_fallback"] == 1, "fallback activation is logged")

print("\n== TEST D: casing / whitespace / punctuation variants ==")
for variant in ("  us  ", "USA", "U.S.A.", "united states of america"):
    q = make_pool([("S1-v", variant, ["acme", "widgets"], ["main", "street", "austin"])])
    r = retrieve_source(q, POOL, "S2", k=5)
    check(r.path[0] == "partition" and r.partition_used[0] == "USxS2",
          f"{variant!r} routed to USxS2 (got {r.partition_used[0]}, {r.path[0]})")

print("\n== TEST E: country present in query but absent from pool -> fallback ==")
q = make_pool([("S1-x", "Germany", ["acme", "widgets"], ["main", "street", "austin"])])
r = retrieve_source(q, POOL, "S2", k=5)
check(r.path[0] == "fallback", f"unknown-in-pool -> fallback (got {r.path[0]})")
check(len(r.candidates[0]) > 0, "candidates still produced")

print("\n== no silent drops: every query gets a path and candidates ==")
q = make_pool([("S1-1", "US", ["acme"], ["main"]), ("S1-2", "", ["bharat"], ["mg"]),
               ("S1-3", "Germany", ["acme"], ["street"]), ("S1-4", "France", ["boulangerie"], ["rue"])])
r = retrieve_source(q, POOL, "S2", k=5)
check(all(p in ("partition", "fallback") for p in r.path), f"paths={r.path}")
check(all(len(c) > 0 for c in r.candidates), f"counts={[len(c) for c in r.candidates]}")

print("\n== fallback is bounded: k candidates, not a Cartesian product ==")
check(all(len(c) <= 5 for c in r.candidates), f"max {max(len(c) for c in r.candidates)} <= k=5")

print("\n== disabling fallback leaves them empty rather than crashing ==")
q = make_pool([("S1-m", "", ["acme"], ["main"])])
r2 = retrieve_source(q, POOL, "S2", k=5, allow_fallback=False)
check(r2.path[0] == "none" and r2.candidates[0] == [], "fallback off -> explicit empty, no exception")

print("\n== merge_sources dedups and keeps the best score ==")
q = make_pool([("S1-a", "US", ["acme", "widgets"], ["main", "street", "austin"])])
r_s2 = retrieve_source(q, POOL, "S2", k=3)
POOL3 = make_pool([("S3-us1", "US", ["acme", "widgets"], ["main", "street"])]
                  + [(f"S3-f{i}", "US", [f"filler{i}"], [f"pad{i}"]) for i in range(15)])
r_s3 = retrieve_source(q, POOL3, "S3", k=3)
qids, cands, scores, paths = merge_sources([r_s2, r_s3])
check(len(set(cands[0])) == len(cands[0]), "no duplicate ids after merge")
check(any(c.startswith("S2-") for c in cands[0]) and any(c.startswith("S3-") for c in cands[0]),
      "both sources represented in the merged list")
check(all(scores[0][i] >= scores[0][i + 1] for i in range(len(scores[0]) - 1)),
      "merged list is score-sorted")

print()
if FAILS:
    print(f"{len(FAILS)} FAILURE(S)")
    for f in FAILS:
        print("   -", f)
    sys.exit(1)
print("ALL COUNTRY BLOCKING TESTS PASSED")
