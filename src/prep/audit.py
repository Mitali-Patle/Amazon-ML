"""PHASE 2 -- structural audit (fail-fast gate).

Re-verifies, from the parquet files, every invariant the strategy rests on.
If INV-3 (many-to-one) or INV-4 (country partition) fails, the split design
and the blocking design are both invalid and the pipeline must stop.
"""
from __future__ import annotations

import json

import polars as pl

from .config import EXPECTED_GT_PAIRS, PARQUET, REPORTS

CRITICAL = {"INV-1", "INV-2", "INV-3", "INV-4", "INV-5"}


def load_pairs() -> pl.DataFrame:
    """Explode ground truth into one row per (source1_entity_id, matched_id)."""
    gt = pl.read_parquet(PARQUET / "train_ground_truth.parquet")
    return (
        gt.filter(pl.col("matched_entity_ids") != "")
        .with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .rename({"matched_entity_ids": "matched_id"})
    )


def audit() -> dict:
    res: dict = {"invariants": {}, "stats": {}, "failed": []}

    s1 = pl.read_parquet(PARQUET / "train_source1.parquet", columns=["entity_id", "country"])
    s2 = pl.read_parquet(PARQUET / "train_source2.parquet", columns=["entity_id", "country"])
    s3 = pl.read_parquet(PARQUET / "train_source3.parquet", columns=["entity_id", "country"])
    gt = pl.read_parquet(PARQUET / "train_ground_truth.parquet")
    pairs = load_pairs()

    def record(key, ok, detail):
        res["invariants"][key] = {"ok": bool(ok), "detail": detail}
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {key}: {detail}")
        if not ok:
            res["failed"].append(key)

    # INV-1: GT <-> Source-1 identity
    s1_ids = set(s1["entity_id"].to_list())
    gt_ids = set(gt["source1_entity_id"].to_list())
    record("INV-1", s1_ids == gt_ids,
           f"GT<->S1 id sets identical | only_in_s1={len(s1_ids - gt_ids)} only_in_gt={len(gt_ids - s1_ids)}")

    # INV-2: every matched id resolves in S2/S3, and none is an S1 id
    pool_ids = set(s2["entity_id"].to_list()) | set(s3["entity_id"].to_list())
    matched = pairs["matched_id"].to_list()
    dangling = sum(1 for m in matched if m not in pool_ids)
    self_match = sum(1 for m in matched if m.startswith("S1-"))
    record("INV-2", dangling == 0 and self_match == 0,
           f"all matched ids resolve | dangling={dangling} s1_self_matches={self_match}")

    # INV-3: MANY-TO-ONE -- no S2/S3 record claimed by two S1 entities  [CRITICAL]
    n_total, n_unique = len(matched), len(set(matched))
    record("INV-3", n_total == n_unique == EXPECTED_GT_PAIRS,
           f"many-to-one | total={n_total:,} unique={n_unique:,} expected={EXPECTED_GT_PAIRS:,}")

    # INV-4: COUNTRY PARTITION -- exhaustive over all pairs  [CRITICAL]
    country = pl.concat([s1, s2, s3])
    joined = (
        pairs.join(s1.rename({"entity_id": "source1_entity_id", "country": "c1"}),
                   on="source1_entity_id", how="left")
        .join(country.rename({"entity_id": "matched_id", "country": "c2"}),
              on="matched_id", how="left")
    )
    disagree = int((joined["c1"] != joined["c2"]).sum())
    unresolved = int(joined["c2"].is_null().sum())
    record("INV-4", disagree == 0 and unresolved == 0,
           f"country agreement over {joined.height:,} pairs | disagree={disagree} unresolved={unresolved}")

    # INV-5: Source 1 is deduplicated on (normalised name, address, country)
    dupe = (
        pl.read_parquet(PARQUET / "train_source1.parquet")
        .group_by(["business_name", "business_address", "country"])
        .len()
        .filter(pl.col("len") > 1)
    )
    record("INV-5", dupe.height == 0, f"S1 exact duplicates on (name,address,country) = {dupe.height}")

    # ---- descriptive statistics (not gates) ----
    per_entity = pairs.group_by("source1_entity_id").len()
    dist = per_entity["len"].value_counts().sort("len")
    n_single = gt.height - per_entity.height
    res["stats"] = {
        "s1_entities": s1.height,
        "singletons": n_single,
        "singleton_pct": round(100 * n_single / gt.height, 2),
        "total_pairs": n_total,
        "mean_matches_per_entity": round(n_total / gt.height, 3),
        "max_matches": int(per_entity["len"].max()),
        "match_count_distribution": {0: n_single, **{int(k): int(v) for k, v in dist.iter_rows()}},
        "pairs_by_source": {
            "S2": int(pairs["matched_id"].str.starts_with("S2-").sum()),
            "S3": int(pairs["matched_id"].str.starts_with("S3-").sum()),
        },
        "country_crosstab": {
            f"{a}->{b}": int(c) for a, b, c in
            joined.group_by(["c1", "c2"]).len().iter_rows()
        },
        "unmatched_pool_records": len(pool_ids) - n_unique,
    }
    res["ok"] = not res["failed"]
    res["critical_failed"] = sorted(set(res["failed"]) & CRITICAL)
    (REPORTS / "phase2_audit.json").write_text(json.dumps(res, indent=2, default=str))
    return res


if __name__ == "__main__":
    print("PHASE 2 -- structural audit")
    r = audit()
    s = r["stats"]
    print(f"\n  singletons          {s['singletons']:>9,d} ({s['singleton_pct']}%)")
    print(f"  total true pairs    {s['total_pairs']:>9,d}  mean {s['mean_matches_per_entity']}/entity, max {s['max_matches']}")
    print(f"  pairs by source     S2={s['pairs_by_source']['S2']:,}  S3={s['pairs_by_source']['S3']:,}")
    print(f"  country cross-tab   {s['country_crosstab']}")
    print(f"  unmatched pool      {s['unmatched_pool_records']:,} S2/S3 records match nothing")
    if r["critical_failed"]:
        print(f"\nCRITICAL INVARIANT FAILED: {r['critical_failed']} -- strategy is invalid, STOP")
        raise SystemExit(2)
    if r["failed"]:
        raise SystemExit(1)
    print("\nPHASE 2 OK -- all invariants hold")
