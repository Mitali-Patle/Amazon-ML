"""PHASE 4 -- leakage-safe 95/5 internal split, and PHASE 5 -- validation pool.

Split unit is the Source-1 entity, stratified by country, seeded.

Why entity-level grouping is sufficient (report H9): the match relation is
strictly many-to-one (INV-3), so the bipartite match graph is a FOREST OF STARS
centred on Source-1 entities. No connected component can span two entities, so
component-level grouping and entity-level grouping are the same partition and
no transitive leakage is possible. This is asserted, not assumed.

Validation pool (D12): validation entities must be searched against a pool with
the same distractor density as test (S1 : S2+S3 = 1 : 5.75), not against the 5%
slice of S2/S3 -- which would be ~20x too easy and would inflate every number.
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl

from .config import PARQUET, REPORTS, SEED, TEST_POOL_RATIO, VAL_FRACTION, VALIDATION
from .audit import load_pairs


def build_split() -> dict:
    rng = np.random.default_rng(SEED)
    s1 = pl.read_parquet(PARQUET / "train_source1.parquet", columns=["entity_id", "country"])

    # stratified by country, deterministic given SEED
    assign = []
    for country in sorted(s1["country"].unique().to_list()):
        ids = s1.filter(pl.col("country") == country)["entity_id"].to_numpy()
        order = rng.permutation(len(ids))
        n_val = int(round(len(ids) * VAL_FRACTION))
        fold = np.array(["train"] * len(ids), dtype=object)
        fold[order[:n_val]] = "val"
        assign.append(pl.DataFrame({"entity_id": ids, "country": country, "fold": fold}))
    split = pl.concat(assign)
    split.write_parquet(VALIDATION / "split_assignment.parquet", compression="zstd")

    pairs = load_pairs()
    fold_of = dict(zip(split["entity_id"], split["fold"]))
    pairs = pairs.with_columns(
        pl.col("source1_entity_id").replace_strict(fold_of, default=None).alias("fold")
    )

    # ---- leakage assertions ----
    errors = []
    # A: every matched record is owned by exactly one entity, hence one fold
    owner_folds = pairs.group_by("matched_id").agg(pl.col("fold").n_unique().alias("nf"))
    multi = int((owner_folds["nf"] > 1).sum())
    if multi:
        errors.append(f"{multi} S2/S3 records are claimed across both folds (violates INV-3)")
    # B: split covers the full entity set exactly once
    if split.height != s1.height:
        errors.append(f"split rows {split.height} != S1 rows {s1.height}")
    if split["entity_id"].n_unique() != split.height:
        errors.append("duplicate entity_id in split assignment")

    val = split.filter(pl.col("fold") == "val")
    train = split.filter(pl.col("fold") == "train")

    # ---- validation pool sized to the TEST distractor ratio ----
    val_ids = set(val["entity_id"].to_list())
    val_owned = pairs.filter(pl.col("fold") == "val")["matched_id"].to_list()
    owned_all = set(pairs["matched_id"].to_list())

    pool_rows, pool_stats = [], {}
    for country in sorted(val["country"].unique().to_list()):
        n_val_ent = int((val["country"] == country).sum())
        target = int(round(n_val_ent * TEST_POOL_RATIO))
        owned_c = [m for m in val_owned]  # filtered per-country below via lookup
        pool_stats[country] = {"val_entities": n_val_ent, "target_pool": target}

    # country lookup for S2/S3 records
    s23 = pl.concat([
        pl.read_parquet(PARQUET / "train_source2.parquet", columns=["entity_id", "country"]),
        pl.read_parquet(PARQUET / "train_source3.parquet", columns=["entity_id", "country"]),
    ])
    owned_by_val = set(val_owned)
    s23 = s23.with_columns([
        pl.col("entity_id").is_in(owned_by_val).alias("owned_by_val"),
        pl.col("entity_id").is_in(owned_all).alias("owned_by_any"),
    ])

    for country in sorted(val["country"].unique().to_list()):
        sub = s23.filter(pl.col("country") == country)
        must = sub.filter(pl.col("owned_by_val"))["entity_id"].to_numpy()
        # distractors: records owned by TRAIN entities or owned by nobody.
        # Including train-owned records is correct -- at test time an entity is
        # searched against records belonging to other entities too.
        distract = sub.filter(~pl.col("owned_by_val"))["entity_id"].to_numpy()
        target = pool_stats[country]["target_pool"]
        n_pad = max(0, target - len(must))
        n_pad = min(n_pad, len(distract))
        picked = rng.choice(distract, size=n_pad, replace=False) if n_pad else np.array([], dtype=object)
        ids = np.concatenate([must, picked])
        pool_rows.append(pl.DataFrame({"entity_id": ids, "country": country}))
        pool_stats[country].update({
            "owned_by_val": int(len(must)),
            "distractors_added": int(n_pad),
            "pool_size": int(len(ids)),
            "achieved_ratio": round(len(ids) / pool_stats[country]["val_entities"], 3),
        })

    pool = pl.concat(pool_rows)
    pool.write_parquet(VALIDATION / "val_pool.parquet", compression="zstd")

    for c, st in pool_stats.items():
        if abs(st["achieved_ratio"] - TEST_POOL_RATIO) > 0.05:
            errors.append(f"val pool ratio for {c} is {st['achieved_ratio']}, target {TEST_POOL_RATIO}")

    report = {
        "seed": SEED,
        "val_fraction": VAL_FRACTION,
        "train_entities": train.height,
        "val_entities": val.height,
        "achieved_val_fraction": round(val.height / split.height, 5),
        "stratification": {
            c: {
                "train": int(((train["country"] == c)).sum()),
                "val": int(((val["country"] == c)).sum()),
                "val_pct": round(100 * int((val["country"] == c).sum())
                                 / int((split["country"] == c).sum()), 3),
            } for c in sorted(split["country"].unique().to_list())
        },
        "val_pool": pool_stats,
        "records_claimed_across_folds": multi,
        "errors": errors,
        "ok": not errors,
    }
    (REPORTS / "phase4_split.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print("PHASE 4/5 -- split + validation pool")
    r = build_split()
    print(f"  train entities {r['train_entities']:>9,d}")
    print(f"  val entities   {r['val_entities']:>9,d}   ({100*r['achieved_val_fraction']:.3f}%)")
    for c, st in r["stratification"].items():
        print(f"    {c:6s} train={st['train']:>9,d} val={st['val']:>7,d}  ({st['val_pct']}% val)")
    print(f"  records claimed across folds: {r['records_claimed_across_folds']} (must be 0)")
    print("  validation pool:")
    for c, st in r["val_pool"].items():
        print(f"    {c:6s} entities={st['val_entities']:>7,d} owned={st['owned_by_val']:>8,d} "
              f"+distractors={st['distractors_added']:>8,d} -> pool={st['pool_size']:>9,d} "
              f"ratio={st['achieved_ratio']}")
    if r["errors"]:
        print("\nERRORS:")
        for e in r["errors"]:
            print("  -", e)
        raise SystemExit(1)
    print("\nPHASE 4/5 OK")
