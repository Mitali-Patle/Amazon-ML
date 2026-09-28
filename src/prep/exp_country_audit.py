"""INVESTIGATION -- is `country` safe as a hard retrieval boundary?

Answers, from the data rather than from the playbook:
  1. exact byte-level country values in all 6 files (casing/whitespace/variants)
  2. empty / missing country
  3. countries in test but not train, and vice versa
  4. for every country: does it have records in S1 AND in S2 AND in S3?
     (an S1 country with an empty pool is the silent-zero-candidate failure)
  5. how much recall a hard country block would cost if metadata were noisy
"""
from __future__ import annotations

import json
import unicodedata
from collections import Counter

import polars as pl

from .config import PARQUET, REPORTS, SOURCES, SPLITS


def describe(v: str) -> dict:
    return {
        "repr": repr(v),
        "len": len(v),
        "has_leading_ws": v != v.lstrip(),
        "has_trailing_ws": v != v.rstrip(),
        "casefold": v.casefold(),
        "stripped_casefold": v.strip().casefold(),
        "codepoints": [hex(ord(c)) for c in v][:12],
        "nfkc_differs": unicodedata.normalize("NFKC", v) != v,
    }


def main() -> dict:
    out: dict = {"per_file": {}, "variant_analysis": {}, "coverage": {}, "risks": []}

    per_split: dict[str, dict[str, Counter]] = {}
    for split in SPLITS:
        per_split[split] = {}
        for src in SOURCES:
            name = f"{split}_{src}"
            df = pl.read_parquet(PARQUET / f"{name}.parquet", columns=["country"])
            c = Counter(df["country"].to_list())
            per_split[split][src] = c
            out["per_file"][name] = {
                "n_distinct": len(c),
                "values": {repr(k): v for k, v in c.most_common()},
                "empty_or_whitespace": sum(v for k, v in c.items() if not k.strip()),
                "details": {repr(k): describe(k) for k in c},
            }
            print(f"  {name:16s} {len(c)} distinct: " +
                  "  ".join(f"{k!r}:{v:,}" for k, v in c.most_common()))

    # ---- variant analysis: do any two distinct values collapse under normalisation? ----
    all_vals = set()
    for split in SPLITS:
        for src in SOURCES:
            all_vals |= set(per_split[split][src])
    groups: dict[str, list[str]] = {}
    for v in all_vals:
        groups.setdefault(v.strip().casefold(), []).append(v)
    collisions = {k: v for k, v in groups.items() if len(v) > 1}
    out["variant_analysis"] = {
        "distinct_raw_values": sorted(repr(v) for v in all_vals),
        "collapse_under_strip_casefold": {k: [repr(x) for x in v] for k, v in collisions.items()},
        "any_variants": bool(collisions),
        "any_whitespace_issues": any(v != v.strip() for v in all_vals),
        "any_empty": any(not v.strip() for v in all_vals),
        "any_nfkc_differs": any(unicodedata.normalize("NFKC", v) != v for v in all_vals),
    }
    print(f"\n  distinct country values across ALL 6 files: {sorted(repr(v) for v in all_vals)}")
    print(f"  variants collapsing under strip+casefold: {collisions or 'NONE'}")
    print(f"  whitespace issues: {out['variant_analysis']['any_whitespace_issues']}  "
          f"empty: {out['variant_analysis']['any_empty']}  "
          f"NFKC differs: {out['variant_analysis']['any_nfkc_differs']}")

    # ---- coverage: every S1 country must have a non-empty S2 and S3 pool ----
    for split in SPLITS:
        s1c, s2c, s3c = (per_split[split][s] for s in SOURCES)
        cov = {}
        for c in sorted(set(s1c) | set(s2c) | set(s3c)):
            cov[c] = {"S1": s1c.get(c, 0), "S2": s2c.get(c, 0), "S3": s3c.get(c, 0)}
            if s1c.get(c, 0) and not s2c.get(c, 0) and not s3c.get(c, 0):
                out["risks"].append(
                    f"{split}: country {c!r} has {s1c[c]:,} S1 entities but EMPTY S2+S3 pool "
                    f"-> hard block yields zero candidates for all of them")
            elif s1c.get(c, 0) and (not s2c.get(c, 0) or not s3c.get(c, 0)):
                out["risks"].append(
                    f"{split}: country {c!r} missing one source (S2={s2c.get(c,0)}, S3={s3c.get(c,0)})")
        out["coverage"][split] = cov
        print(f"\n  {split} coverage (S1 / S2 / S3 record counts per country):")
        for c, v in cov.items():
            ratio = (v["S2"] + v["S3"]) / v["S1"] if v["S1"] else float("inf")
            print(f"    {c!r:12s} S1={v['S1']:>9,d}  S2={v['S2']:>9,d}  S3={v['S3']:>9,d}"
                  f"   pool/S1 ratio={ratio:.2f}")

    train_countries = set(per_split["train"]["source1"])
    test_countries = set(per_split["test"]["source1"])
    out["unseen_in_test"] = sorted(test_countries - train_countries)
    out["absent_from_test"] = sorted(train_countries - test_countries)
    print(f"\n  countries in TEST but not TRAIN: {out['unseen_in_test']}")
    print(f"  countries in TRAIN but not TEST: {out['absent_from_test'] or 'none'}")

    # ---- what would a hard block cost if labels were noisy? (train evidence) ----
    gt = pl.read_parquet(PARQUET / "train_ground_truth.parquet")
    pairs = (gt.filter(pl.col("matched_entity_ids") != "")
             .with_columns(pl.col("matched_entity_ids").str.split(","))
             .explode("matched_entity_ids"))
    ctry = pl.concat([
        pl.read_parquet(PARQUET / f"train_{s}.parquet", columns=["entity_id", "country"])
        for s in SOURCES])
    j = (pairs.join(ctry.rename({"entity_id": "source1_entity_id", "country": "c1"}),
                    on="source1_entity_id", how="left")
         .join(ctry.rename({"entity_id": "matched_entity_ids", "country": "c2"}),
               on="matched_entity_ids", how="left"))
    n = j.height
    cross = int((j["c1"] != j["c2"]).sum())
    out["hard_block_cost_train"] = {
        "pairs": n, "cross_country_pairs": cross,
        "recall_lost_pct": round(100 * cross / n, 6),
        "upper_bound_95pct_if_zero": round(100 * 3 / n, 6),
    }
    print(f"\n  hard-block recall cost on TRAIN: {cross}/{n:,} = "
          f"{100*cross/n:.6f}%  (rule-of-three 95% upper bound if 0: "
          f"{100*3/n:.6f}%)")

    (REPORTS / "exp_country_audit.json").write_text(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    print("COUNTRY AUDIT\n")
    r = main()
    print("\n  RISKS FOUND:" if r["risks"] else "\n  no structural coverage risks found")
    for x in r["risks"]:
        print("   -", x)
