"""PHASE 1 -- raw TSV -> parquet, with integrity verification.

Raw files are read-only. Nothing here writes inside the dataset directory.

Two settings are load-bearing and easy to get wrong:
  * ``separator='\\t'`` -- addresses and ID lists contain commas.
  * empty strings must stay empty strings, NOT become null. 3.3% of S2/S3
    addresses are empty and 4.41% of true-matched records have one; treating
    them as missing and dropping them would discard real matches.
"""
from __future__ import annotations

import json
import time

import polars as pl

from .config import EXPECTED_ROWS, PARQUET, RAW, REPORTS, SOURCES, SPLITS

SCHEMA_SOURCE = {
    "entity_id": pl.String,
    "business_name": pl.String,
    "business_address": pl.String,
    "country": pl.String,
}
SCHEMA_GT = {"source1_entity_id": pl.String, "matched_entity_ids": pl.String}


def _read_tsv(path, schema):
    return pl.read_csv(
        path,
        separator="\t",
        has_header=True,
        schema_overrides=schema,
        quote_char=None,          # no quoting in these files; disabling avoids mis-parses
        empty_string_is_null=False,   # "" must stay "" -- see module docstring
        null_values=[],
        infer_schema_length=0,
    )


def ingest_all() -> dict:
    report = {"files": {}, "errors": []}
    t_all = time.time()

    for split in SPLITS:
        for src in SOURCES:
            name = f"{split}_{src}"
            t0 = time.time()
            df = _read_tsv(RAW / split / f"{name}.tsv", SCHEMA_SOURCE)
            prefix = {"source1": "S1-", "source2": "S2-", "source3": "S3-"}[src]

            n = df.height
            info = {
                "rows": n,
                "expected": EXPECTED_ROWS[name],
                "columns": df.columns,
                "null_cells": int(df.null_count().sum_horizontal()[0]),
                "empty_name": int((df["business_name"] == "").sum()),
                "empty_address": int((df["business_address"] == "").sum()),
                "empty_country": int((df["country"] == "").sum()),
                "bad_prefix": int((~df["entity_id"].str.starts_with(prefix)).sum()),
                "unique_ids": int(df["entity_id"].n_unique()),
                "countries": dict(
                    df["country"].value_counts().iter_rows()  # type: ignore[arg-type]
                ),
                "seconds": round(time.time() - t0, 1),
            }
            if n != EXPECTED_ROWS[name]:
                report["errors"].append(f"{name}: row count {n} != expected {EXPECTED_ROWS[name]}")
            if info["bad_prefix"]:
                report["errors"].append(f"{name}: {info['bad_prefix']} ids with wrong prefix")
            if info["unique_ids"] != n:
                report["errors"].append(f"{name}: entity_id not unique ({info['unique_ids']} of {n})")
            if info["null_cells"]:
                report["errors"].append(f"{name}: {info['null_cells']} null cells (expected 0)")

            df.write_parquet(PARQUET / f"{name}.parquet", compression="zstd")
            info["parquet_mb"] = round((PARQUET / f"{name}.parquet").stat().st_size / 2**20, 1)
            report["files"][name] = info
            print(f"  {name:22s} {n:>9,d} rows  {info['seconds']:>5.1f}s  "
                  f"-> {info['parquet_mb']:>6.1f} MB parquet")

    # ground truth
    t0 = time.time()
    gt = _read_tsv(RAW / "train" / "train_ground_truth.tsv", SCHEMA_GT)
    n = gt.height
    info = {
        "rows": n,
        "expected": EXPECTED_ROWS["train_ground_truth"],
        "columns": gt.columns,
        "empty_match_lists": int((gt["matched_entity_ids"] == "").sum()),
        "unique_s1": int(gt["source1_entity_id"].n_unique()),
        "seconds": round(time.time() - t0, 1),
    }
    if n != EXPECTED_ROWS["train_ground_truth"]:
        report["errors"].append(f"ground_truth: {n} rows != {EXPECTED_ROWS['train_ground_truth']}")
    if info["unique_s1"] != n:
        report["errors"].append("ground_truth: source1_entity_id not unique")
    gt.write_parquet(PARQUET / "train_ground_truth.parquet", compression="zstd")
    info["parquet_mb"] = round((PARQUET / "train_ground_truth.parquet").stat().st_size / 2**20, 1)
    report["files"]["train_ground_truth"] = info
    print(f"  {'train_ground_truth':22s} {n:>9,d} rows  {info['seconds']:>5.1f}s  "
          f"-> {info['parquet_mb']:>6.1f} MB parquet")

    report["total_seconds"] = round(time.time() - t_all, 1)
    report["ok"] = not report["errors"]
    (REPORTS / "phase1_ingest.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print("PHASE 1 -- ingestion")
    r = ingest_all()
    print(f"\ntotal {r['total_seconds']}s")
    if r["errors"]:
        print("ERRORS:")
        for e in r["errors"]:
            print("  -", e)
        raise SystemExit(1)
    print("PHASE 1 OK -- all row counts, prefixes, uniqueness and null checks passed")
