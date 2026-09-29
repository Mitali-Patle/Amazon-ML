"""PHASE 3 -- deterministic per-record representations.

Fits nothing, so it is leakage-free and runs on train and test alike (report D18).
Raw fields are always carried through unchanged (D3).

Layers produced per record:
    raw        business_name, business_address, country
    norm       name_norm, addr_norm
    script     name_script, addr_script
    derived    name_tokens[], addr_tokens[]   (transliterated forms UNIONED in, D6)
               legal_suffix[], name_core      (name_core is FEATURE ONLY, never a key)
               addr_pin, addr_numbers[]

Deliberately NOT produced: a state/street-expanded token stream (D5, measured
-1.04pp recall) and char-n-gram retrieval keys (D7, measured -62pp).
"""
from __future__ import annotations

import json
import time
from multiprocessing import Pool

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from . import normalize as N
from .config import INGEST_CHUNK, PARQUET, REPORTS, REPR, SOURCES, SPLITS

ARROW_SCHEMA = pa.schema([
    ("entity_id", pa.string()),
    ("country", pa.string()),
    ("business_name", pa.string()),
    ("business_address", pa.string()),
    ("name_norm", pa.string()),
    ("addr_norm", pa.string()),
    ("name_script", pa.string()),
    ("addr_script", pa.string()),
    ("name_tokens", pa.list_(pa.string())),
    ("addr_tokens", pa.list_(pa.string())),
    ("legal_suffix", pa.list_(pa.string())),
    ("name_core", pa.string()),
    ("addr_pin", pa.string()),
    ("addr_numbers", pa.list_(pa.string())),
])


def _process_rows(rows):
    """Worker: list of (entity_id, name, addr, country) -> column-wise lists."""
    out = {k: [] for k in ARROW_SCHEMA.names}
    for eid, name, addr, country in rows:
        n_norm = N.normalize(name)
        a_norm = N.normalize(addr)
        n_tok = N.tokens_with_translit(n_norm)
        a_tok = N.tokens_with_translit(a_norm)
        # suffix extraction runs on the ORIGINAL tokens but resolves Indic forms
        # internally, so a Devanagari "लिमिटेड" still yields the canonical "ltd"
        plain = N.tokens_of(n_norm)
        out["entity_id"].append(eid)
        out["country"].append(country)
        out["business_name"].append(name)
        out["business_address"].append(addr)
        out["name_norm"].append(n_norm)
        out["addr_norm"].append(a_norm)
        out["name_script"].append(N.script_of(name))
        out["addr_script"].append(N.script_of(addr))
        out["name_tokens"].append(n_tok)
        out["addr_tokens"].append(a_tok)
        out["legal_suffix"].append(N.extract_suffixes(plain))
        out["name_core"].append(N.name_core(plain))
        out["addr_pin"].append(N.addr_pin(addr))
        out["addr_numbers"].append(N.addr_numbers(addr))
    return out


def build_one(name: str, pool: Pool, workers: int) -> dict:
    src = PARQUET / f"{name}.parquet"
    dst = REPR / f"{name}.parquet"
    df = pl.read_parquet(src)
    n = df.height
    t0 = time.time()

    writer = pq.ParquetWriter(dst, ARROW_SCHEMA, compression="zstd")
    stats = {"rows": 0, "name_script": {}, "addr_script": {},
             "translit_names": 0, "empty_addr_tokens": 0, "empty_name_tokens": 0}
    try:
        for start in range(0, n, INGEST_CHUNK):
            chunk = df.slice(start, INGEST_CHUNK)
            rows = list(zip(chunk["entity_id"], chunk["business_name"],
                            chunk["business_address"], chunk["country"]))
            # split the chunk across workers
            step = max(1, len(rows) // workers + 1)
            parts = [rows[i:i + step] for i in range(0, len(rows), step)]
            merged = {k: [] for k in ARROW_SCHEMA.names}
            for res in pool.map(_process_rows, parts):
                for k, v in res.items():
                    merged[k].extend(v)
            writer.write_table(pa.Table.from_pydict(merged, schema=ARROW_SCHEMA))

            stats["rows"] += len(rows)
            for s in merged["name_script"]:
                stats["name_script"][s] = stats["name_script"].get(s, 0) + 1
            for s in merged["addr_script"]:
                stats["addr_script"][s] = stats["addr_script"].get(s, 0) + 1
            for raw_n, toks in zip(merged["name_norm"], merged["name_tokens"]):
                if N.has_indic(raw_n):
                    stats["translit_names"] += 1
            stats["empty_addr_tokens"] += sum(1 for t in merged["addr_tokens"] if not t)
            stats["empty_name_tokens"] += sum(1 for t in merged["name_tokens"] if not t)
    finally:
        writer.close()

    stats["seconds"] = round(time.time() - t0, 1)
    stats["mb"] = round(dst.stat().st_size / 2**20, 1)
    stats["row_check"] = (stats["rows"] == n)
    print(f"  {name:22s} {n:>9,d} rows  {stats['seconds']:>6.1f}s  -> {stats['mb']:>6.1f} MB"
          f"   translit={stats['translit_names']:>8,d}")
    return stats


def main(workers: int = 18) -> dict:
    report = {"files": {}, "errors": []}
    with Pool(workers) as pool:
        for split in SPLITS:
            for src in SOURCES:
                name = f"{split}_{src}"
                st = build_one(name, pool, workers)
                report["files"][name] = st
                if not st["row_check"]:
                    report["errors"].append(f"{name}: row count mismatch after representation")
                if st["empty_name_tokens"]:
                    report["errors"].append(
                        f"{name}: {st['empty_name_tokens']} records lost all name tokens")
    report["ok"] = not report["errors"]
    (REPORTS / "phase3_represent.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print("PHASE 3 -- representations")
    r = main()
    if r["errors"]:
        print("\nERRORS:")
        for e in r["errors"]:
            print("  -", e)
        raise SystemExit(1)
    print("\nPHASE 3 OK -- rows preserved, no record lost its name tokens")
