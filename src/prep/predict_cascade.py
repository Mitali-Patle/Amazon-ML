"""Cascade inference on the official test set: GBDT -> cross-encoder -> submission.

THREE PASSES, deliberately. A single pass would need the feature-path
representation tables (~11.8GB peak, measured) AND the raw-text tables the
cross-encoder needs, simultaneously. That exceeds 15GB and OOMs. Splitting the
work keeps every stage bounded:

  A. score all 86.6M test pairs with the GBDT, stream (eid, cid, p) to parquet.
     Holds the representation tables; never holds raw text.
  B. load ONLY the lean raw-text tables and the ~1M band pairs, score them with
     the cross-encoder on GPU. Never holds the representation tables.
  C. merge, decide per entity, deconflict globally, write and validate.

Pass B uses RAW business_name / business_address / country because that is what
the cross-encoder was trained on (build_ambiguous.py). Feeding it normalised
text would be a train/inference mismatch and silently degrade it.
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from .config import (BAND_HI, BAND_HI_FRANCE, BAND_LO, BAND_LO_FRANCE, CANDIDATES,
                     DATA, PARQUET, REPORTS, REPR, ROOT)
from .decide import choose_k, deconflict
from .pair_features import add_pair_features, join_query_pool, load_pool_table, load_query_table
from .submission import (CANDIDATE_HEADER, MATCHING_HEADER, OUTPUT, preflight,
                         required_entities, run_official_validator, write_tsv)
from .train_matcher import encode

GBDT_PATH = DATA / "models" / "matcher.pkl"
CE_DIR = DATA / "models" / "crossencoder"
SCORES = DATA / "scores" / "test"
USE_JOINT_CAL = True   # set False to disable joint recalibration
ENTITY_SUBCHUNK = 20_000   # ~6M pairs per sub-chunk at K=150/source
RAW_COLS = ["entity_id", "business_name", "business_address", "country"]


# ---------------------------------------------------------------- pass A
def pass_a(shard_dir: Path) -> dict:
    SCORES.mkdir(parents=True, exist_ok=True)
    with GBDT_PATH.open("rb") as fh:
        b = pickle.load(fh)
    q_tab, pool_tab = load_query_table("test"), load_pool_table("test")
    print(f"[A] representations loaded: q={q_tab.height:,} pool={pool_tab.height:,}")

    shards = sorted(shard_dir.glob("*.parquet"))
    n = 0
    t0 = time.time()
    for i, sh in enumerate(shards, 1):
        out = SCORES / f"scores_{i:05d}.parquet"
        if out.exists():
            n += pl.read_parquet(out, columns=["p"]).height
            continue
        wide = pl.read_parquet(sh, columns=["source1_entity_id", "candidate_entity_ids"])
        # SUB-CHUNK within the shard. A shard is 100k entities; at K=150/source
        # that is ~30M pairs, and materialising the joined feature frame for all
        # of them at once exhausts RAM (the same failure the first predict_submit
        # hit). Entities are independent here, so slicing the shard is free.
        parts = []
        for start in range(0, wide.height, ENTITY_SUBCHUNK):
            sl = wide.slice(start, ENTITY_SUBCHUNK)
            ex = (sl.with_columns(pl.col("candidate_entity_ids").str.split(","))
                  .explode("candidate_entity_ids")
                  .rename({"candidate_entity_ids": "candidate_entity_id"})
                  .filter(pl.col("candidate_entity_id").is_not_null()
                          & (pl.col("candidate_entity_id") != "")))
            del sl
            if ex.height == 0:
                continue
            feat = add_pair_features(join_query_pool(ex, q_tab, pool_tab))
            del ex
            X, _ = encode(feat, b["features"], b["cat_cols"], cat_levels=b["cat_levels"])
            p = b["model"].predict_proba(X)[:, 1].astype(np.float32)
            del X
            parts.append(feat.select(["source1_entity_id", "candidate_entity_id"])
                         .with_columns(pl.Series("p", p)))
            n += feat.height
            del feat, p
            gc.collect()
        del wide
        if not parts:
            continue
        pl.concat(parts).write_parquet(out, compression="zstd")
        del parts
        gc.collect()
        print(f"  [A] shard {i}/{len(shards)}: {n:,} scored ({time.time()-t0:.0f}s)", flush=True)
    del q_tab, pool_tab
    gc.collect()
    return {"pairs": n, "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- pass B
def pass_b(lo: float, hi: float, batch: int = 1024, max_len: int = 128) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from .train_crossencoder import Collate, PairDS, predict

    t0 = time.time()
    # France entities use the wider band (see config): the GBDT extrapolates on
    # France and its confidence there is less reliable, so more pairs are handed
    # to the cross-encoder.
    fr_ids = set(pl.read_parquet(PARQUET / "test_source1.parquet",
                                 columns=["entity_id", "country"])
                 .filter(pl.col("country") == "France")["entity_id"].to_list())
    parts = []
    for f in sorted(SCORES.glob("scores_*.parquet")):
        d = pl.read_parquet(f).with_columns(
            pl.col("source1_entity_id").is_in(fr_ids).alias("_fr"))
        parts.append(d.filter(
            ((~pl.col("_fr")) & (pl.col("p") > lo) & (pl.col("p") < hi))
            | (pl.col("_fr") & (pl.col("p") > BAND_LO_FRANCE) & (pl.col("p") < BAND_HI_FRANCE))
        ).drop("_fr"))
    band = pl.concat(parts)
    del parts
    n_fr = int(band["source1_entity_id"].is_in(fr_ids).sum())
    print(f"[B] band pairs: {band.height:,} total "
          f"(non-France {lo}-{hi}: {band.height - n_fr:,}; "
          f"France {BAND_LO_FRANCE}-{BAND_HI_FRANCE}: {n_fr:,})")
    if band.height == 0:
        return {"band_pairs": 0}

    q = pl.read_parquet(REPR / "test_source1.parquet", columns=RAW_COLS)
    pool = pl.concat([pl.read_parquet(REPR / f"test_{s}.parquet", columns=RAW_COLS)
                      for s in ("source2", "source3")])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(CE_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(CE_DIR).to(device)
    with (CE_DIR / "calibrator.pkl").open("rb") as fh:
        iso = pickle.load(fh)["iso"]

    # CHUNK the band. France's wider band pushes this to ~21M pairs, and joining
    # raw text onto all of them at once exhausts RAM (this is what killed the
    # first attempt). Each chunk is joined, scored and released independently.
    out_dir = DATA / "scores" / "band_ce_parts"
    out_dir.mkdir(parents=True, exist_ok=True)
    CH = 2_000_000
    n_done = 0
    for ci, start in enumerate(range(0, band.height, CH)):
        part_path = out_dir / f"part_{ci:04d}.parquet"
        if part_path.exists():
            n_done += pl.read_parquet(part_path, columns=["p_ce"]).height
            continue
        sl = band.slice(start, CH)
        joined = (sl
                  .join(q.rename({"entity_id": "source1_entity_id",
                                  "business_name": "q_business_name",
                                  "business_address": "q_business_address",
                                  "country": "q_country"}),
                        on="source1_entity_id", how="left")
                  .join(pool.rename({"entity_id": "candidate_entity_id",
                                     "business_name": "c_business_name",
                                     "business_address": "c_business_address",
                                     "country": "c_country"}),
                        on="candidate_entity_id", how="left"))
        del sl
        dl = DataLoader(PairDS(joined, tok, max_len, with_label=False),
                        batch_size=batch, collate_fn=Collate(tok, max_len),
                        num_workers=6, pin_memory=True)
        cal = iso.predict(predict(model, dl, device)).astype(np.float32)
        joined.select(["source1_entity_id", "candidate_entity_id"]).with_columns(
            pl.Series("p_ce", cal)).write_parquet(part_path, compression="zstd")
        n_done += len(cal)
        del joined, dl, cal
        gc.collect()
        print(f"  [B] chunk {ci+1}/{(band.height+CH-1)//CH}: {n_done:,} scored "
              f"({time.time()-t0:.0f}s)", flush=True)

    pl.concat([pl.read_parquet(f) for f in sorted(out_dir.glob("part_*.parquet"))]
              ).write_parquet(DATA / "scores" / "test_band_ce.parquet", compression="zstd")
    print(f"[B] scored {n_done:,} band pairs in {time.time()-t0:.0f}s on {device}")
    return {"band_pairs": int(n_done), "seconds": round(time.time() - t0, 1),
            "device": str(device)}


# ---------------------------------------------------------------- pass C
def pass_c(shard_dir: Path, blend_w: float) -> dict:
    t0 = time.time()
    ce_path = DATA / "scores" / "test_band_ce.parquet"
    ce = pl.read_parquet(ce_path) if ce_path.exists() else None
    ce_map: dict[tuple[str, str], float] = {}
    if ce is not None:
        for a, c, p in ce.iter_rows():
            ce_map[(a, c)] = p
        print(f"[C] cross-encoder scores for {len(ce_map):,} pairs")

    jc_path = DATA / "models" / "joint_calibrator.pkl"
    jc = None
    if jc_path.exists() and USE_JOINT_CAL:
        with jc_path.open("rb") as fh:
            jc = pickle.load(fh)["iso"]
        print("[C] applying joint calibrator to the blended score")

    selected: dict[str, list[tuple[str, float]]] = {}
    k_hist: Counter[int] = Counter()
    replaced = 0
    for f in sorted(SCORES.glob("scores_*.parquet")):
        df = pl.read_parquet(f)
        by_ent: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for e, c, p in zip(df["source1_entity_id"].to_list(),
                           df["candidate_entity_id"].to_list(), df["p"].to_list()):
            q = ce_map.get((e, c))
            if q is not None:
                p = blend_w * q + (1.0 - blend_w) * p
                replaced += 1
            by_ent[e].append((c, p))
        del df
        for e, items in by_ent.items():
            probs = np.fromiter((x[1] for x in items), dtype=np.float64, count=len(items))
            if jc is not None:
                probs = jc.predict(probs)
            order = np.argsort(-probs)
            k, _ = choose_k(probs)
            k_hist[k] += 1
            selected[e] = [items[j] for j in order[:k]] if k else []
        del by_ent
        gc.collect()
    print(f"[C] {replaced:,} pair scores replaced by the cross-encoder")

    entities = required_entities("test")
    for e in entities:
        selected.setdefault(e, [])
    matches = deconflict(selected)
    removed = sum(len(v) for v in selected.values()) - sum(len(v) for v in matches.values())

    pre = preflight(entities, matches, candidates=None)
    if not pre["ok"]:
        print("PREFLIGHT_FAILED", pre["problems"])
        return {"ok": False, "preflight": pre}

    m_info = write_tsv(OUTPUT / "matching_results.tsv", MATCHING_HEADER, entities, matches)
    # candidate_pairs.tsv is streamed from the shards, never held in memory
    have: dict[str, str] = {}
    for sh in sorted(shard_dir.glob("*.parquet")):
        d = pl.read_parquet(sh, columns=["source1_entity_id", "candidate_entity_ids"])
        for eid, ids in zip(d["source1_entity_id"].to_list(), d["candidate_entity_ids"].to_list()):
            have[eid] = ids or ""
        del d
    c_info = write_tsv(OUTPUT / "candidate_pairs.tsv", CANDIDATE_HEADER, entities,
                       {e: (have.get(e, "").split(",") if have.get(e) else []) for e in entities})

    test_dir = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "test"
    val = run_official_validator(OUTPUT / "matching_results.tsv",
                                 OUTPUT / "candidate_pairs.tsv", test_dir)
    rep = {
        "blend_w": blend_w, "ce_replaced": replaced,
        "deconflict_removed": removed,
        "empty_predictions": sum(1 for v in matches.values() if not v),
        "mean_predictions": round(sum(len(v) for v in matches.values()) / len(entities), 3),
        "k_histogram": dict(sorted(k_hist.items())),
        "matching_file": m_info, "candidate_file": c_info,
        "preflight": pre, "validator_passed": val.get("passed"),
        "seconds": round(time.time() - t0, 1),
    }
    print(f"[C] mean preds={rep['mean_predictions']}  empty={rep['empty_predictions']:,}  "
          f"deconflict removed={removed:,}")
    print(f"VALIDATOR passed={val.get('passed')}")
    print(val.get("stdout", "")[-600:])
    return rep


def main(blend_w: float = 1.0, lo: float = BAND_LO, hi: float = BAND_HI,
         only: str | None = None) -> dict:
    shard_dir = CANDIDATES / "test" / "merged"
    rep: dict = {"blend_w": blend_w, "band": [lo, hi]}
    if only in (None, "a"):
        rep["pass_a"] = pass_a(shard_dir)
    if only in (None, "b"):
        rep["pass_b"] = pass_b(lo, hi)
    if only in (None, "c"):
        rep["pass_c"] = pass_c(shard_dir, blend_w)
        if rep["pass_c"].get("validator_passed"):
            print("CASCADE_SUBMISSION_OK")
    (REPORTS / "predict_cascade.json").write_text(json.dumps(rep, indent=2, default=str))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["a", "b", "c"])
    ap.add_argument("--w", type=float, default=1.0)
    a = ap.parse_args()
    main(blend_w=a.w, only=a.only)
