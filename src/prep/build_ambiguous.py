"""PHASE 2 -- extract the ambiguous band as the cross-encoder's training set.

The GBDT resolves 98%+ of pairs with near-certainty: the 0.00-0.01 band holds
10.4M validation pairs containing 1,195 positives (purity 0.0001), and the 0.99+
band holds 243k pairs at purity 0.999. Training a transformer on that mass would
spend all its capacity on pairs already solved.

What it cannot resolve is the band 0.1 < p < 0.9: 1.95% of train pairs, but
44.8% positive and containing 13.4% of all true matches. Those pairs ARE the
bottleneck, and restricting training to them is hard-negative mining by
construction -- every negative here is one the GBDT already found deceptive.

Emits the RAW text fields (not the normalised ones). The entire point of stage 2
is to give the model the information the 40 aggregate features destroyed:
word order, abbreviation, punctuation, script.
"""
from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
import polars as pl

from .config import DATA, REPORTS, REPR
from .train_matcher import encode, load_features

MODEL_PATH = DATA / "models" / "matcher.pkl"
OUT = DATA / "ambiguous"
RAW_COLS = ["entity_id", "business_name", "business_address", "country", "name_script"]


def raw_tables(split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    q = pl.read_parquet(REPR / f"{split}_source1.parquet", columns=RAW_COLS)
    pool = pl.concat([
        pl.read_parquet(REPR / f"{split}_source2.parquet", columns=RAW_COLS),
        pl.read_parquet(REPR / f"{split}_source3.parquet", columns=RAW_COLS),
    ])
    return q, pool


def build(split_dir: str, out_name: str, lo: float, hi: float,
          repr_split: str = "train") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("rb") as fh:
        b = pickle.load(fh)

    feat = load_features(DATA / "features" / "pairs" / split_dir)
    X, _ = encode(feat, b["features"], b["cat_cols"], cat_levels=b["cat_levels"])
    p = b["model"].predict_proba(X)[:, 1].astype(np.float32)
    del X

    keep = (p > lo) & (p < hi)
    sub = feat.select(["source1_entity_id", "candidate_entity_id"] +
                      (["label"] if "label" in feat.columns else [])
                      ).filter(pl.Series(keep)).with_columns(
        pl.Series("p_gbdt", p[keep]))
    del feat

    q, pool = raw_tables(repr_split)
    out = (sub
           .join(q.rename({c: f"q_{c}" for c in RAW_COLS if c != "entity_id"} |
                          {"entity_id": "source1_entity_id"}),
                 on="source1_entity_id", how="left")
           .join(pool.rename({c: f"c_{c}" for c in RAW_COLS if c != "entity_id"} |
                             {"entity_id": "candidate_entity_id"}),
                 on="candidate_entity_id", how="left"))

    path = OUT / f"{out_name}.parquet"
    out.write_parquet(path, compression="zstd")

    rep = {
        "split_dir": split_dir, "band": [lo, hi],
        "total_pairs": int(len(p)), "kept": int(keep.sum()),
        "kept_pct": round(100 * float(keep.mean()), 3),
        "positives": int(out["label"].sum()) if "label" in out.columns else None,
        "positive_rate": round(float(out["label"].mean()), 4) if "label" in out.columns else None,
        "path": str(path), "mb": round(path.stat().st_size / 2**20, 1),
    }
    print(f"{out_name}: {rep['kept']:,} pairs ({rep['kept_pct']}%), "
          f"positives={rep['positives']}, rate={rep['positive_rate']} -> {rep['mb']}MB")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=float, default=0.1)
    ap.add_argument("--hi", type=float, default=0.9)
    args = ap.parse_args()
    rep = {"train": build("train", "train_ambiguous", args.lo, args.hi),
           "val": build("val", "val_ambiguous", args.lo, args.hi)}
    (REPORTS / "ambiguous_band.json").write_text(json.dumps(rep, indent=2, default=str))
    print("AMBIGUOUS_DONE")
