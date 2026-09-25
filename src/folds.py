"""Fixed anchor-level train/val split (ADR-001, updated).

Unit: source1 entity_id. EDA showed the GT match graph is disjoint stars (each S2/S3 id
belongs to exactly one anchor), so an anchor split cannot put the same target record on
both sides. Do NOT group by normalized name (chains -> giant component).

Stratified by (has_match, country), VAL_FRAC=5% (~110k anchors of ~2.2M: ~6k singletons,
enough for a stable F0.5 estimate; a bigger val only inflates Tier-2 full-pool scoring
cost). Seed 42, deterministic. Output: data/interim/split.parquet
(source1_entity_id, country, has_match, is_val). data/test is never touched.

Run: python src/folds.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GT = ROOT / "data/train/train_ground_truth.tsv"
S1 = ROOT / "data/train/train_source1.tsv"
OUT = ROOT / "data/interim/split.parquet"
SEED = 42
VAL_FRAC = 0.05


def load_anchors() -> pd.DataFrame:
    gt = pd.read_csv(GT, sep="\t", dtype=str, keep_default_na=False)
    s1 = pd.read_csv(S1, sep="\t", dtype=str, keep_default_na=False, usecols=["entity_id", "country"])
    df = gt.merge(s1.rename(columns={"entity_id": "source1_entity_id"}),
                  on="source1_entity_id", how="left", validate="1:1")
    assert df["country"].notna().all() and (df["country"] != "").all(), "anchor without country"
    df["has_match"] = df["matched_entity_ids"].str.strip() != ""
    return df


def make_split(df: pd.DataFrame, val_frac: float = VAL_FRAC, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.sort_values("source1_entity_id").reset_index(drop=True)  # order-independent
    is_val = np.zeros(len(df), dtype=bool)
    for _, idx in df.groupby(["has_match", "country"]).indices.items():
        k = int(round(len(idx) * val_frac))
        is_val[rng.choice(idx, size=k, replace=False)] = True
    df["is_val"] = is_val
    return df[["source1_entity_id", "country", "has_match", "is_val"]]


def check_split(split: pd.DataFrame, gt: pd.DataFrame, tol: float = 0.002) -> None:
    key = ["has_match", "country"]
    tr, va = split[~split.is_val], split[split.is_val]
    p_tr = tr.groupby(key).size() / len(tr)
    p_va = va.groupby(key).size() / len(va)
    diff = (p_tr - p_va.reindex(p_tr.index).fillna(0)).abs().max()
    assert diff < tol, f"stratum proportions differ by {diff:.4f}"
    assert split.source1_entity_id.is_unique
    # no S2/S3 id shared across sides
    m = gt.merge(split[["source1_entity_id", "is_val"]], on="source1_entity_id")
    m = m[m.matched_entity_ids.str.strip() != ""]
    ids = m.assign(mid=m.matched_entity_ids.str.split(",")).explode("mid")[["mid", "is_val"]]
    ids["mid"] = ids["mid"].str.strip()
    both = ids.groupby("mid")["is_val"].nunique()
    assert (both == 1).all(), f"{(both > 1).sum()} match ids appear on both sides"
    print(f"OK: max stratum diff {diff:.5f}; {len(ids):,} match ids, none shared across sides")


def main():
    df = load_anchors()
    split = make_split(df)
    check_split(split, df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    split.to_parquet(OUT, index=False)
    print(f"train anchors {int((~split.is_val).sum()):,}  val anchors {int(split.is_val.sum()):,}")
    print(split.groupby(["is_val", "has_match", "country"]).size().unstack(0))


def load_split() -> pd.DataFrame:
    return pd.read_parquet(OUT)


if __name__ == "__main__":
    main()
