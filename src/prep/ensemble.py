"""Ensemble our cascade with a teammate's independent model, at probability level.

Two systems that reach the same score by different routes (theirs: e5 embeddings
+ LightGBM -> cross-encoder; ours: 40 engineered features + GBDT -> cross-encoder)
have largely uncorrelated errors, which is the setup where averaging pays.

THE HARD PART IS THE MISSING PAIRS, not the averaging.

Their file stores only pairs with prob >= 0.10; anything below was dropped. So a
pair present in one file and absent from the other is ambiguous -- it may have
been scored low, or never retrieved at all. Those two cases deserve different
treatment and cannot be distinguished, so the policy is explicit and switchable:

  intersect  - keep only pairs BOTH models retrieved. Maximum precision, loses
               every unique find. Under F_0.5 (precision weighted 2x) this is
               not as conservative as it sounds.
  union_mean - keep every pair either model has; average where both, take the
               single available probability otherwise. Gains their extra recall.
  union_pess - as union_mean, but a pair only one model has is averaged against
               an assumed 0.05 for the silent model. Penalises unique finds.

DEFAULT is union_pess: it captures the recall of pairs the other model missed
while still discounting them for lack of corroboration, which matches how
F_0.5 punishes unsupported predictions.

NOTE ON VALIDATION: their probabilities cover TEST, where we have no labels, so
the blend weight cannot be tuned. A plain 50/50 average of two calibrated models
is the robust choice when tuning is impossible -- do not invent a weight. The
sanity check is distributional: a healthy ensemble should land near the training
statistics (mean ~3.46 matches/entity, ~5.6% singletons).
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from .config import DATA, REPORTS, ROOT
from .decide import choose_k, deconflict
from .submission import (CANDIDATE_HEADER, MATCHING_HEADER, OUTPUT, preflight,
                         required_entities, run_official_validator, write_tsv)

SCORES = DATA / "scores" / "test"
ASSUMED_LOW = 0.05          # stand-in for "the other model did not store this pair"
EXTERNAL = DATA / "external"


def load_ours() -> pl.DataFrame:
    """Our per-pair test probabilities, with the cascade and joint calibration applied."""
    shards = sorted(SCORES.glob("scores_*.parquet"))
    if not shards:
        raise FileNotFoundError("no test score shards -- pass A has not run")
    ours = pl.concat([pl.read_parquet(f) for f in shards])

    ce_path = DATA / "scores" / "test_band_ce.parquet"
    if ce_path.exists():
        ce = pl.read_parquet(ce_path)
        ours = ours.join(ce, on=["source1_entity_id", "candidate_entity_id"], how="left")
        ours = ours.with_columns(
            pl.when(pl.col("p_ce").is_not_null()).then(pl.col("p_ce"))
            .otherwise(pl.col("p")).alias("p")).drop("p_ce")
        print(f"  cross-encoder scores merged for {ce.height:,} pairs")

    jc = DATA / "models" / "joint_calibrator.pkl"
    if jc.exists():
        with jc.open("rb") as fh:
            iso = pickle.load(fh)["iso"]
        ours = ours.with_columns(
            pl.Series("p", iso.predict(ours["p"].to_numpy()).astype(np.float32)))
        print("  joint calibrator applied")
    return ours.rename({"p": "p_ours"})


def load_theirs(path: Path) -> pl.DataFrame:
    """Read the teammate's per-pair probabilities.

    Their header is quoted (`"source1_entity_id"`) while the data rows are not,
    so column names are de-quoted explicitly rather than relying on a quote_char
    setting that would have to be right for both.
    """
    t = pl.read_csv(path, separator="\t", quote_char=None)
    t = t.rename({c: c.strip().strip('"').strip("'") for c in t.columns})
    cols = {c.lower(): c for c in t.columns}
    t = t.rename({cols["source1_entity_id"]: "source1_entity_id",
                  cols["candidate_entity_id"]: "candidate_entity_id",
                  cols["prob"]: "p_theirs"})
    return t.select(["source1_entity_id", "candidate_entity_id", "p_theirs"])


def blend(ours: pl.DataFrame, theirs: pl.DataFrame, policy: str, w: float) -> pl.DataFrame:
    j = ours.join(theirs, on=["source1_entity_id", "candidate_entity_id"], how="full",
                  coalesce=True)
    both = j["p_ours"].is_not_null() & j["p_theirs"].is_not_null()
    only_ours = j["p_ours"].is_not_null() & j["p_theirs"].is_null()
    only_theirs = j["p_ours"].is_null() & j["p_theirs"].is_not_null()
    print(f"  pairs: both={int(both.sum()):,}  ours_only={int(only_ours.sum()):,}  "
          f"theirs_only={int(only_theirs.sum()):,}")

    po = j["p_ours"].fill_null(0.0).to_numpy()
    pt = j["p_theirs"].fill_null(0.0).to_numpy()
    bo, oo, ot = both.to_numpy(), only_ours.to_numpy(), only_theirs.to_numpy()

    p = np.zeros(len(po), dtype=np.float64)
    p[bo] = w * po[bo] + (1 - w) * pt[bo]
    if policy == "intersect":
        p[oo] = 0.0
        p[ot] = 0.0
    elif policy == "union_mean":
        p[oo] = po[oo]
        p[ot] = pt[ot]
    elif policy == "union_pess":
        p[oo] = w * po[oo] + (1 - w) * ASSUMED_LOW
        p[ot] = w * ASSUMED_LOW + (1 - w) * pt[ot]
    else:
        raise ValueError(f"unknown policy {policy!r}")
    return j.select(["source1_entity_id", "candidate_entity_id"]).with_columns(
        pl.Series("p", p.astype(np.float32)))


def france_entities() -> set[str]:
    """Test Source-1 entities whose country is France (14.97% of the test set)."""
    df = pl.read_parquet(DATA / "parquet" / "test_source1.parquet",
                         columns=["entity_id", "country"])
    return set(df.filter(pl.col("country") == "France")["entity_id"].to_list())


def main(theirs_path: Path, policy: str = "union_pess", w: float = 0.5,
         write: bool = True, france_mode: str = "normal") -> dict:
    print("loading our test probabilities...")
    ours = load_ours()
    print(f"  ours: {ours.height:,} pairs")
    theirs = load_theirs(theirs_path)
    print(f"  theirs: {theirs.height:,} pairs")

    print(f"blending (policy={policy}, w={w})...")
    bl = blend(ours, theirs, policy, w)
    del ours, theirs

    cand: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for e, c, p in zip(bl["source1_entity_id"].to_list(),
                       bl["candidate_entity_id"].to_list(), bl["p"].to_list()):
        if p > 0:
            cand[e].append((c, p))
    del bl

    entities = required_entities("test")
    for e in entities:
        cand.setdefault(e, [])

    selected: dict[str, list[tuple[str, float]]] = {}
    for e, items in cand.items():
        if not items:
            selected[e] = []
            continue
        pr = np.fromiter((x[1] for x in items), dtype=np.float64, count=len(items))
        order = np.argsort(-pr)
        k, _ = choose_k(pr)
        selected[e] = [items[i] for i in order[:k]] if k else []
    matches = deconflict(selected)

    # France is 14.97% of test entities and has ZERO training labels, so its
    # behaviour is unvalidatable offline -- the only evidence is the leaderboard.
    # A teammate measured +0.003 from a France-altered submission, so both
    # variants are producible here and the leaderboard decides.
    if france_mode != "normal":
        fr = france_entities()
        n_before = sum(len(matches[e]) for e in fr if e in matches)
        if france_mode == "empty":
            for e in fr:
                matches[e] = []
        print(f"  france_mode={france_mode}: {len(fr):,} entities, "
              f"{n_before:,} predictions -> {sum(len(matches[e]) for e in fr if e in matches):,}")

    sizes = Counter(len(v) for v in matches.values())
    mean_pred = sum(len(v) for v in matches.values()) / len(entities)
    empty = sum(1 for v in matches.values() if not v)
    stats = {
        "policy": policy, "w": w,
        "mean_predictions": round(mean_pred, 3),
        "empty_predictions": empty,
        "empty_pct": round(100 * empty / len(entities), 2),
        "size_histogram": dict(sorted(sizes.items())[:12]),
    }
    print(f"\n  mean predictions/entity = {mean_pred:.3f}   (train truth mean 3.46)")
    print(f"  empty (singleton) = {empty:,} = {100*empty/len(entities):.2f}%   "
          f"(train singleton rate 5.58%)")
    print("  ^ these are the only sanity signal available: test has no labels.")

    if write:
        pre = preflight(entities, matches, candidates=None)
        if not pre["ok"]:
            print("PREFLIGHT_FAILED", pre["problems"])
            return {"ok": False, "preflight": pre, "stats": stats}
        OUTPUT.mkdir(parents=True, exist_ok=True)
        m_info = write_tsv(OUTPUT / "matching_results.tsv", MATCHING_HEADER, entities, matches)
        test_dir = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "test"
        cand_file = OUTPUT / "candidate_pairs.tsv"
        val = run_official_validator(OUTPUT / "matching_results.tsv",
                                     cand_file if cand_file.exists() else None, test_dir)
        stats["matching_file"] = m_info
        stats["validator_passed"] = val.get("passed")
        print(f"\nVALIDATOR passed={val.get('passed')}")
        print(val.get("stdout", "")[-500:])
        if val.get("passed"):
            print("ENSEMBLE_SUBMISSION_OK")
    (REPORTS / f"ensemble_{policy}_{france_mode}.json").write_text(json.dumps(stats, indent=2, default=str))
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--theirs", type=Path,
                    default=EXTERNAL / "v3_pair_probabilities.tsv")
    ap.add_argument("--policy", default="union_pess",
                    choices=["intersect", "union_mean", "union_pess"])
    ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--france", default="normal", choices=["normal", "empty"],
                    help="'empty' predicts nothing for all France entities")
    a = ap.parse_args()
    main(a.theirs, policy=a.policy, w=a.w, write=not a.dry_run, france_mode=a.france)
