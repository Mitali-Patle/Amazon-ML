"""PHASE 5 -- the GO/NO-GO gate: does the cascade beat stage 1 alone?

Combines the two stages on the validation set and scores the real competition
metric through the real decision layer:

    p_final = p_gbdt                                   outside the band
    p_final = w*p_crossencoder + (1-w)*p_gbdt          inside the band

Sweeps the blend weight w. w=0 reproduces stage 1 exactly, which is the honest
baseline the cascade must beat, and it is computed here rather than quoted so
the comparison uses identical code on identical entities.

If no w improves on w=0, the cascade is dropped and stage 1 ships unchanged.
"""
from __future__ import annotations

import json
import pickle
from collections import defaultdict

import numpy as np
import polars as pl

from .audit import load_pairs
from .config import DATA, REPORTS, REPR
from .decide import decide_all
from .metric import score_breakdown
from .train_matcher import encode, load_features

MODEL_PATH = DATA / "models" / "matcher.pkl"
CE_PREDS = DATA / "reports" / "val_crossencoder_preds.parquet"


def main(weights=(0.0, 0.5, 0.7, 0.85, 1.0)) -> dict:
    # stage-1 probabilities for EVERY validation pair
    with MODEL_PATH.open("rb") as fh:
        b = pickle.load(fh)
    val = load_features(DATA / "features" / "pairs" / "val")
    X, _ = encode(val, b["features"], b["cat_cols"], cat_levels=b["cat_levels"])
    p_gbdt = b["model"].predict_proba(X)[:, 1].astype(np.float64)
    del X
    base = val.select(["source1_entity_id", "candidate_entity_id"]).with_columns(
        pl.Series("p_gbdt", p_gbdt))
    del val

    ce = pl.read_parquet(CE_PREDS).select(
        ["source1_entity_id", "candidate_entity_id", "p_ce"])
    merged = base.join(ce, on=["source1_entity_id", "candidate_entity_id"], how="left")
    n_band = int(merged["p_ce"].is_not_null().sum())
    print(f"val pairs={merged.height:,}  in ambiguous band={n_band:,} "
          f"({100*n_band/merged.height:.2f}%)")

    val_ids = set(pl.read_parquet(DATA / "validation" / "split_assignment.parquet")
                  .filter(pl.col("fold") == "val")["entity_id"].to_list())
    truth = {e: set() for e in val_ids}
    for a, c in load_pairs().filter(pl.col("source1_entity_id").is_in(val_ids)).iter_rows():
        truth[a].add(c)
    country = dict(pl.read_parquet(REPR / "train_source1.parquet",
                                   columns=["entity_id", "country"])
                   .filter(pl.col("entity_id").is_in(val_ids)).iter_rows())

    eids = merged["source1_entity_id"].to_list()
    cids = merged["candidate_entity_id"].to_list()
    pg = merged["p_gbdt"].to_numpy()
    pc = merged["p_ce"].to_numpy()
    has_ce = ~np.isnan(pc)

    # optional joint recalibration of the blended score (fitted on TRAIN only)
    jc_path = DATA / "models" / "joint_calibrator.pkl"
    jc = None
    if jc_path.exists():
        import pickle as _pk
        with jc_path.open("rb") as _fh:
            jc = _pk.load(_fh)["iso"]
        print("joint calibrator loaded -- evaluating with and without it")

    results = {}
    for w in weights:
        p = pg.copy()
        p[has_ce] = w * pc[has_ce] + (1.0 - w) * pg[has_ce]
        cand = defaultdict(list)
        for e, c, pr in zip(eids, cids, p.tolist()):
            cand[e].append((c, pr))
        for e in val_ids:
            cand.setdefault(e, [])
        pred, diag = decide_all(cand, apply_deconflict=True)
        br = score_breakdown(pred, truth, groups=country)
        results[f"w={w}"] = {
            "macro_f05": br["macro_f05"],
            "singleton_accuracy": br["singleton_accuracy"],
            "mean_predictions": diag["mean_predictions"],
            "by_country": {g: v["macro_f05"] for g, v in br["by_group"].items()},
        }
        tag = "  <- stage 1 only (baseline)" if w == 0.0 else ""
        print(f"  w={w:<5} macro F_0.5 = {br['macro_f05']:.4f}   "
              f"singleton={br['singleton_accuracy']:.4f}   "
              f"mean_pred={diag['mean_predictions']:.3f}{tag}")

    if jc is not None:
        for w in (0.85, 1.0):
            p = pg.copy()
            p[has_ce] = w * pc[has_ce] + (1.0 - w) * pg[has_ce]
            p = jc.predict(p)
            cand = defaultdict(list)
            for e, c, pr in zip(eids, cids, p.tolist()):
                cand[e].append((c, pr))
            for e in val_ids:
                cand.setdefault(e, [])
            pred, diag = decide_all(cand, apply_deconflict=True)
            br = score_breakdown(pred, truth, groups=country)
            results[f"w={w}+jointcal"] = {
                "macro_f05": br["macro_f05"],
                "singleton_accuracy": br["singleton_accuracy"],
                "mean_predictions": diag["mean_predictions"],
                "by_country": {g: v["macro_f05"] for g, v in br["by_group"].items()},
            }
            print(f"  w={w}+jointcal macro F_0.5 = {br['macro_f05']:.4f}   "
                  f"singleton={br['singleton_accuracy']:.4f}   "
                  f"mean_pred={diag['mean_predictions']:.3f}")

    baseline = results["w=0.0"]["macro_f05"]
    best_w = max(results, key=lambda k: results[k]["macro_f05"])
    best = results[best_w]["macro_f05"]
    out = {"pairs": merged.height, "band_pairs": n_band, "results": results,
           "baseline_stage1": baseline, "best_w": best_w, "best_macro_f05": best,
           "improvement": round(best - baseline, 5),
           "verdict": "GO" if best > baseline + 1e-4 else "NO-GO"}
    (REPORTS / "cascade_eval.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nstage 1 alone : {baseline:.4f}")
    print(f"best cascade  : {best:.4f}  ({best_w})")
    print(f"improvement   : {out['improvement']:+.4f}")
    print(f"VERDICT: {out['verdict']}")
    return out


if __name__ == "__main__":
    main()
