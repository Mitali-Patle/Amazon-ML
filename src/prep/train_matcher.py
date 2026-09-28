"""Train the pair matcher, calibrate it, and score it with the real metric.

Pipeline: train features -> GBDT -> probability calibration -> decision layer
-> macro F_0.5 on the internal validation set.

TWO THINGS THIS MODULE EXISTS TO GET RIGHT:

1. CALIBRATION IS NOT OPTIONAL. The decision layer maximises expected F_0.5 from
   per-candidate probabilities. Measured in tests/test_decide.py: with calibrated
   probabilities the rule beats a hindsight-tuned fixed threshold (0.7453 vs
   0.7313); with miscalibrated ones it loses by ~7pp. A raw GBDT score is not a
   calibrated probability, so an isotonic calibrator is fitted on a held-out
   slice of the INTERNAL-TRAIN data -- never on validation, which must stay a
   clean estimate of test behaviour.

2. THE SCORE IS MACRO PER ENTITY, NOT PAIRWISE. Optimising pairwise AUC or
   pairwise F points the wrong way: entities with <=2 matches are 28.0% of the
   macro average but only 11.4% of pairs, and singletons are 5.58% of the score
   while contributing 0% of pairs. Every number reported here is the real
   competition metric computed through the real decision layer.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from .audit import load_pairs
from .config import DATA, REPORTS, REPR, SEED, VALIDATION
from .decide import decide_all
from .metric import score_breakdown

FEATURES_DIR = DATA / "features" / "pairs"
MODEL_DIR = DATA / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

ID_COLS = {"source1_entity_id", "candidate_entity_id", "label"}


def load_features(split_dir: Path) -> pl.DataFrame:
    shards = sorted(split_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no feature shards in {split_dir}")
    return pl.concat([pl.read_parquet(s) for s in shards])


NUMERIC = (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8,
           pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8, pl.Boolean)


def feature_columns(df: pl.DataFrame) -> list[str]:
    """Numeric feature columns only. Ids and label are excluded by name;
    string columns (script pair labels, source) are one-hot encoded separately.

    The dtype list must cover EVERY width the feature module emits. It writes
    Int8 flags (exact-match, pin-equal, country-equal, cross-script) and UInt32
    counts; an incomplete list drops them silently, with no error and a quietly
    worse model. Asserted in `assert_all_features_used` below.
    """
    return [c for c, t in df.schema.items() if c not in ID_COLS and t in NUMERIC]


def assert_all_features_used(df: pl.DataFrame, feats: list[str],
                             cat_cols: list[str]) -> None:
    """Fail loudly if any produced column is neither used nor deliberately an id."""
    accounted = set(feats) | set(cat_cols) | ID_COLS
    dropped = [c for c in df.columns if c not in accounted]
    if dropped:
        raise ValueError(
            f"{len(dropped)} feature columns would be silently dropped: {dropped}. "
            "Add their dtype to NUMERIC or to the categorical handling.")


def encode(df: pl.DataFrame, feats: list[str], cat_cols: list[str],
           cat_levels: dict[str, list[str]] | None = None
           ) -> tuple[np.ndarray, dict[str, list[str]]]:
    """Numeric block + one-hot for the few categorical columns."""
    X = df.select(feats).to_numpy().astype(np.float32)
    levels = cat_levels or {}
    blocks = [X]
    for c in cat_cols:
        vals = df[c].to_list()
        if c not in levels:
            levels[c] = sorted({v for v in vals if v is not None})
        idx = {v: i for i, v in enumerate(levels[c])}
        oh = np.zeros((len(vals), len(levels[c])), dtype=np.float32)
        for r, v in enumerate(vals):
            j = idx.get(v)
            if j is not None:
                oh[r, j] = 1.0
        blocks.append(oh)
    return np.hstack(blocks), levels


def main() -> dict:
    t0 = time.time()
    from sklearn.calibration import CalibratedClassifierCV  # noqa: PLC0415
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415

    train = load_features(FEATURES_DIR / "train")
    print(f"train pairs: {train.height:,}  positives: {int(train['label'].sum()):,} "
          f"({100*train['label'].mean():.3f}%)")

    feats = feature_columns(train)
    cat_cols = [c for c, t in train.schema.items()
                if c not in ID_COLS and t == pl.String]
    assert_all_features_used(train, feats, cat_cols)
    print(f"numeric features: {len(feats)}  categorical: {cat_cols}")

    y = train["label"].to_numpy().astype(np.int8)
    X, levels = encode(train, feats, cat_cols)

    # Hold out a slice of INTERNAL-TRAIN for calibration. Validation is never
    # touched here -- it has to remain an honest estimate of test behaviour.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y))
    n_cal = max(1, int(0.15 * len(y)))
    cal_idx, fit_idx = perm[:n_cal], perm[n_cal:]
    print(f"fit rows: {len(fit_idx):,}   calibration rows: {len(cal_idx):,}")

    # max_iter raised 300 -> 1000: the 300-iteration run hit its cap with
    # early stopping never firing (n_iter_ == max_iter), i.e. the model was
    # still improving when it was cut off. Patience raised so stopping is
    # decided by the validation curve rather than by the budget.
    clf = HistGradientBoostingClassifier(
        max_iter=1000, learning_rate=0.1, max_leaf_nodes=63,
        min_samples_leaf=50, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
        random_state=SEED,
    )
    t_fit = time.time()
    clf.fit(X[fit_idx], y[fit_idx])
    print(f"GBDT fitted in {time.time()-t_fit:.0f}s ({clf.n_iter_} iterations)")

    # Isotonic calibration on the held-out train slice.
    # sklearn >= 1.6 replaced cv="prefit" with FrozenEstimator; fall back for
    # older versions so the module is not pinned to one sklearn release.
    t_cal = time.time()
    try:
        from sklearn.frozen import FrozenEstimator  # noqa: PLC0415
        cal = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic")
    except ImportError:  # pragma: no cover - older sklearn
        cal = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
    cal.fit(X[cal_idx], y[cal_idx])
    print(f"calibrated in {time.time()-t_cal:.0f}s")

    del X, train

    # ---------------- evaluate on internal validation ----------------
    val = load_features(FEATURES_DIR / "val")
    Xv, _ = encode(val, feats, cat_cols, cat_levels=levels)
    pv = cal.predict_proba(Xv)[:, 1]
    print(f"val pairs: {val.height:,}")

    # calibration quality: mean predicted probability vs actual positive rate
    yv = val["label"].to_numpy().astype(np.int8)
    bins = np.clip((pv * 10).astype(int), 0, 9)
    calib = [(round(0.1 * b + 0.05, 2), int((bins == b).sum()),
              round(float(pv[bins == b].mean()), 4) if (bins == b).any() else None,
              round(float(yv[bins == b].mean()), 4) if (bins == b).any() else None)
             for b in range(10)]

    cand: dict[str, list[tuple[str, float]]] = {}
    for eid, cid, p in zip(val["source1_entity_id"].to_list(),
                           val["candidate_entity_id"].to_list(), pv.tolist()):
        cand.setdefault(eid, []).append((cid, p))

    # every validation entity must be scored, including ones whose candidate
    # list is empty -- omitting them would silently inflate the macro average
    val_ids = set(pl.read_parquet(VALIDATION / "split_assignment.parquet")
                  .filter(pl.col("fold") == "val")["entity_id"].to_list())
    for e in val_ids:
        cand.setdefault(e, [])

    truth: dict[str, set[str]] = {e: set() for e in val_ids}
    for a, b in load_pairs().filter(pl.col("source1_entity_id").is_in(val_ids)).iter_rows():
        truth[a].add(b)

    country = dict(pl.read_parquet(REPR / "train_source1.parquet",
                                   columns=["entity_id", "country"])
                   .filter(pl.col("entity_id").is_in(val_ids)).iter_rows())

    results = {}
    for name, kwargs in (("with_deconflict", {"apply_deconflict": True}),
                         ("no_deconflict", {"apply_deconflict": False})):
        pred, diag = decide_all(cand, **kwargs)
        br = score_breakdown(pred, truth, groups=country)
        results[name] = {"decision": diag, "score": br}
        print(f"\n[{name}] macro F_0.5 = {br['macro_f05']:.4f}   "
              f"singleton acc = {br['singleton_accuracy']}   "
              f"mean preds = {diag['mean_predictions']}")
        print(f"   by country: { {g: v['macro_f05'] for g, v in br['by_group'].items()} }")

    best = max(results, key=lambda k: results[k]["score"]["macro_f05"])
    macro = results[best]["score"]["macro_f05"]

    import pickle  # noqa: PLC0415
    with (MODEL_DIR / "matcher.pkl").open("wb") as fh:
        pickle.dump({"model": cal, "features": feats, "cat_cols": cat_cols,
                     "cat_levels": levels}, fh)

    report = {
        "train_pairs": int(len(y)), "positives": int(y.sum()),
        "n_features": int(len(feats) + sum(len(v) for v in levels.values())),
        "gbdt_iterations": int(clf.n_iter_),
        "calibration_bins": calib,
        "results": results, "best_config": best,
        "val_macro_f05": macro,
        "total_seconds": round(time.time() - t0, 1),
    }
    (REPORTS / "matcher.json").write_text(json.dumps(report, indent=2, default=str))

    print("\ncalibration check (bin, n, mean predicted, actual rate):")
    for row in calib:
        if row[1]:
            print(f"   {row[0]:.2f}  n={row[1]:>8,d}  pred={row[2]}  actual={row[3]}")
    print(f"\nVAL_MACRO_F05 {macro:.4f}  (best config: {best})")
    print(f"MATCHER_DONE in {report['total_seconds']}s")
    return report


if __name__ == "__main__":
    main()
