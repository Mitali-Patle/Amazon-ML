"""Fit a JOINT calibrator over the blended cascade score.

Stage 1 (GBDT) and stage 2 (cross-encoder) are each isotonically calibrated on
their own, but they are never calibrated TOGETHER. The blend therefore has a
seam, and it is measurable on validation:

    predicted 0.1987 -> actual 0.1485      (over-confident by 34%)
    predicted 0.3856 -> actual 0.3290      (over-confident by 17%)

That range is exactly where the decision rule decides whether to take one more
candidate, so over-confidence there converts directly into false positives --
which F_0.5 penalises at 2x. It also inflates singleton predictions, since the
empty-vs-nonempty comparison uses the same numbers.

The calibrator is fitted on the INTERNAL-TRAIN split only (GBDT scores plus
cross-encoder scores on the train band), then applied at inference. Validation
is never used to fit it, so the validation estimate stays honest.
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import polars as pl

from .config import DATA, REPORTS
from .train_matcher import encode, load_features

OUT = DATA / "models" / "joint_calibrator.pkl"
CE_DIR = DATA / "models" / "crossencoder"


def train_band_scores(batch: int = 256, max_len: int = 128) -> pl.DataFrame:
    """Cross-encoder scores for the TRAIN ambiguous band."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from .train_crossencoder import Collate, PairDS, predict

    amb = pl.read_parquet(DATA / "ambiguous" / "train_ambiguous.parquet")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(CE_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(CE_DIR).to(device)
    with (CE_DIR / "calibrator.pkl").open("rb") as fh:
        iso = pickle.load(fh)["iso"]
    dl = DataLoader(PairDS(amb, tok, max_len, with_label=False),
                    batch_size=batch, collate_fn=Collate(tok, max_len), num_workers=2)
    p = iso.predict(predict(model, dl, device)).astype(np.float32)
    return amb.select(["source1_entity_id", "candidate_entity_id"]).with_columns(
        pl.Series("p_ce", p))


def main() -> dict:
    from sklearn.isotonic import IsotonicRegression

    with (DATA / "models" / "matcher.pkl").open("rb") as fh:
        b = pickle.load(fh)
    tr = load_features(DATA / "features" / "pairs" / "train")
    X, _ = encode(tr, b["features"], b["cat_cols"], cat_levels=b["cat_levels"])
    pg = b["model"].predict_proba(X)[:, 1].astype(np.float64)
    del X
    base = tr.select(["source1_entity_id", "candidate_entity_id", "label"]).with_columns(
        pl.Series("p_gbdt", pg))
    del tr, pg

    ce = train_band_scores()
    m = base.join(ce, on=["source1_entity_id", "candidate_entity_id"], how="left")
    blended = np.where(m["p_ce"].is_null().to_numpy(),
                       m["p_gbdt"].to_numpy(), m["p_ce"].fill_null(0).to_numpy())
    y = m["label"].to_numpy()
    print(f"train pairs={m.height:,}  band={int(m['p_ce'].is_not_null().sum()):,}  "
          f"positives={int(y.sum()):,}")

    before = []
    for lo, hi in [(.01, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, .99)]:
        k = (blended >= lo) & (blended < hi)
        if k.sum() > 50:
            before.append({"band": f"{lo}-{hi}", "n": int(k.sum()),
                           "pred": round(float(blended[k].mean()), 4),
                           "actual": round(float(y[k].mean()), 4)})

    iso = IsotonicRegression(out_of_bounds="clip").fit(blended, y)
    after_p = iso.predict(blended)
    after = []
    for lo, hi in [(.01, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, .99)]:
        k = (after_p >= lo) & (after_p < hi)
        if k.sum() > 50:
            after.append({"band": f"{lo}-{hi}", "n": int(k.sum()),
                          "pred": round(float(after_p[k].mean()), 4),
                          "actual": round(float(y[k].mean()), 4)})

    with OUT.open("wb") as fh:
        pickle.dump({"iso": iso}, fh)

    print("\nBEFORE (train):")
    for r in before:
        print(f"   {r['band']:<9} n={r['n']:>8,d}  pred={r['pred']:.4f}  actual={r['actual']:.4f}")
    print("AFTER (train):")
    for r in after:
        print(f"   {r['band']:<9} n={r['n']:>8,d}  pred={r['pred']:.4f}  actual={r['actual']:.4f}")

    rep = {"train_pairs": int(m.height), "before": before, "after": after,
           "path": str(OUT)}
    (REPORTS / "joint_calibration.json").write_text(json.dumps(rep, indent=2, default=str))
    print("JOINT_CALIBRATION_DONE")
    return rep


if __name__ == "__main__":
    main()
