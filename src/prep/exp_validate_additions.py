"""Can we VALIDATE the additive ensemble? Partly -- and the part that matters.

The exact blend is unmeasurable: their probabilities cover test, our labels cover
validation, no overlap. But the additive design has only one free parameter that
can hurt us -- ADD_MIN, the confidence bar a pair must clear before we bolt it on
top of their submission. That IS measurable on our own validation set.

The simulation, run entirely on labelled validation data:

  base      = our cascade's decision-layer output (stands in for "their submission")
  additions = pairs scoring >= T that the decision layer did NOT already select
  measure   = macro F_0.5 of base, vs base + additions, at each T

If adding at threshold T raises macro F_0.5 here, the same operation is very
likely safe on top of their submission -- the pairs are ours, the threshold is
ours, and the metric is the real one. If it LOWERS it, we must not ship it.

This does not prove the combined score beats 0.974. It proves whether our
additions are net-positive or net-negative, which is exactly what sank the
previous attempt (that one REMOVED 249,750 of their ids; this one removes none).
"""
from __future__ import annotations

import json
import pickle
from collections import defaultdict

import numpy as np
import polars as pl

from .audit import load_pairs
from .config import DATA, REPORTS, VALIDATION
from .decide import choose_k, deconflict
from .metric import entity_f_beta
from .train_matcher import encode, load_features

THRESHOLDS = (0.90, 0.95, 0.97, 0.99, 0.995)


def main() -> dict:
    with (DATA / "models" / "matcher.pkl").open("rb") as fh:
        b = pickle.load(fh)
    val = load_features(DATA / "features" / "pairs" / "val")
    X, _ = encode(val, b["features"], b["cat_cols"], cat_levels=b["cat_levels"])
    pg = b["model"].predict_proba(X)[:, 1].astype(np.float64)
    del X

    ce = pl.read_parquet(DATA / "reports" / "val_crossencoder_preds.parquet").select(
        ["source1_entity_id", "candidate_entity_id", "p_ce"])
    m = (val.select(["source1_entity_id", "candidate_entity_id", "label"])
         .with_columns(pl.Series("p_gbdt", pg))
         .join(ce, on=["source1_entity_id", "candidate_entity_id"], how="left"))
    del val, pg
    p = np.where(m["p_ce"].is_null().to_numpy(), m["p_gbdt"].to_numpy(),
                 m["p_ce"].fill_null(0).to_numpy())

    jp = DATA / "models" / "joint_calibrator.pkl"
    if jp.exists():
        with jp.open("rb") as fh:
            p = pickle.load(fh)["iso"].predict(p)

    eids = m["source1_entity_id"].to_list()
    cids = m["candidate_entity_id"].to_list()
    labs = m["label"].to_list()
    del m

    cand: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for e, c, pr in zip(eids, cids, p.tolist()):
        cand[e].append((c, pr))
    val_ids = set(pl.read_parquet(VALIDATION / "split_assignment.parquet")
                  .filter(pl.col("fold") == "val")["entity_id"].to_list())
    truth = {e: set() for e in val_ids}
    for a, c in load_pairs().filter(pl.col("source1_entity_id").is_in(val_ids)).iter_rows():
        truth[a].add(c)
    for e in val_ids:
        cand.setdefault(e, [])

    # BASE: the ordinary decision-layer output
    chosen = {}
    for e, items in cand.items():
        if not items:
            chosen[e] = []
            continue
        pr = np.fromiter((x[1] for x in items), dtype=np.float64, count=len(items))
        order = np.argsort(-pr)
        k, _ = choose_k(pr)
        chosen[e] = [(items[i][0], items[i][1]) for i in order[:k]]
    base = deconflict(chosen)
    base_score = float(np.mean([entity_f_beta(base[e], truth[e]) for e in truth]))
    base_ids = sum(len(v) for v in base.values())
    print(f"BASE (our cascade): macro F_0.5 = {base_score:.4f}, {base_ids:,} ids\n")

    lab_map = {(e, c): l for e, c, l in zip(eids, cids, labs)}
    rows = []
    print(f"{'threshold':>10} {'additions':>10} {'precision':>10} {'macro F_0.5':>12} {'delta':>9}")
    for T in THRESHOLDS:
        aug = {e: list(v) for e, v in base.items()}
        claimed = {c for v in base.values() for c in v}
        n_add = n_correct = 0
        for e, items in cand.items():
            have = {c for c in aug.get(e, [])}
            for c, pr in items:
                if pr >= T and c not in have and c not in claimed:
                    aug.setdefault(e, []).append(c)
                    claimed.add(c)
                    n_add += 1
                    n_correct += int(lab_map.get((e, c), 0))
        s = float(np.mean([entity_f_beta(aug[e], truth[e]) for e in truth]))
        prec = n_correct / n_add if n_add else None
        rows.append({"threshold": T, "additions": n_add,
                     "precision": round(prec, 4) if prec is not None else None,
                     "macro_f05": round(s, 5), "delta": round(s - base_score, 5)})
        print(f"{T:>10.3f} {n_add:>10,d} {(f'{prec:.4f}' if prec else '-'):>10} "
              f"{s:>12.4f} {s-base_score:>+9.5f}")

    best = max(rows, key=lambda r: r["macro_f05"])
    verdict = "SAFE" if best["delta"] > 0 else "DO NOT ADD"
    print(f"\nbest threshold: {best['threshold']} -> {best['macro_f05']:.4f} "
          f"({best['delta']:+.5f})")
    print(f"VERDICT: {verdict}")
    print("\nNOTE: this validates the ADDITION step only. The combined score against")
    print("their 0.974 cannot be measured -- no labelled overlap exists.")

    out = {"base_macro_f05": round(base_score, 5), "base_ids": base_ids,
           "rows": rows, "best": best, "verdict": verdict}
    (REPORTS / "validate_additions.json").write_text(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    main()
