"""What stands between the cascade's 0.9654 and 0.98?

Decomposes the remaining loss into buckets that map to DIFFERENT fixes, so the
next hours go to the one that actually pays:

  A. unretrievable   - no true match in the candidate set. Only better blocking
                       (higher K) helps; no model can recover these.
  B. singleton FP    - a singleton we predicted something for. Costs a full 1.0
                       each. Fixed by the empty-vs-nonempty decision.
  C. false negatives - a retrievable true match we scored too low.
  D. false positives - a non-match we scored too high.

Also reports the ceiling reachable if each bucket alone were solved, and checks
whether the blended probabilities are still CALIBRATED -- the decision rule is
only optimal if they are, and stage 1 and stage 2 are calibrated separately,
never jointly.
"""
from __future__ import annotations

import json
import pickle
from collections import defaultdict

import numpy as np
import polars as pl

from .audit import load_pairs
from .config import DATA, REPORTS, REPR, VALIDATION
from .decide import choose_k, deconflict
from .metric import entity_f_beta
from .train_matcher import encode, load_features


def main() -> dict:
    with (DATA / "models" / "matcher.pkl").open("rb") as fh:
        b = pickle.load(fh)
    val = load_features(DATA / "features" / "pairs" / "val")
    X, _ = encode(val, b["features"], b["cat_cols"], cat_levels=b["cat_levels"])
    pg = b["model"].predict_proba(X)[:, 1].astype(np.float64)
    del X
    base = val.select(["source1_entity_id", "candidate_entity_id", "label"]).with_columns(
        pl.Series("p_gbdt", pg))
    del val, pg

    ce = pl.read_parquet(DATA / "reports" / "val_crossencoder_preds.parquet").select(
        ["source1_entity_id", "candidate_entity_id", "p_ce"])
    m = base.join(ce, on=["source1_entity_id", "candidate_entity_id"], how="left")
    p = np.where(m["p_ce"].is_null().to_numpy(), m["p_gbdt"].to_numpy(),
                 m["p_ce"].fill_null(0).to_numpy())
    y = m["label"].to_numpy()
    print(f"val pairs={m.height:,}  band={int(m['p_ce'].is_not_null().sum()):,}")

    # ---- is the BLENDED probability still calibrated? ----
    print("\n=== calibration of the blended score (rule is only optimal if calibrated) ===")
    calib = []
    for lo, hi in [(0, .01), (.01, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, .99), (.99, 1.01)]:
        k = (p >= lo) & (p < hi)
        if k.sum():
            calib.append({"band": f"{lo}-{hi}", "n": int(k.sum()),
                          "mean_p": round(float(p[k].mean()), 4),
                          "actual": round(float(y[k].mean()), 4)})
            print(f"  {lo:.2f}-{hi:<5.2f} n={k.sum():>9,d}  pred={p[k].mean():.4f}  actual={y[k].mean():.4f}")

    cand: dict[str, list[tuple[str, float]]] = defaultdict(list)
    lab: dict[str, set[str]] = defaultdict(set)
    for e, c, pr, yy in zip(m["source1_entity_id"].to_list(),
                            m["candidate_entity_id"].to_list(), p.tolist(), y.tolist()):
        cand[e].append((c, pr))
        if yy:
            lab[e].add(c)
    del m

    val_ids = set(pl.read_parquet(VALIDATION / "split_assignment.parquet")
                  .filter(pl.col("fold") == "val")["entity_id"].to_list())
    truth = {e: set() for e in val_ids}
    for a, c in load_pairs().filter(pl.col("source1_entity_id").is_in(val_ids)).iter_rows():
        truth[a].add(c)
    for e in val_ids:
        cand.setdefault(e, [])

    chosen = {}
    for e, items in cand.items():
        if not items:
            chosen[e] = []
            continue
        pr = np.array([x[1] for x in items])
        order = np.argsort(-pr)
        k, _ = choose_k(pr)
        chosen[e] = [(items[i][0], items[i][1]) for i in order[:k]]
    final = deconflict(chosen)

    n = len(truth)
    cur = float(np.mean([entity_f_beta(final[e], truth[e]) for e in truth]))
    print(f"\ncascade macro F_0.5 = {cur:.4f}")

    # ---- loss buckets ----
    loss = {"unretrievable": 0.0, "singleton_fp": 0.0, "false_neg": 0.0, "false_pos": 0.0}
    counts = {k: 0 for k in loss}
    for e, t in truth.items():
        got, s = set(final[e]), entity_f_beta(final[e], truth[e])
        gap = 1.0 - s
        if gap <= 1e-12:
            continue
        pool = {c for c, _ in cand[e]}
        if not t:
            loss["singleton_fp"] += gap
            counts["singleton_fp"] += 1
        elif not (t & pool):
            loss["unretrievable"] += gap
            counts["unretrievable"] += 1
        else:
            no_fp = entity_f_beta(got & t, t)
            all_tp = entity_f_beta(got | (t & pool), t)
            fp_part, fn_part = max(0.0, no_fp - s), max(0.0, all_tp - s)
            loss["false_pos"] += fp_part
            loss["false_neg"] += fn_part
            counts["false_pos"] += fp_part > 1e-9
            counts["false_neg"] += fn_part > 1e-9

    print("\n=== WHERE THE REMAINING LOSS IS (macro F_0.5 points) ===")
    tot = 1.0 - cur
    for k, v in sorted(loss.items(), key=lambda x: -x[1]):
        print(f"  {k:16s} {v/n:.5f}  ({100*(v/n)/tot:5.1f}% of remaining loss)  "
              f"entities={counts[k]:,}")

    print("\n=== CEILING IF EACH BUCKET ALONE WERE SOLVED ===")
    for k in loss:
        print(f"  fix {k:16s} -> {cur + loss[k]/n:.4f}")
    print(f"  fix everything       -> 1.0000")
    print(f"\n  to reach 0.98 we must recover {0.98 - cur:.4f} of the {tot:.4f} remaining")

    out = {"cascade_macro_f05": round(cur, 5),
           "loss_per_entity": {k: round(v / n, 5) for k, v in loss.items()},
           "entities_affected": counts,
           "ceiling_if_fixed": {k: round(cur + v / n, 4) for k, v in loss.items()},
           "needed_for_098": round(0.98 - cur, 5),
           "calibration": calib}
    (REPORTS / "gap_to_98.json").write_text(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    main()
