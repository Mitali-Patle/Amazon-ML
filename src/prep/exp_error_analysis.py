"""READ-ONLY diagnostic: decompose the gap between our score and the oracle.

Nothing here trains, tunes or writes into the pipeline. It loads the existing
matcher and the existing validation features and answers one question with
evidence: of the 0.0545 between our 0.9288 and the 0.9833 oracle ceiling, how
much is MODEL error (wrong probabilities) and how much is DECISION error (right
probabilities, wrong cut)?

The separating experiment is the "oracle-k" run: keep the model's probabilities
exactly as they are, but choose the prefix length k per entity WITH hindsight.
Whatever that recovers is decision-layer headroom; whatever it does not recover
is genuine model error and needs a better matcher.
"""
from __future__ import annotations

import json
import pickle
from collections import Counter, defaultdict

import numpy as np
import polars as pl

from .audit import load_pairs
from .config import DATA, REPORTS, REPR, VALIDATION
from .decide import choose_k, deconflict
from .metric import entity_f_beta, score_breakdown
from .train_matcher import encode, feature_columns, load_features, ID_COLS

MODEL_PATH = DATA / "models" / "matcher.pkl"


def best_k_with_hindsight(items, truth_set):
    """Highest F_0.5 achievable from these probabilities by choosing k optimally."""
    if not items:
        return 0, (1.0 if not truth_set else 0.0)
    probs = np.array([p for _, p in items])
    order = np.argsort(-probs)
    ids = [items[i][0] for i in order]
    best_k, best_s = 0, entity_f_beta([], truth_set)
    for k in range(1, len(ids) + 1):
        s = entity_f_beta(ids[:k], truth_set)
        if s > best_s:
            best_k, best_s = k, s
    return best_k, best_s


def main() -> dict:
    with MODEL_PATH.open("rb") as fh:
        b = pickle.load(fh)
    model, feats, cat_cols, levels = b["model"], b["features"], b["cat_cols"], b["cat_levels"]

    val = load_features(DATA / "features" / "pairs" / "val")
    X, _ = encode(val, feats, cat_cols, cat_levels=levels)
    p = model.predict_proba(X)[:, 1]
    del X
    print(f"val pairs={val.height:,}  positives={int(val['label'].sum()):,}")

    eids = val["source1_entity_id"].to_list()
    cids = val["candidate_entity_id"].to_list()
    labs = val["label"].to_list()
    scripts = val["name_script_pair"].to_list()
    srcs = val["candidate_source"].to_list()

    cand: dict[str, list[tuple[str, float]]] = defaultdict(list)
    retrieved_truth: dict[str, set[str]] = defaultdict(set)
    pair_script: dict[tuple[str, str], str] = {}
    for e, c, pr, l, sc, sr in zip(eids, cids, p, labs, scripts, srcs):
        cand[e].append((c, float(pr)))
        pair_script[(e, c)] = sc
        if l:
            retrieved_truth[e].add(c)

    val_ids = set(pl.read_parquet(VALIDATION / "split_assignment.parquet")
                  .filter(pl.col("fold") == "val")["entity_id"].to_list())
    truth: dict[str, set[str]] = {e: set() for e in val_ids}
    for a, c in load_pairs().filter(pl.col("source1_entity_id").is_in(val_ids)).iter_rows():
        truth[a].add(c)
    for e in val_ids:
        cand.setdefault(e, [])

    country = dict(pl.read_parquet(REPR / "train_source1.parquet",
                                   columns=["entity_id", "country"])
                   .filter(pl.col("entity_id").is_in(val_ids)).iter_rows())

    # ---------------- scenarios ----------------
    def decide_current(items):
        if not items:
            return []
        pr = np.array([x[1] for x in items])
        order = np.argsort(-pr)
        k, _ = choose_k(pr)
        return [items[i][0] for i in order[:k]]

    scen: dict[str, dict[str, list[str]]] = {}
    scen["current"] = {e: decide_current(v) for e, v in cand.items()}
    scen["current+deconflict"] = deconflict(
        {e: [(c, dict(v).get(c, 0.0)) for c in scen["current"][e]] for e, v in cand.items()})
    # model probabilities kept, k chosen with hindsight -> DECISION headroom
    scen["oracle_k"] = {}
    kdiff = Counter()
    for e, v in cand.items():
        k, _ = best_k_with_hindsight(v, truth[e])
        pr = np.array([x[1] for x in v]) if v else np.array([])
        order = np.argsort(-pr) if v else []
        scen["oracle_k"][e] = [v[i][0] for i in order[:k]]
        cur = len(scen["current"][e])
        kdiff[("under" if k > cur else "over" if k < cur else "same")] += 1
    # perfect matcher on retrieved candidates -> BLOCKING ceiling
    scen["oracle_full"] = {e: (truth[e] & {c for c, _ in v}) for e, v in cand.items()}

    results = {}
    for name, pred in scen.items():
        results[name] = score_breakdown(pred, truth, groups=country)
        print(f"{name:22s} macro={results[name]['macro_f05']:.4f}  "
              f"singleton_acc={results[name]['singleton_accuracy']}")

    # ---------------- loss attribution on the current run ----------------
    cur = scen["current+deconflict"]
    loss_singleton = loss_fp = loss_fn = loss_unretrievable = 0.0
    n = len(truth)
    seg_loss = defaultdict(float)
    per_entity_loss = []
    for e, t in truth.items():
        got = set(cur[e])
        s = entity_f_beta(got, t)
        loss = 1.0 - s
        per_entity_loss.append((loss, e))
        if not t:
            loss_singleton += loss
        else:
            reach = t & {c for c, _ in cand[e]}
            if not reach:
                loss_unretrievable += loss
            else:
                # how much of this entity's loss is FP vs FN
                no_fp = entity_f_beta(got & t, t)          # drop all false positives
                all_tp = entity_f_beta(got | reach, t)     # add every reachable miss
                loss_fp += max(0.0, no_fp - s)
                loss_fn += max(0.0, all_tp - s)
        seg_loss[country[e]] += loss
    tot_loss = 1.0 - results["current+deconflict"]["macro_f05"]

    attribution = {
        "total_loss": round(tot_loss, 5),
        "singleton_errors": round(loss_singleton / n, 5),
        "unretrievable_entities": round(loss_unretrievable / n, 5),
        "false_positives": round(loss_fp / n, 5),
        "false_negatives": round(loss_fn / n, 5),
        "k_vs_hindsight": dict(kdiff),
    }

    # segment detail
    seg = {}
    for c in set(country.values()):
        ids = [e for e in truth if country[e] == c]
        seg[c] = {"entities": len(ids),
                  "macro": round(np.mean([entity_f_beta(cur[e], truth[e]) for e in ids]), 4),
                  "loss_share": round(seg_loss[c] / (tot_loss * n), 4) if tot_loss else 0}

    # cross-script pairs: are they being missed?
    cs_tot = cs_found = 0
    for e, t in truth.items():
        for c in t:
            sc = pair_script.get((e, c))
            if sc and sc != "latin|latin":
                cs_tot += 1
                if c in cur[e]:
                    cs_found += 1
    lat_tot = lat_found = 0
    for e, t in truth.items():
        for c in t:
            if pair_script.get((e, c)) == "latin|latin":
                lat_tot += 1
                if c in cur[e]:
                    lat_found += 1

    # worst entities
    per_entity_loss.sort(reverse=True)
    worst = [{"entity": e, "loss": round(l, 3), "n_truth": len(truth[e]),
              "n_pred": len(cur[e]), "n_reachable": len(truth[e] & {c for c, _ in cand[e]})}
             for l, e in per_entity_loss[:8]]

    out = {
        "scenarios": {k: {"macro_f05": v["macro_f05"],
                          "singleton_accuracy": v["singleton_accuracy"],
                          "by_country": {g: x["macro_f05"] for g, x in v["by_group"].items()}}
                      for k, v in results.items()},
        "loss_attribution": attribution,
        "by_country": seg,
        "recall_of_final_predictions": {
            "cross_script": {"true_pairs": cs_tot,
                             "recovered": round(100 * cs_found / cs_tot, 2) if cs_tot else None},
            "latin_latin": {"true_pairs": lat_tot,
                            "recovered": round(100 * lat_found / lat_tot, 2) if lat_tot else None},
        },
        "worst_entities": worst,
    }
    (REPORTS / "error_analysis.json").write_text(json.dumps(out, indent=2, default=str))

    print("\n=== LOSS ATTRIBUTION (macro F_0.5 points) ===")
    for k, v in attribution.items():
        print(f"   {k:26s} {v}")
    print("\n=== BY COUNTRY ===")
    for c, v in seg.items():
        print(f"   {c:8s} n={v['entities']:>7,d}  macro={v['macro']:.4f}  "
              f"share of loss={100*v['loss_share']:.1f}%")
    print("\n=== FINAL-PREDICTION RECALL BY SCRIPT ===")
    for k, v in out["recall_of_final_predictions"].items():
        print(f"   {k:14s} pairs={v['true_pairs']:>8,d}  recovered={v['recovered']}%")
    return out


if __name__ == "__main__":
    main()
