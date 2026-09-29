"""The official competition metric: macro-averaged F_0.5 per Source-1 entity.

    F_0.5 = (1.25 * P * R) / (0.25 * P + R)

computed PER Source-1 entity, then averaged over ALL entities in the
evaluation set. Precision is weighted 2x recall, so a false merge costs about
twice a missed match.

Three things about this metric drive the whole decision layer, and all three
are easy to get wrong:

  1. SINGLETONS COUNT. An entity with no true matches scores 1.0 when you
     correctly predict an empty list and 0.0 when you predict anything at all.
     5.58% of training entities are singletons, so they are 5.58% of the score
     while contributing 0% of the pairs -- a pair-level objective ignores them
     entirely.

  2. IT IS MACRO OVER ENTITIES, NOT PAIRS. Entities with few matches carry
     weight far above their share of pairs: those with <=2 matches are 28.0% of
     the macro average but only 11.4% of true pairs. Optimising pairwise F_0.5
     optimises the wrong quantity.

  3. PREDICTING NOTHING WHEN UNSURE IS OFTEN CORRECT. For an entity with 4 true
     matches, finding 3 with no false positives scores 0.938; finding all 4 plus
     one false positive scores 0.952; finding 3 plus two false positives scores
     0.682. The asymmetry is steep.

Worked example from the problem statement (regression-tested):
    predicted [S2-00047, S2-00193, S3-00812], truth [S2-00047, S3-00812]
    P = 2/3, R = 2/2 = 1.0  ->  F_0.5 = 0.714
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

BETA2 = 0.25          # beta^2 for beta = 0.5
ONE_PLUS_BETA2 = 1.25


def entity_f_beta(predicted: Iterable[str], truth: Iterable[str]) -> float:
    """F_0.5 for a single Source-1 entity.

    Both arguments are treated as SETS -- the official format forbids duplicate
    ids within a list, and scoring them twice would be wrong regardless.
    """
    p = set(predicted)
    t = set(truth)

    if not t:
        # Singleton: full credit only for predicting nothing.
        return 1.0 if not p else 0.0
    if not p:
        # Truth exists but nothing predicted: recall 0, precision undefined.
        return 0.0

    tp = len(p & t)
    if tp == 0:
        return 0.0

    precision = tp / len(p)
    recall = tp / len(t)
    return (ONE_PLUS_BETA2 * precision * recall) / (BETA2 * precision + recall)


def macro_f_beta(predictions: Mapping[str, Iterable[str]],
                 truth: Mapping[str, Iterable[str]],
                 *, strict: bool = True) -> float:
    """Macro-average F_0.5 over every entity in `truth`.

    The denominator is ALWAYS len(truth): an entity the submission omits scores
    0, it does not shrink the average. That mirrors the official scorer, where
    every Source-1 test entity must appear.

    strict=True raises if the submission contains entities absent from truth
    (a sign of a mis-joined or wrong-split submission). strict=False ignores them.
    """
    if strict:
        extra = set(predictions) - set(truth)
        if extra:
            raise ValueError(
                f"{len(extra)} predicted entities are not in truth, e.g. {sorted(extra)[:3]}")
    if not truth:
        return 0.0
    total = 0.0
    for eid, t in truth.items():
        total += entity_f_beta(predictions.get(eid, ()), t)
    return total / len(truth)


def score_breakdown(predictions: Mapping[str, Iterable[str]],
                    truth: Mapping[str, Iterable[str]],
                    groups: Mapping[str, str] | None = None) -> dict:
    """Macro F_0.5 plus the diagnostics needed to steer the decision layer.

    `groups` optionally maps entity_id -> a label (country, script, ...) to get
    a per-group macro score, since the test country mix differs sharply from
    train and a single number hides that.
    """
    per_entity: dict[str, float] = {}
    n_singleton = n_singleton_correct = 0
    tp = fp = fn = 0
    n_empty_pred = 0

    for eid, t in truth.items():
        t = set(t)
        p = set(predictions.get(eid, ()))
        s = entity_f_beta(p, t)
        per_entity[eid] = s
        if not t:
            n_singleton += 1
            if not p:
                n_singleton_correct += 1
        if not p:
            n_empty_pred += 1
        tp += len(p & t)
        fp += len(p - t)
        fn += len(t - p)

    n = len(truth) or 1
    macro = sum(per_entity.values()) / n
    micro_p = tp / (tp + fp) if (tp + fp) else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f = ((ONE_PLUS_BETA2 * micro_p * micro_r) / (BETA2 * micro_p + micro_r)
               if (BETA2 * micro_p + micro_r) else 0.0)

    out = {
        "macro_f05": round(macro, 6),
        "entities": len(truth),
        "singletons": n_singleton,
        "singletons_correct": n_singleton_correct,
        "singleton_accuracy": round(n_singleton_correct / n_singleton, 4) if n_singleton else None,
        "singleton_score_share": round(n_singleton / n, 4),
        "empty_predictions": n_empty_pred,
        "pair_tp": tp, "pair_fp": fp, "pair_fn": fn,
        "micro_precision": round(micro_p, 6),
        "micro_recall": round(micro_r, 6),
        "micro_f05": round(micro_f, 6),
        "perfect_entities": sum(1 for v in per_entity.values() if v == 1.0),
        "zero_entities": sum(1 for v in per_entity.values() if v == 0.0),
    }
    if groups:
        agg: dict[str, list[float]] = {}
        for eid, s in per_entity.items():
            agg.setdefault(groups.get(eid, "?"), []).append(s)
        out["by_group"] = {g: {"n": len(v), "macro_f05": round(sum(v) / len(v), 6)}
                           for g, v in sorted(agg.items())}
    return out
