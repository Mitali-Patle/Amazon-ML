"""Decision layer: candidate scores -> the final match set per Source-1 entity.

This is where F_0.5 is actually won or lost. Candidate generation sets the recall
ceiling; this module decides how much of that ceiling to claim.

THE ALGEBRA. Substituting P = TP/|S| and R = TP/|T| into
F_0.5 = 1.25*P*R / (0.25*P + R) reduces exactly to

        F_0.5 = 1.25 * TP / (0.25*|T| + |S|)

(verified against the direct form in tests). Two consequences:

  * the score is LINEAR in true positives, so candidates can be considered in
    descending-probability order and only PREFIXES of that order need checking --
    the optimal set is always a prefix, no subset search required;
  * |S| enters with 4x the weight of |T|. Adding a prediction costs 4x what the
    same-sized truth contributes, which is precisely why, for every entity with
    >=2 true matches, omitting a match beats adding a wrong one (0.9375 vs
    0.8333 at |T|=4). Only single-match entities reward aggression, because
    missing their one match scores 0.

THE RULE. Given calibrated probabilities p_i for each candidate, pick the k that
maximises expected score:

    E[F(top-k)]  ~=  1.25 * sum_{i<k} p_i / (0.25 * sum_all p_i + k)     k > 0
    E[F(empty)]   =  prod_i (1 - p_i)          # probability the entity IS a singleton

The k>0 form is the standard plug-in approximation (expected values substituted
into the closed form); the k=0 form is exact. Comparing them handles singleton
detection as a natural consequence rather than a separate threshold -- which
matters because singletons are statistically invisible from the Source-1 record
and can only be detected by the absence of a confident match.

CALIBRATION MATTERS MORE THAN THE RULE. The rule consumes probabilities. Fed raw
uncalibrated scores it still works structurally but loses its optimality, so the
matcher's output must be calibrated on the internal-train split before this runs.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

BETA2 = 0.25
ONE_PLUS_BETA2 = 1.25


def expected_scores(probs: Sequence[float]) -> np.ndarray:
    """Expected F_0.5 for every prefix length k = 0..n, given sorted-desc probs."""
    p = np.asarray(probs, dtype=np.float64)
    n = len(p)
    out = np.empty(n + 1, dtype=np.float64)
    # k = 0: exact -- the entity is a singleton with probability prod(1 - p_i)
    out[0] = float(np.prod(1.0 - p)) if n else 1.0
    if n == 0:
        return out
    expected_truth = float(p.sum())
    cum_tp = np.cumsum(p)
    ks = np.arange(1, n + 1, dtype=np.float64)
    out[1:] = ONE_PLUS_BETA2 * cum_tp / (BETA2 * expected_truth + ks)
    return out


def choose_k(probs: Sequence[float], max_k: int | None = None) -> tuple[int, float]:
    """Return (best prefix length, its expected score). k=0 means predict empty."""
    p = np.asarray(probs, dtype=np.float64)
    if p.size == 0:
        return 0, 1.0
    order = np.argsort(-p)
    scores = expected_scores(p[order])
    if max_k is not None:
        scores = scores[: max_k + 1]
    k = int(np.argmax(scores))
    return k, float(scores[k])


def decide_entity(candidate_ids: Sequence[str], probs: Sequence[float],
                  max_k: int | None = None) -> list[str]:
    """Final match list for one entity: the expected-F_0.5-optimal prefix."""
    if not candidate_ids:
        return []
    p = np.asarray(probs, dtype=np.float64)
    order = np.argsort(-p)
    k, _ = choose_k(p, max_k=max_k)
    return [candidate_ids[i] for i in order[:k]]


# --------------------------------------------------------------------------
# mutual-exclusivity deconfliction
# --------------------------------------------------------------------------
def deconflict(assignments: dict[str, list[tuple[str, float]]]) -> dict[str, list[str]]:
    """Enforce the verified many-to-one constraint across entities.

    No Source-2/3 record is ever claimed by two Source-1 entities -- verified
    exhaustively over all 7,638,365 training pairs. So whenever two entities
    claim the same record, at least one is a false merge, and under a metric
    that weights precision 2x that is expensive.

    Resolves greedily: each contested record goes to its highest-probability
    claimant, losers drop it. Greedy rather than optimal assignment because the
    exact version is min-cost flow over ~10^8 edges; greedy captures nearly all
    of the benefit at negligible cost, and every resolution strictly removes a
    guaranteed false positive.

    Input:  entity_id -> [(candidate_id, probability), ...]
    Output: entity_id -> [candidate_id, ...] with no candidate appearing twice.
    """
    best: dict[str, tuple[str, float]] = {}
    for eid, items in assignments.items():
        for cid, p in items:
            cur = best.get(cid)
            if cur is None or p > cur[1] or (p == cur[1] and eid < cur[0]):
                best[cid] = (eid, p)
    out: dict[str, list[str]] = {}
    for eid, items in assignments.items():
        out[eid] = [cid for cid, _ in items if best[cid][0] == eid]
    return out


def decide_all(candidates: dict[str, list[tuple[str, float]]],
               max_k: int | None = None,
               apply_deconflict: bool = True) -> tuple[dict[str, list[str]], dict]:
    """Full decision pass over every entity, with diagnostics.

    Deconfliction runs AFTER per-entity selection so it only ever arbitrates
    between records both entities actually wanted to predict.
    """
    chosen: dict[str, list[tuple[str, float]]] = {}
    k_hist: dict[int, int] = {}
    for eid, items in candidates.items():
        if not items:
            chosen[eid] = []
            k_hist[0] = k_hist.get(0, 0) + 1
            continue
        ids = [c for c, _ in items]
        ps = [p for _, p in items]
        order = np.argsort(-np.asarray(ps, dtype=np.float64))
        k, _ = choose_k(ps, max_k=max_k)
        chosen[eid] = [(ids[i], ps[i]) for i in order[:k]]
        k_hist[k] = k_hist.get(k, 0) + 1

    before = sum(len(v) for v in chosen.values())
    if apply_deconflict:
        final = deconflict(chosen)
    else:
        final = {e: [c for c, _ in v] for e, v in chosen.items()}
    after = sum(len(v) for v in final.values())

    diag = {
        "entities": len(candidates),
        "predicted_pairs_before_deconflict": before,
        "predicted_pairs_after_deconflict": after,
        "removed_by_deconflict": before - after,
        "predicted_empty": sum(1 for v in final.values() if not v),
        "mean_predictions": round(after / len(candidates), 3) if candidates else 0.0,
        "k_histogram": dict(sorted(k_hist.items())),
    }
    return final, diag
