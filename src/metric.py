"""F_0.5 metric for entity resolution.

ASSUMPTION (UNCONFIRMED, official formula not yet in docs/PROBLEM_BRIEF.md):
  Per source1 anchor, compare predicted id set P vs true id set T:
      TP=|P&T|, FP=|P-T|, FN=|T-P|
      F_b = (1+b^2)TP / ((1+b^2)TP + b^2*FN + FP),  b=0.5  (precision-weighted)
  Singletons (T empty): P empty -> 1.0; P non-empty -> 0.0.
  Non-singleton with P empty -> 0.0 (falls out of the formula: TP=0).
  Score = mean over anchors (AVERAGING="macro"). Alternative "micro" pools TP/FP/FN
  globally (singletons then contribute nothing) -- swap via the `averaging` arg.
Inputs: dict/Series {anchor_id: iterable of ids or comma-separated string}.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

BETA = 0.5
AVERAGING = "macro"  # "macro" (per-anchor mean, singletons count) | "micro" (pooled counts)


def _to_set(x) -> frozenset:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return frozenset()
    if isinstance(x, str):
        return frozenset(t.strip() for t in x.split(",") if t.strip())
    return frozenset(str(t).strip() for t in x if str(t).strip())


def _counts(y_true, y_pred):
    ids = list(y_true.keys())
    tp = np.zeros(len(ids)); fp = np.zeros(len(ids)); fn = np.zeros(len(ids))
    for i, a in enumerate(ids):
        t = _to_set(y_true[a]); p = _to_set(y_pred.get(a))
        tp[i] = len(t & p); fp[i] = len(p - t); fn[i] = len(t - p)
    return ids, tp, fp, fn


def _fbeta(tp, fp, fn, beta):
    b2 = beta ** 2
    den = (1 + b2) * tp + b2 * fn + fp
    return np.where(den == 0, 1.0, (1 + b2) * tp / np.where(den == 0, 1, den))  # den==0: both empty


def per_anchor_scores(y_true, y_pred, beta: float = BETA) -> pd.Series:
    ids, tp, fp, fn = _counts(y_true, y_pred)
    return pd.Series(_fbeta(tp, fp, fn, beta), index=ids)


def f_beta_score(y_true, y_pred, beta: float = BETA, averaging: str = AVERAGING) -> float:
    """Anchors missing from y_pred are treated as empty predictions."""
    if averaging == "macro":
        return float(per_anchor_scores(y_true, y_pred, beta).mean())
    if averaging == "micro":
        _, tp, fp, fn = _counts(y_true, y_pred)
        return float(_fbeta(tp.sum(), fp.sum(), fn.sum(), beta))
    raise ValueError(averaging)
