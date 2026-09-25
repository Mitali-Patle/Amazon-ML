"""F_0.5 metric for entity resolution -- CONFIRMED official (PDF; see docs/PROBLEM_BRIEF.md sec. 4).

Per source1 entity, compare predicted id set P vs true id set T:
    TP=|P&T|, FP=|P-T|, FN=|T-P|
    F_b = (1+b^2)TP / ((1+b^2)TP + b^2*FN + FP),  b=0.5  (== 1.25*P*R/(0.25*P+R))
Singleton (T empty): P empty -> 1.0; any prediction -> 0.0.
Non-singleton with P empty -> 0.0 (TP=0). Entities missing from y_pred count as empty.
Score = MACRO mean over ALL source1 entities in y_true (default; this is the official rule).
AVERAGING="micro" (pooled TP/FP/FN) is kept only as a diagnostic; never report it as the score.
PDF worked example: P={S2-47,S2-193,S3-812} vs T={S2-47,S3-812} -> 2.5/3.5 = 0.714.

Duplicate ids inside a predicted list: the official validator REJECTS them (whole file unscored),
so real submissions must not contain any. Default here is strict=True: each extra copy counts as a
false positive (surfaces bugs in CV / the submission gate). strict=False collapses duplicates to a
set (lenient, diagnostic only).
Key guard: if y_pred is non-empty and shares ZERO keys with y_true -> ValueError (dtype/format
mismatch, e.g. int vs "S1-..."). If overlap < 100% -> UserWarning, unless allow_missing=True
(genuine partial scoring; missing anchors still count as empty predictions). An entirely empty
y_pred is allowed (all-empty baseline). Extra y_pred keys are ignored.
Empty y_true raises ValueError (mean over zero entities would be NaN). For per-country/stratum
loops use score_slices(), which returns None for empty slices instead of raising.
Inputs: dict/Series {anchor_id: iterable of ids or comma-separated string}.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

BETA = 0.5
AVERAGING = "macro"  # "macro" (per-anchor mean, singletons count) | "micro" (pooled counts)


def _is_na(x) -> bool:
    """True only for scalar missing values (None, NaN, pd.NA, NaT); never for lists/arrays."""
    if x is None or x is pd.NA:
        return True
    return bool(np.ndim(x) == 0 and not isinstance(x, str) and pd.isna(x))


def _to_list(x) -> list:
    if _is_na(x):
        return []
    if isinstance(x, str):
        return [t.strip() for t in x.split(",") if t.strip()]
    return [str(t).strip() for t in x if not _is_na(t) and str(t).strip()]


def _to_set(x) -> frozenset:
    return frozenset(_to_list(x))


def _check_keys(y_true, y_pred, allow_missing: bool, stacklevel: int = 3) -> None:
    """stacklevel counts from the caller of _check_keys (1 = the function that calls it)."""
    if len(y_pred) == 0:
        return
    tk = set(y_true.keys()); n_ov = len(tk & set(y_pred.keys()))
    if n_ov == 0:
        raise ValueError(
            f"y_pred shares zero keys with y_true (e.g. true key {next(iter(tk))!r} "
            f"vs pred key {next(iter(y_pred.keys()))!r}): dtype/format mismatch?")
    if n_ov < len(tk) and not allow_missing:
        warnings.warn(f"{len(tk) - n_ov}/{len(tk)} y_true anchors missing from y_pred are scored "
                      "as empty predictions; pass allow_missing=True if intended.", UserWarning,
                      stacklevel=stacklevel + 1)


def _counts(y_true, y_pred, strict: bool = True, allow_missing: bool = False, _sl: int = 3):
    if len(y_true) == 0:
        raise ValueError("y_true is empty: metric undefined")
    _check_keys(y_true, y_pred, allow_missing, _sl)
    ids = list(y_true.keys())
    tp = np.zeros(len(ids)); fp = np.zeros(len(ids)); fn = np.zeros(len(ids))
    for i, a in enumerate(ids):
        t = _to_set(y_true[a]); pl = _to_list(y_pred.get(a)); p = frozenset(pl)
        tp[i] = len(t & p); fn[i] = len(t - p)
        fp[i] = (len(pl) - tp[i]) if strict else len(p - t)  # strict: duplicate copies are FPs
    return ids, tp, fp, fn


def _fbeta(tp, fp, fn, beta):
    b2 = beta ** 2
    den = (1 + b2) * tp + b2 * fn + fp
    return np.where(den == 0, 1.0, (1 + b2) * tp / np.where(den == 0, 1, den))  # den==0: both empty


def per_anchor_scores(y_true, y_pred, beta: float = BETA, strict: bool = True,
                      allow_missing: bool = False, _sl: int = 3) -> pd.Series:
    ids, tp, fp, fn = _counts(y_true, y_pred, strict, allow_missing, _sl + 1)
    return pd.Series(_fbeta(tp, fp, fn, beta), index=ids)


def f_beta_score(y_true, y_pred, beta: float = BETA, averaging: str = AVERAGING,
                 strict: bool = True, allow_missing: bool = False) -> float:
    """Anchors missing from y_pred are treated as empty predictions (warns unless allow_missing)."""
    if averaging == "macro":
        return float(per_anchor_scores(y_true, y_pred, beta, strict, allow_missing, 3).mean())
    if averaging == "micro":
        _, tp, fp, fn = _counts(y_true, y_pred, strict, allow_missing, 3)
        return float(_fbeta(tp.sum(), fp.sum(), fn.sum(), beta))
    raise ValueError(averaging)


def score_slices(y_true, y_pred, groups, beta: float = BETA, averaging: str = AVERAGING,
                 strict: bool = True, allow_missing: bool = False) -> dict:
    """Score per slice. groups: {anchor_id: label} (or Series). Returns {label: score or None};
    None when a slice has no anchors in y_true (so per-country loops never hit the empty ValueError).
    Key guard runs ONCE on the full y_true/y_pred (zero overlap -> ValueError; partial overlap ->
    UserWarning unless allow_missing=True). Only anchors present in both y_true and groups are used."""
    _check_keys(y_true, y_pred, allow_missing, 2)
    groups = dict(groups)
    out = {g: None for g in groups.values()}
    buckets: dict = {}
    for a, g in groups.items():
        if a in y_true:
            buckets.setdefault(g, []).append(a)
    for g, anchors in buckets.items():
        yt = {a: y_true[a] for a in anchors}
        yp = {a: y_pred[a] for a in anchors if a in y_pred}
        out[g] = f_beta_score(yt, yp, beta, averaging, strict, True)  # already checked globally
    return out
