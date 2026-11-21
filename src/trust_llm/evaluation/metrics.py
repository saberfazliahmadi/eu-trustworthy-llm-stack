"""Evaluation metrics implemented from first principles (verified against scipy in the test-suite)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel


def perplexity(token_logprobs: Sequence[float]) -> float:
    """PPL = exp(-1/N * sum log p(x_i | x_<i)) using natural-log probabilities."""
    if len(token_logprobs) == 0:
        raise ValueError("perplexity requires at least one token")
    return float(math.exp(-float(np.mean(token_logprobs))))


def accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("inputs must be non-empty and of equal length")
    return float(np.mean([a == b for a, b in zip(y_true, y_pred)]))


def _average_ranks(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    sorted_vals = arr[order]
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0  # average rank for ties
        i = j + 1
    return ranks


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation = Pearson correlation of average ranks (tie-aware)."""
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("need two equal-length sequences with >= 2 items")
    rx, ry = _average_ranks(x), _average_ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = math.sqrt(float((rx**2).sum() * (ry**2).sum()))
    return float((rx * ry).sum() / denom) if denom else float("nan")


def cohens_kappa(a: Sequence, b: Sequence) -> float:
    """Cohen's kappa = (p_o - p_e) / (1 - p_e)."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length non-empty sequences")
    labels = sorted(set(a) | set(b), key=str)
    n = len(a)
    p_o = sum(x == y for x, y in zip(a, b)) / n
    p_e = sum((list(a).count(lab) / n) * (list(b).count(lab) / n) for lab in labels)
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return float((p_o - p_e) / (1 - p_e))


def thresholded_kappa(scores_a: Sequence[float], scores_b: Sequence[float], threshold: float = 3.0) -> float:
    """Binarise Likert scores at `threshold` (>= threshold -> 1) and compute Cohen's kappa."""
    return cohens_kappa([int(s >= threshold) for s in scores_a], [int(s >= threshold) for s in scores_b])


class McNemarResult(BaseModel):
    b: int  # A correct, B wrong
    c: int  # A wrong, B correct
    statistic: float
    p_value: float
    method: str


def mcnemar_test(correct_a: Sequence[bool], correct_b: Sequence[bool], exact: bool | None = None) -> McNemarResult:
    """McNemar's test on paired correctness. Exact binomial when b + c < 25, else chi2 with continuity."""
    if len(correct_a) != len(correct_b):
        raise ValueError("paired inputs required")
    b = sum(1 for x, y in zip(correct_a, correct_b) if x and not y)
    c = sum(1 for x, y in zip(correct_a, correct_b) if (not x) and y)
    n = b + c
    use_exact = (n < 25) if exact is None else exact
    if n == 0:
        return McNemarResult(b=b, c=c, statistic=0.0, p_value=1.0, method="exact" if use_exact else "chi2_cc")
    if use_exact:
        k = min(b, c)
        p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n
        return McNemarResult(b=b, c=c, statistic=float(k), p_value=min(1.0, p), method="exact")
    stat = (abs(b - c) - 1) ** 2 / n
    p = math.erfc(math.sqrt(stat / 2.0))  # survival function of chi2 with 1 dof
    return McNemarResult(b=b, c=c, statistic=float(stat), p_value=float(p), method="chi2_cc")


class PRF(BaseModel):
    precision: float
    recall: float
    f1: float


class MulticlassReport(BaseModel):
    accuracy: float
    micro: PRF
    macro: PRF
    per_class: dict[str, PRF]
    support: dict[str, int]
    micro_identity_holds: bool


def _prf(tp: int, fp: int, fn: int) -> PRF:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return PRF(precision=p, recall=r, f1=f)


def multiclass_prf(y_true: Sequence[str], y_pred: Sequence[str]) -> MulticlassReport:
    """Micro and macro P/R/F1. For single-label multiclass, micro-P = micro-R = micro-F1 = accuracy."""
    labels = sorted(set(y_true) | set(y_pred))
    per: dict[str, PRF] = {}
    tp_tot = fp_tot = fn_tot = 0
    support = {}
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        per[lab] = _prf(tp, fp, fn)
        support[lab] = sum(1 for t in y_true if t == lab)
        tp_tot, fp_tot, fn_tot = tp_tot + tp, fp_tot + fp, fn_tot + fn
    micro = _prf(tp_tot, fp_tot, fn_tot)
    present = [lab for lab in labels if support[lab] > 0]
    macro = PRF(
        precision=float(np.mean([per[lab].precision for lab in present])),
        recall=float(np.mean([per[lab].recall for lab in present])),
        f1=float(np.mean([per[lab].f1 for lab in present])),
    )
    acc = accuracy(list(y_true), list(y_pred))
    identity = all(abs(v - acc) < 1e-12 for v in (micro.precision, micro.recall, micro.f1))
    return MulticlassReport(accuracy=acc, micro=micro, macro=macro, per_class=per, support=support, micro_identity_holds=identity)


class ZTestResult(BaseModel):
    p1: float
    p2: float
    n1: int
    n2: int
    z: float
    p_value: float
    significant: bool


def two_proportion_z_test(x1: int, n1: int, x2: int, n2: int, alpha: float = 0.05) -> ZTestResult:
    """Pooled two-proportion z-test; significant iff |z| > z_{1-alpha/2} (1.96 for alpha = 0.05)."""
    if n1 == 0 or n2 == 0:
        raise ValueError("group sizes must be positive")
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    p_value = math.erfc(abs(z) / math.sqrt(2))
    crit = {0.05: 1.96, 0.01: 2.576, 0.10: 1.645}.get(alpha, 1.96)
    return ZTestResult(p1=p1, p2=p2, n1=n1, n2=n2, z=z, p_value=p_value, significant=abs(z) > crit)
