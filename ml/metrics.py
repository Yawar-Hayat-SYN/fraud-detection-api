"""Evaluation metrics.

The headline metric is precision at a fixed review budget, not AUC. A fraud
team can only manually review so many transactions a day, so what matters is
how many real frauds land in the top k by score. AUC averages over every
threshold, including ones nobody would ever operate at.

Every function accepts numpy arrays, lists, or polars Series.

Metrics that are undefined for the given labels (recall with no frauds, AUC
with only one class) return nan rather than raising or dividing by zero.
"""

import math
from collections.abc import Sequence

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from ml import config

ArrayLike = np.ndarray | pl.Series | Sequence[float]


def _as_arrays(y_true: ArrayLike, y_score: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Convert and validate inputs: 1-D, same length, binary labels, finite scores."""
    yt = np.asarray(y_true)
    ys = np.asarray(y_score, dtype=float)

    if yt.ndim != 1 or ys.ndim != 1:
        raise ValueError("y_true and y_score must be 1-D")
    if len(yt) != len(ys):
        raise ValueError(f"length mismatch: y_true has {len(yt)}, y_score has {len(ys)}")
    if len(yt) == 0:
        raise ValueError("y_true and y_score are empty")
    if not np.isin(yt, (0, 1)).all():
        raise ValueError("y_true must contain only 0/1 (or True/False), with no nulls")
    if not np.isfinite(ys).all():
        raise ValueError("y_score contains NaN or infinite values")

    return yt.astype(int), ys


def _resolve_k(k: int | float, n: int) -> int:
    """Turn k into a row count.

    An int is a count and must be in [1, n]. A float is a fraction of rows in
    (0, 1], rounded half up to the nearest row, and at least 1.
    """
    if isinstance(k, bool):
        raise TypeError("k must be an int or a float, not a bool")
    if isinstance(k, (int, np.integer)):
        if not 1 <= k <= n:
            raise ValueError(f"k={k} must be between 1 and the number of rows ({n})")
        return int(k)
    if isinstance(k, (float, np.floating)):
        if not 0 < k <= 1:
            raise ValueError(f"fractional k={k} must be in (0, 1]")
        # Round half up rather than ceil: 0.3 * 10 is 3.0000000000000004 in
        # floating point, which ceil would turn into 4.
        return max(1, min(n, math.floor(k * n + 0.5)))
    raise TypeError(f"k must be an int or a float, got {type(k).__name__}")


def _top_k_labels(yt: np.ndarray, ys: np.ndarray, k: int) -> np.ndarray:
    """Labels of the k highest-scoring rows.

    Ties at the cutoff are broken by original row order (stable sort), so the
    result is deterministic.
    """
    order = np.argsort(-ys, kind="stable")
    return yt[order[:k]]


def precision_at_k(y_true: ArrayLike, y_score: ArrayLike, k: int | float) -> float:
    """Share of the top k rows by score that are fraud.

    k is a row count (int) or a fraction of rows (float in (0, 1]).
    """
    yt, ys = _as_arrays(y_true, y_score)
    k_rows = _resolve_k(k, len(yt))
    return float(_top_k_labels(yt, ys, k_rows).sum() / k_rows)


def recall_at_k(y_true: ArrayLike, y_score: ArrayLike, k: int | float) -> float:
    """Share of all frauds that land in the top k rows by score.

    nan if there are no frauds.
    """
    yt, ys = _as_arrays(y_true, y_score)
    n_pos = yt.sum()
    if n_pos == 0:
        return math.nan
    k_rows = _resolve_k(k, len(yt))
    return float(_top_k_labels(yt, ys, k_rows).sum() / n_pos)


def recall_at_fpr(y_true: ArrayLike, y_score: ArrayLike, fpr: float) -> float:
    """Highest recall reachable with a threshold whose FPR is at most fpr.

    nan unless both classes are present.
    """
    if not 0 <= fpr <= 1:
        raise ValueError(f"fpr={fpr} must be in [0, 1]")
    yt, ys = _as_arrays(y_true, y_score)
    if yt.min() == yt.max():
        return math.nan
    fprs, tprs, _ = roc_curve(yt, ys)
    return float(tprs[fprs <= fpr].max())


def pr_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Area under the precision-recall curve, as average precision.

    For random scores this is about the base rate. nan if there are no frauds.
    """
    yt, ys = _as_arrays(y_true, y_score)
    if yt.sum() == 0:
        return math.nan
    return float(average_precision_score(yt, ys))


def roc_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Area under the ROC curve. nan unless both classes are present."""
    yt, ys = _as_arrays(y_true, y_score)
    if yt.min() == yt.max():
        return math.nan
    return float(roc_auc_score(yt, ys))


def _metrics(yt: np.ndarray, ys: np.ndarray, k: int | float, fpr: float) -> dict:
    n_pos = int(yt.sum())
    return {
        "n": len(yt),
        "n_positive": n_pos,
        "base_rate": n_pos / len(yt),
        "k": _resolve_k(k, len(yt)),
        "precision_at_k": precision_at_k(yt, ys, k),
        "recall_at_k": recall_at_k(yt, ys, k),
        "fpr": fpr,
        "recall_at_fpr": recall_at_fpr(yt, ys, fpr),
        "pr_auc": pr_auc(yt, ys),
        "roc_auc": roc_auc(yt, ys),
    }


def evaluate(
    y_true: ArrayLike,
    y_score: ArrayLike,
    product: ArrayLike | None = None,
    k: int | float = config.REVIEW_BUDGET,
    fpr: float = config.TARGET_FPR,
) -> dict:
    """All metrics in one dict, plus a per-product breakdown if product is given.

    The breakdown is under "by_product", keyed by product value. Each product
    gets the same review rate as the whole set: an int k is converted to the
    equivalent fraction of rows first, so a small product is never asked for
    more rows than it has.
    """
    yt, ys = _as_arrays(y_true, y_score)
    result = _metrics(yt, ys, k, fpr)

    if product is not None:
        prod = np.asarray(product)
        if prod.shape != yt.shape:
            raise ValueError(f"product has {len(prod)} rows, y_true has {len(yt)}")
        rate = result["k"] / len(yt)
        result["by_product"] = {
            str(p): _metrics(yt[prod == p], ys[prod == p], rate, fpr)
            for p in sorted(set(prod.tolist()), key=str)
        }

    return result
