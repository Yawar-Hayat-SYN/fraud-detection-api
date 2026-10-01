"""Tests for ml.metrics, against hand-computed values."""

import math

import numpy as np
import polars as pl
import pytest

from ml.metrics import (
    evaluate,
    pr_auc,
    precision_at_k,
    recall_at_fpr,
    recall_at_k,
    roc_auc,
)

# Ten rows, deliberately not in score order. Sorted by score descending:
#
#   score  0.95 0.90 0.80 0.70 0.60 0.50 0.40 0.30 0.20 0.10
#   label     1    0    1    1    0    0    1    0    0    0
#
# Top 3 holds 2 frauds -> precision 2/3. Top 5 holds 3 -> precision 3/5.
# There are 4 frauds in total, so recall at 5 is 3/4.
SCORES = [0.30, 0.95, 0.50, 0.70, 0.10, 0.90, 0.40, 0.20, 0.80, 0.60]
LABELS = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0]


def test_precision_at_k_by_hand():
    assert precision_at_k(LABELS, SCORES, 3) == pytest.approx(2 / 3)
    assert precision_at_k(LABELS, SCORES, 5) == pytest.approx(3 / 5)


def test_recall_at_k_by_hand():
    assert recall_at_k(LABELS, SCORES, 3) == pytest.approx(2 / 4)
    assert recall_at_k(LABELS, SCORES, 5) == pytest.approx(3 / 4)


def test_recall_at_fpr_by_hand():
    # 6 negatives. Allowing one false positive (FPR 1/6) lets the threshold
    # drop to 0.70, which catches 3 of 4 frauds. Zero FPR stops at 0.95: 1/4.
    assert recall_at_fpr(LABELS, SCORES, 1 / 6) == pytest.approx(3 / 4)
    assert recall_at_fpr(LABELS, SCORES, 0.0) == pytest.approx(1 / 4)


def test_perfect_separation():
    y = [0, 0, 0, 0, 1, 1, 1]
    s = [0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9]

    assert precision_at_k(y, s, 3) == 1.0
    assert roc_auc(y, s) == 1.0
    assert pr_auc(y, s) == 1.0
    assert recall_at_fpr(y, s, 0.0) == 1.0


def test_random_scores_give_pr_auc_near_base_rate():
    rng = np.random.default_rng(0)
    y = rng.random(50_000) < 0.05

    result = pr_auc(y, rng.random(50_000))

    assert result == pytest.approx(y.mean(), abs=0.005)


@pytest.mark.parametrize(("fraction", "count"), [(0.3, 3), (0.5, 5), (0.1, 1), (1.0, 10)])
def test_fractional_k_matches_integer_k(fraction, count):
    assert precision_at_k(LABELS, SCORES, fraction) == precision_at_k(LABELS, SCORES, count)
    assert recall_at_k(LABELS, SCORES, fraction) == recall_at_k(LABELS, SCORES, count)


def test_all_negative_labels_do_not_divide_by_zero():
    y = [0] * 10

    with np.errstate(all="raise"):
        assert precision_at_k(y, SCORES, 3) == 0.0
        assert math.isnan(recall_at_k(y, SCORES, 3))
        assert math.isnan(recall_at_fpr(y, SCORES, 0.01))
        assert math.isnan(pr_auc(y, SCORES))
        assert math.isnan(roc_auc(y, SCORES))
        result = evaluate(y, SCORES, k=3)

    assert result["base_rate"] == 0.0


@pytest.mark.parametrize("convert", [list, np.array, pl.Series], ids=["list", "numpy", "polars"])
def test_accepts_lists_numpy_and_polars(convert):
    assert precision_at_k(convert(LABELS), convert(SCORES), 3) == pytest.approx(2 / 3)


@pytest.mark.parametrize("k", [0, 11, 0.0, 1.5, True])
def test_invalid_k_raises(k):
    with pytest.raises((ValueError, TypeError)):
        precision_at_k(LABELS, SCORES, k)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="length mismatch"):
        precision_at_k(LABELS, SCORES[:-1], 3)
    with pytest.raises(ValueError, match="0/1"):
        precision_at_k([2] * 10, SCORES, 3)
    with pytest.raises(ValueError, match="NaN"):
        precision_at_k(LABELS, [math.nan] * 10, 3)


def test_evaluate_overall_and_by_product():
    product = ["W", "W", "C", "C", "W", "C", "W", "C", "W", "C"]

    result = evaluate(LABELS, SCORES, product=product, k=0.4, fpr=1 / 6)

    assert result["k"] == 4
    assert result["precision_at_k"] == pytest.approx(3 / 4)
    assert result["recall_at_fpr"] == pytest.approx(3 / 4)
    assert set(result["by_product"]) == {"C", "W"}

    # W rows by score: 0.95(1) 0.80(1) 0.40(1) 0.30(0) 0.10(0).
    # 40% of 5 rows is 2: both frauds.
    w = result["by_product"]["W"]
    assert w["n"] == 5
    assert w["k"] == 2
    assert w["precision_at_k"] == 1.0
    assert w["roc_auc"] == 1.0


def test_evaluate_int_k_becomes_same_rate_per_product():
    product = ["W"] * 8 + ["C"] * 2
    result = evaluate(LABELS, SCORES, product=product, k=5)

    assert result["by_product"]["W"]["k"] == 4
    assert result["by_product"]["C"]["k"] == 1
