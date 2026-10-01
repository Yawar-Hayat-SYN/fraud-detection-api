"""Regression tests that lock in facts about the training data.

Every number here came out of EDA on the real train set. They are
intentionally brittle: nothing in them is a property the code should
guarantee, they are just what the data looked like when it was validated.

A failure means one of two things:
- the data changed (a different download, a regenerated Parquet), or
- a loading bug was introduced (rows dropped or duplicated, a bad join,
  nulls filled or coerced on the way in).

Either way, find out which before touching these numbers. Do not update an
expected value just to make a test pass.

All tests read the real train Parquet and are marked slow:
    pytest tests/test_data_facts.py
    pytest -m "not slow"     # to skip them
"""

import polars as pl
import pytest

from ml import config
from ml.data import load_transactions

pytestmark = pytest.mark.slow

RATE_TOLERANCE = 0.01  # relative


def rate(expected: float) -> object:
    return pytest.approx(expected, rel=RATE_TOLERANCE)


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    if not config.TRAIN_PARQUET.exists():
        pytest.skip(f"{config.TRAIN_PARQUET} not found -- run python ml/load_raw.py")
    return load_transactions(
        columns=[config.TARGET_COL, config.PRODUCT_COL, "card1", "addr1", "dist1"]
    )


@pytest.fixture(scope="module")
def by_product(df) -> dict[str, dict]:
    """Per-ProductCD row count, fraud rate and null rates, keyed by product."""
    stats = df.group_by(config.PRODUCT_COL).agg(
        pl.len().alias("rows"),
        pl.col(config.TARGET_COL).mean().alias("fraud_rate"),
        pl.col("addr1").is_null().mean().alias("addr1_null"),
        pl.col("dist1").is_null().mean().alias("dist1_null"),
    )
    return {row[config.PRODUCT_COL]: row for row in stats.iter_rows(named=True)}


@pytest.fixture(scope="module")
def card1_counts(df) -> pl.Series:
    """Row count for each distinct card1 value."""
    return df.group_by("card1").len()["len"]


# --- Rows and labels ----------------------------------------------------------


def test_row_count(df):
    assert df.height == 590_540


def test_overall_fraud_rate(df):
    assert df[config.TARGET_COL].mean() == rate(0.0350)


@pytest.mark.parametrize(
    ("product", "expected"),
    [("W", 0.0204), ("C", 0.1169), ("R", 0.0378), ("H", 0.0477), ("S", 0.0590)],
)
def test_fraud_rate_by_product(by_product, product, expected):
    assert by_product[product]["fraud_rate"] == rate(expected)


def test_product_counts(by_product):
    counts = {p: s["rows"] for p, s in by_product.items()}

    assert counts == {"W": 439_670, "C": 68_519, "R": 37_699, "H": 33_024, "S": 11_628}


# --- Structural nulls ---------------------------------------------------------


def test_addr1_mostly_null_for_c(by_product):
    assert by_product["C"]["addr1_null"] == rate(0.95)


@pytest.mark.parametrize("product", ["W", "R", "H"])
def test_addr1_mostly_present_for_w_r_h(by_product, product):
    assert by_product[product]["addr1_null"] < 0.01


@pytest.mark.parametrize("product", ["C", "H", "R", "S"])
def test_dist1_is_w_only(by_product, product):
    assert by_product[product]["dist1_null"] == 1.0


# --- card1 cardinality --------------------------------------------------------


def test_card1_distinct_values(card1_counts):
    assert card1_counts.len() == 13_553


def test_card1_max_group_size(card1_counts):
    assert card1_counts.max() == 14_932


def test_card1_median_group_size(card1_counts):
    assert card1_counts.median() == 4


def test_card1_singletons(card1_counts):
    assert (card1_counts == 1).sum() == 3_444
