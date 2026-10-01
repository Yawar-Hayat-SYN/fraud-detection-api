"""Tests for ml.features.uid.

The null tests come first on purpose. Building a UID by casting components to
str and concatenating turns a null into the literal text "null", so every row
missing a component lands in one giant shared bucket. Nothing errors; the
UID-level aggregates just quietly become garbage. This bug has already been hit
once on this project.
"""

import polars as pl
import pytest

from ml import config
from ml.data import load_transactions
from ml.features.time import add_time_features
from ml.features.uid import add_uid, add_uid_coarse, add_uid_tier


def frame(**overrides) -> pl.DataFrame:
    """Two rows with identical components. Overrides replace whole columns."""
    data = {"card1": [1000, 1000], "addr1": [315.0, 315.0], "D1n": [-12.0, -12.0]}
    data.update(overrides)
    return pl.DataFrame(data, schema={"card1": pl.Int64, "addr1": pl.Float64, "D1n": pl.Float64})


# --- Null handling: the critical cases -------------------------------------


def test_null_addr1_gives_null_uid():
    out = add_uid(frame(addr1=[None, 315.0]))

    assert out["uid"][0] is None
    assert out["uid"][1] is not None
    assert not out["uid"].drop_nulls().str.contains("null").any()


def test_null_d1n_gives_null_uid():
    out = add_uid(frame(D1n=[None, -12.0]))

    assert out["uid"][0] is None
    assert out["uid"][1] is not None


def test_null_card1_gives_null_uid_and_uid_coarse():
    out = add_uid_coarse(add_uid(frame(card1=[None, 1000])))

    assert out["uid"][0] is None
    assert out["uid_coarse"][0] is None


def test_nan_component_gives_null_uid():
    """NaN is a value in polars, not a null, so it needs its own guard."""
    out = add_uid(frame(D1n=[float("nan"), -12.0]))

    assert out["uid"][0] is None


def test_rows_with_different_missing_components_do_not_share_a_uid():
    out = add_uid(
        pl.DataFrame(
            {"card1": [1, 2, 3], "addr1": [None, None, 10.0], "D1n": [5.0, 6.0, None]},
            schema={"card1": pl.Int64, "addr1": pl.Float64, "D1n": pl.Float64},
        )
    )

    assert out["uid"].null_count() == 3


def test_uid_coarse_ignores_d1n():
    out = add_uid_coarse(frame(D1n=[None, -12.0]))

    assert out["uid_coarse"][0] is not None
    assert out["uid_coarse"][0] == out["uid_coarse"][1]


# --- Identity -----------------------------------------------------------------


def test_identical_components_give_identical_uid():
    out = add_uid(frame())

    assert out["uid"][0] == out["uid"][1]


@pytest.mark.parametrize(
    "override",
    [
        {"card1": [1000, 1001]},
        {"addr1": [315.0, 316.0]},
        {"D1n": [-12.0, -13.0]},
    ],
    ids=["card1", "addr1", "D1n"],
)
def test_any_differing_component_gives_different_uid(override):
    out = add_uid(frame(**override))

    assert out["uid"][0] != out["uid"][1]


def test_components_are_not_ambiguous_when_joined():
    """(1, 23) and (12, 3) must not both join to "123"."""
    out = add_uid(
        pl.DataFrame(
            {"card1": [1, 12], "addr1": [23.0, 3.0], "D1n": [0.0, 0.0]},
            schema={"card1": pl.Int64, "addr1": pl.Float64, "D1n": pl.Float64},
        )
    )

    assert out["uid"][0] != out["uid"][1]


# --- Tier ---------------------------------------------------------------------


def test_uid_tier():
    df = pl.DataFrame(
        {"card1": [1, 1, None], "addr1": [2.0, 2.0, 2.0], "D1n": [3.0, None, 3.0]},
        schema={"card1": pl.Int64, "addr1": pl.Float64, "D1n": pl.Float64},
    )

    out = add_uid_tier(add_uid_coarse(add_uid(df)))

    assert out["uid_tier"].to_list() == ["full", "coarse", "none"]


# --- Frame handling -----------------------------------------------------------


def test_lazy_and_eager_agree_and_input_unchanged():
    df = frame(addr1=[None, 315.0])
    before = df.columns.copy()

    eager = add_uid_tier(add_uid_coarse(add_uid(df)))
    lazy = add_uid_tier(add_uid_coarse(add_uid(df.lazy()))).collect()

    assert df.columns == before
    assert eager.equals(lazy)


# --- Real data ----------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not config.TRAIN_PARQUET.exists(),
    reason=f"{config.TRAIN_PARQUET} not found -- run python ml/load_raw.py",
)
def test_uid_splits_large_card1_groups():
    """Raw card1's biggest group is 14,932 rows. A working UID breaks it up.

    If nulls were collapsing into a shared string, the incomplete rows would
    form one huge group and this would fail.
    """
    lf = load_transactions(columns=["card1", "addr1", "D1", config.TIME_COL], lazy=True)
    sizes = (
        add_uid(add_time_features(lf))
        .filter(pl.col("uid").is_not_null())
        .group_by("uid")
        .len()
        .collect()
    )

    assert sizes["len"].max() < 1000
