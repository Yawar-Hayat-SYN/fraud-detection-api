"""Tests for ml.features.time.add_time_features."""

from datetime import datetime, timedelta

import polars as pl
from polars.testing import assert_frame_equal

from ml.config import SECONDS_PER_DAY, TIME_COL, TRANSACTION_DT_REFERENCE
from ml.features.time import add_time_features

NEW_COLUMNS = ["day", "timestamp", "hour", "dayofweek", "D1n"]


def make_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            TIME_COL: [0, SECONDS_PER_DAY - 1, 3 * SECONDS_PER_DAY + 7200, 10 * SECONDS_PER_DAY],
            "D1": [0.0, None, 2.0, None],
        }
    )


def test_known_transaction_dt_gives_expected_day():
    out = add_time_features(make_frame())

    assert out["day"].to_list() == [0, 0, 3, 10]


def test_known_transaction_dt_gives_expected_timestamp_hour_and_dayofweek():
    out = add_time_features(make_frame())

    expected_ts = TRANSACTION_DT_REFERENCE + timedelta(days=3, hours=2)
    assert out["timestamp"][2] == expected_ts
    assert out["hour"][2] == 2
    assert out["dayofweek"][2] == expected_ts.weekday()
    assert out["hour"][1] == 23


def test_d1n_is_null_where_d1_is_null():
    out = add_time_features(make_frame())

    assert out["D1n"].is_null().to_list() == out["D1"].is_null().to_list()
    assert out["D1n"].to_list() == [0.0, None, 1.0, None]


def test_input_frame_is_unchanged():
    df = make_frame()
    before = df.columns.copy()

    add_time_features(df)

    assert df.columns == before
    assert_frame_equal(df, make_frame())


def test_dataframe_and_lazyframe_give_identical_results():
    eager = add_time_features(make_frame())
    lazy = add_time_features(make_frame().lazy())

    assert isinstance(eager, pl.DataFrame)
    assert isinstance(lazy, pl.LazyFrame)
    assert_frame_equal(eager, lazy.collect())
    assert eager.columns == make_frame().columns + NEW_COLUMNS
    assert isinstance(eager["timestamp"][0], datetime)
