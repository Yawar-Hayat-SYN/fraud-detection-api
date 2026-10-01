"""Time features derived from TransactionDT.

TransactionDT's origin is arbitrary (see ml.config), so timestamp, hour and
dayofweek are only meaningful relative to each other, not as real calendar
values. day and D1n are pure offsets and do not depend on the anchor at all.
"""

from typing import TypeVar

import polars as pl

from ml.config import SECONDS_PER_DAY, TIME_COL, TIMESTAMP_COL, TRANSACTION_DT_REFERENCE

Frame = TypeVar("Frame", pl.DataFrame, pl.LazyFrame)


def add_time_features(df: Frame) -> Frame:
    """Return df with day, timestamp, hour, dayofweek and D1n added.

    Accepts a DataFrame or LazyFrame and returns the same type. The input is
    not modified.

    - day: whole days since the TransactionDT origin.
    - timestamp: TRANSACTION_DT_REFERENCE + TransactionDT seconds.
    - hour: hour of day, 0-23.
    - dayofweek: 0-6, Monday = 0 (same convention as datetime.weekday()).
    - D1n: day - D1, the day the card was first seen. Constant per card, which
      is what makes it usable as a UID component. Null wherever D1 is null.
    """
    # Each new column is built from shared expressions rather than from the
    # columns added before it, so everything fits in one with_columns call.
    # Polars evaluates the repeated subexpressions only once.
    day = (pl.col(TIME_COL) // SECONDS_PER_DAY).cast(pl.Int64)
    timestamp = pl.lit(TRANSACTION_DT_REFERENCE) + pl.duration(seconds=pl.col(TIME_COL))

    return df.with_columns(
        day.alias("day"),
        timestamp.alias(TIMESTAMP_COL),
        timestamp.dt.hour().alias("hour"),
        (timestamp.dt.weekday() - 1).alias("dayofweek"),
        # Plain subtraction propagates nulls; no fill_null / fill_nan here.
        (day - pl.col("D1")).alias("D1n"),
    )
