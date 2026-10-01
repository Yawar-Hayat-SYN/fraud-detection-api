"""Time features derived from TransactionDT, and TransactionDT conversions.

Every conversion between TransactionDT and a datetime in this project goes
through here. Do not inline the arithmetic anywhere else -- see the note on
TRANSACTION_DT_REFERENCE in ml.config about why.

TransactionDT's origin is arbitrary (see ml.config), so timestamp, hour and
dayofweek are only meaningful relative to each other, not as real calendar
values. day and D1n are pure offsets and do not depend on the anchor at all.
"""

from datetime import datetime, timedelta
from typing import TypeVar

import polars as pl

from ml.config import SECONDS_PER_DAY, TIME_COL, TIMESTAMP_COL, TRANSACTION_DT_REFERENCE

Frame = TypeVar("Frame", pl.DataFrame, pl.LazyFrame)


def dt_to_timestamp(transaction_dt: int | float) -> datetime:
    """Convert a single raw TransactionDT value to a datetime.

    >>> dt_to_timestamp(86400)
    datetime.datetime(2017, 12, 2, 0, 0)
    """
    return TRANSACTION_DT_REFERENCE + timedelta(seconds=float(transaction_dt))


def timestamp_to_dt(ts: datetime) -> float:
    """Convert a datetime back to the raw TransactionDT scale.

    Needed when you want to express a split cutoff as a readable date but
    filter on the raw column.
    """
    return (ts - TRANSACTION_DT_REFERENCE).total_seconds()


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
