"""Conversions between the dataset's raw TransactionDT and real datetimes.

Every time conversion in this project goes through here. Do not inline the
arithmetic anywhere else -- see the note on TRANSACTION_DT_REFERENCE in
config.py about why.
"""

from datetime import datetime, timedelta

import pandas as pd

from ml.config import TIME_COL, TIMESTAMP_COL, TRANSACTION_DT_REFERENCE


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


def add_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised version for a whole DataFrame.

    Uses pandas' to_timedelta rather than applying dt_to_timestamp row by row
    -- on 590k rows the difference is seconds versus minutes.
    """
    df = df.copy()
    df[TIMESTAMP_COL] = TRANSACTION_DT_REFERENCE + pd.to_timedelta(df[TIME_COL], unit="s")
    return df


def describe_range(df: pd.DataFrame, label: str) -> dict:
    """Print and return the time range of a frame. Used for the sanity check."""
    lo = df[TIMESTAMP_COL].min()
    hi = df[TIMESTAMP_COL].max()
    span = (hi - lo).days

    print(f"{label}:")
    print(f"  from {lo:%Y-%m-%d %H:%M}")
    print(f"  to   {hi:%Y-%m-%d %H:%M}")
    print(f"  span {span} days, {len(df):,} rows")

    return {"min": lo, "max": hi, "days": span}
