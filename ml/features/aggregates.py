"""Point-in-time entity aggregates.

For each row, every statistic is computed over that entity's PRIOR
transactions only: those with a TransactionDT strictly before the row's own.
At scoring time in production nothing is known about transactions that have
not happened yet, so a feature that peeks at them is a leak. Leaks make
validation scores look better, not worse, so nothing downstream will flag one.
tests/test_aggregates.py is the guard, above all the test that appends future
rows and checks that past rows don't change.

How it stays point-in-time:

- Only running (cumulative or rolling) operations within each entity, in
  time order. Never a group_by().agg() joined back to the frame: that
  aggregates over the whole group, future included.
- Running totals are shifted by one row, so a row's own transaction is never
  in its own history.
- Rows tied on TransactionDT within an entity happened "at the same time", so
  they are excluded from each other's history. Every row in a tie takes the
  value from the first row of the tie, whose shifted running total covers
  exactly the rows strictly before that timestamp. The other rows forward-fill
  it. Forward-fill only looks backwards.
- Window counts use rolling_sum_by with closed="left", i.e. [t - w, t): the
  current timestamp, ties included, is outside the window.

Semantics:

- No history is not a history of zero. An entity's first transaction gets
  txn_count_prior = 0 and null mean/std/max/seconds_since_prev.
- Amount statistics are over prior rows with a non-null amount. std is the
  sample std (ddof=1), null until there are two prior amounts.
- Rows with a null entity get null for every aggregate. They are not pooled
  into one shared "null" entity, and they never enter anyone else's history.
- Window counts include a transaction exactly w seconds earlier.
- Output rows keep the input's order.
"""

from typing import TypeVar

import polars as pl

from ml.config import AGG_WINDOWS, AMT_COL, TIME_COL

Frame = TypeVar("Frame", pl.DataFrame, pl.LazyFrame)

_ROW_NR = "__agg_row_nr"


def _prior(value: pl.Expr, entity: str, fill: float | int | None = None) -> pl.Expr:
    """value's running total as of the last transaction strictly before this
    row's TransactionDT, within entity.

    value must be a running (inclusive) total over rows sorted by entity, time.
    Shifting by one gives "everything before this row"; taking that only on
    the first row of each tie and forward-filling gives every tied row the
    same, tie-free history.

    fill replaces the null that the shift leaves on an entity's first row.
    Without it, values that must stay null (max, previous time) stay null
    until the entity has a real value, and once non-null they never go back.
    """
    t = pl.col(TIME_COL)
    is_new_time = (t != t.shift(1)).fill_null(True)
    before = value.shift(1)
    if fill is not None:
        before = before.fill_null(fill)
    return pl.when(is_new_time).then(before).forward_fill().over(entity)


def add_entity_aggregates(
    df: Frame,
    entity_col: str,
    windows: dict[str, int] | None = None,
) -> Frame:
    """Add point-in-time aggregates of entity_col's prior transactions.

    Adds, prefixed with "{entity_col}_":
        txn_count_prior     number of prior transactions
        amt_mean_prior      mean prior TransactionAmt
        amt_std_prior       sample std of prior TransactionAmt
        amt_max_prior       max prior TransactionAmt
        seconds_since_prev  seconds since the previous transaction
        txn_count_{name}    prior transactions in the last `seconds`, for
                            each name -> seconds in windows

    windows defaults to ml.config.AGG_WINDOWS (1h, 24h, 7d). Accepts a
    DataFrame or LazyFrame and returns the same type, rows in input order.
    """
    windows = AGG_WINDOWS if windows is None else windows
    for name, seconds in windows.items():
        if seconds <= 0:
            raise ValueError(f"window {name!r} must be a positive number of seconds, got {seconds}")

    e = pl.col(entity_col)
    t = pl.col(TIME_COL)
    amt = pl.col(AMT_COL)
    p = f"{entity_col}_"

    # Running totals, inclusive of the current row, in (entity, time) order.
    run_count = t.cum_count().cast(pl.Int64)
    run_n_amt = amt.is_not_null().cast(pl.Int64).cum_sum()
    run_sum = amt.fill_null(0).cum_sum()
    run_sumsq = amt.fill_null(0).pow(2).cum_sum()
    # cum_max is null on null-amount rows; forward-fill carries the max over.
    run_max = amt.cum_max().forward_fill()

    count = _prior(run_count, entity_col, fill=0)
    n_amt = _prior(run_n_amt, entity_col, fill=0)
    total = _prior(run_sum, entity_col, fill=0)
    sumsq = _prior(run_sumsq, entity_col, fill=0)
    max_ = _prior(run_max, entity_col)
    prev_time = _prior(t, entity_col)

    mean = pl.when(n_amt > 0).then(total / n_amt)
    # Sample variance from running sums. Clipped at 0 because floating-point
    # cancellation can leave a tiny negative where the true value is 0.
    var = pl.when(n_amt > 1).then(((sumsq - total.pow(2) / n_amt) / (n_amt - 1)).clip(lower_bound=0))

    ones = t.is_not_null().cast(pl.Int64)
    window_counts = {
        f"{p}txn_count_{name}": ones.rolling_sum_by(
            t, window_size=f"{seconds}i", closed="left", min_samples=0
        ).over(entity_col)
        for name, seconds in windows.items()
    }

    features = {
        f"{p}txn_count_prior": count,
        f"{p}amt_mean_prior": mean,
        f"{p}amt_std_prior": var.sqrt(),
        f"{p}amt_max_prior": max_,
        f"{p}seconds_since_prev": t - prev_time,
        **window_counts,
    }

    return (
        df.with_row_index(_ROW_NR)
        .sort([entity_col, TIME_COL], maintain_order=True)
        .with_columns(
            # .over() puts all null-entity rows in one group; blank them out
            # rather than hand them aggregates of each other.
            pl.when(e.is_not_null()).then(expr).alias(name)
            for name, expr in features.items()
        )
        .sort(_ROW_NR)
        .drop(_ROW_NR)
    )
