"""Reconstruct an approximate card identity.

The dataset has no customer ID. D1 is "days since the card began being used",
so D1n = day - D1 is near-constant for one card, and card1 + addr1 + D1n
approximates a card identifier. Label purity within these groups is 97.4%,
against 79.5% for a shuffled baseline.

Null handling is the whole point of this module. If any component is missing
the UID is NULL, never a string. Casting to str and concatenating would turn a
null into the text "null", putting every incomplete row into one huge shared
bucket. Nothing would error; every UID-level aggregate would just be quietly
wrong. That bug has already been hit once on this project.
"""

from typing import TypeVar

import polars as pl

from ml.config import UID_COARSE_COL, UID_COARSE_COMPONENTS, UID_COL, UID_COMPONENTS

Frame = TypeVar("Frame", pl.DataFrame, pl.LazyFrame)

# Components are numeric, so this can never appear inside one and two
# different component tuples can never join to the same string.
_SEPARATOR = "_"


def _null_safe_key(components: list[str]) -> pl.Expr:
    """Join components into one string key, or null if any is missing.

    NaN counts as missing: in polars it is a float value, not a null, so a
    float column with NaN would otherwise produce a key containing "NaN".
    fill_nan is a no-op on non-float columns.
    """
    cols = [pl.col(c).fill_nan(None) for c in components]
    # Explicit guard, rather than relying on concat_str's default null
    # behaviour: if that default (or someone's edit) ever changes, this still
    # holds.
    any_missing = pl.any_horizontal([c.is_null() for c in cols])
    return (
        pl.when(any_missing)
        .then(pl.lit(None, dtype=pl.String))
        .otherwise(pl.concat_str([c.cast(pl.String) for c in cols], separator=_SEPARATOR))
    )


def add_uid(df: Frame) -> Frame:
    """Add "uid" from card1, addr1, D1n. Null if any component is null.

    D1n comes from ml.features.time.add_time_features, which must run first.
    """
    return df.with_columns(_null_safe_key(UID_COMPONENTS).alias(UID_COL))


def add_uid_coarse(df: Frame) -> Frame:
    """Add "uid_coarse" from card1 and addr1. Null if either is null.

    A fallback for rows where D1n is missing and the full UID can't be built.
    """
    return df.with_columns(_null_safe_key(UID_COARSE_COMPONENTS).alias(UID_COARSE_COL))


def add_uid_tier(df: Frame) -> Frame:
    """Add "uid_tier": "full", "coarse" or "none", the finest UID available.

    Needs the uid and uid_coarse columns, so run add_uid and add_uid_coarse
    first.
    """
    return df.with_columns(
        pl.when(pl.col(UID_COL).is_not_null())
        .then(pl.lit("full"))
        .when(pl.col(UID_COARSE_COL).is_not_null())
        .then(pl.lit("coarse"))
        .otherwise(pl.lit("none"))
        .alias("uid_tier")
    )
