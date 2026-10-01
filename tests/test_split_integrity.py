"""Integrity checks on the committed split.

These guard the single assumption every metric in this project rests on:
that validation rows come strictly after training rows, and that no row is
duplicated or lost between the two.

A leak here does not raise an error anywhere. It just makes the model look
better than it is, silently, for the rest of the project. That is why this is
a test and not a print statement.

Run from the repo root:
    pytest tests/test_split_integrity.py -v
"""

import json

import polars as pl
import pytest

from ml.config import ID_COL, SPLIT_PATH, TIME_COL, TRAIN_PARQUET


@pytest.fixture(scope="module")
def split() -> dict:
    if not SPLIT_PATH.exists():
        pytest.skip(f"{SPLIT_PATH} not found -- run python -m ml.make_split")
    with open(SPLIT_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def dt_by_id() -> dict[int, int]:
    """Map TransactionID -> TransactionDT for every row in the source file.

    Module-scoped so the Parquet file is read once for the whole test module
    rather than once per test.
    """
    if not TRAIN_PARQUET.exists():
        pytest.skip(f"{TRAIN_PARQUET} not found -- run python -m ml.load_raw")
    df = pl.scan_parquet(TRAIN_PARQUET).select(ID_COL, TIME_COL).collect()
    return dict(zip(df[ID_COL].to_list(), df[TIME_COL].to_list()))


def test_train_ends_before_val_begins(split, dt_by_id):
    """No training row may occur at or after the first validation row."""
    latest_train = max(dt_by_id[i] for i in split["train_ids"])
    earliest_val = min(dt_by_id[i] for i in split["val_ids"])

    assert latest_train < earliest_val, (
        f"Temporal leak: training data runs to TransactionDT={latest_train:,} "
        f"but validation starts at {earliest_val:,}. The model would be "
        f"learning from the future it is being tested on."
    )


def test_no_overlap_between_sets(split):
    """A row in both sets is memorised, then scored as if it were unseen."""
    train_ids = set(split["train_ids"])
    val_ids = set(split["val_ids"])
    overlap = train_ids & val_ids

    assert not overlap, (
        f"{len(overlap):,} TransactionIDs appear in both train and val. "
        f"Examples: {sorted(overlap)[:5]}"
    )


def test_split_covers_every_row_exactly_once(split, dt_by_id):
    """Union of the two sets must equal the source, with nothing duplicated."""
    train_ids = split["train_ids"]
    val_ids = split["val_ids"]
    all_ids = set(dt_by_id)

    assert len(train_ids) + len(val_ids) == len(all_ids), (
        f"Row count mismatch: {len(train_ids):,} train + {len(val_ids):,} val "
        f"= {len(train_ids) + len(val_ids):,}, but the source has {len(all_ids):,}."
    )

    assert set(train_ids) | set(val_ids) == all_ids, "Split does not cover every source row"

    assert len(set(train_ids)) == len(train_ids), "Duplicate IDs within train_ids"
    assert len(set(val_ids)) == len(val_ids), "Duplicate IDs within val_ids"


def test_recorded_stats_match_actual_ids(split):
    """The row counts written into the JSON must match the lists beside them.

    Cheap to check, and catches the case where someone edits the summary by
    hand without regenerating the split.
    """
    assert split["train"]["rows"] == len(split["train_ids"])
    assert split["val"]["rows"] == len(split["val_ids"])
