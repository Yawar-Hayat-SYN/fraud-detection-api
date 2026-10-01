"""The single entry point for reading transaction data.

Everything in ml/ that needs transaction rows goes through load_transactions.
Do not call pl.read_parquet / pd.read_parquet on the raw files directly:
this module is where the train/val split is applied, and bypassing it is how
validation rows leak into training.

Paths come from ml.config. They are looked up on the config module at call
time (config.TRAIN_PARQUET, not a from-import), so tests can monkeypatch them.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

import polars as pl

from ml import config

Split = Literal["train", "val"]
_SPLITS = ("train", "val")


def _check_split_name(split: str) -> None:
    if split not in _SPLITS:
        raise ValueError(f"split must be one of {_SPLITS} or None, got {split!r}")


@lru_cache(maxsize=4)
def _read_split_file(path: Path) -> dict:
    """Parse the split JSON once per path. It holds ~590k IDs, so re-reading
    it on every call adds up."""
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}\n"
            f"Generate it from the repo root with:\n"
            f"    python -m ml.make_split\n"
            f"(needs {config.TRAIN_PARQUET} to exist first)."
        )
    with open(path) as f:
        return json.load(f)


def load_split_ids(split: Split) -> list[int]:
    """Return the TransactionIDs belonging to the given split."""
    _check_split_name(split)
    # Copy, so a caller mutating the list cannot corrupt the cached file.
    return list(_read_split_file(config.SPLIT_PATH)[f"{split}_ids"])


def get_split_metadata() -> dict:
    """Return the split's cutoff and row counts, without the ID lists."""
    data = _read_split_file(config.SPLIT_PATH)
    return {
        "version": data["version"],
        "cutoff_transaction_dt": data["cutoff_transaction_dt"],
        "cutoff_datetime": data["cutoff_datetime"],
        "train_rows": data["train"]["rows"],
        "val_rows": data["val"]["rows"],
    }


def load_transactions(
    columns: list[str] | None = None,
    split: Split | None = None,
    lazy: bool = False,
) -> pl.DataFrame | pl.LazyFrame:
    """Read the training transactions, optionally projected and split.

    Args:
        columns: Columns to read. TransactionID is always included, because
            the split is applied by filtering on it. None reads every column.
        split: None for every row, or "train" / "val" for that side of the
            committed temporal split.
        lazy: Return the LazyFrame instead of collecting it.
    """
    if split is not None:
        _check_split_name(split)

    path = config.TRAIN_PARQUET
    if not path.exists():
        raise FileNotFoundError(
            f"Transaction data not found: {path}\n"
            f"Download the raw CSVs from Kaggle, then convert them to Parquet:\n"
            f"    kaggle competitions download -c {config.KAGGLE_COMPETITION} -p {config.RAW_DIR}\n"
            f"    unzip {config.RAW_DIR / config.KAGGLE_COMPETITION}.zip -d {config.RAW_DIR}\n"
            f"    python ml/load_raw.py"
        )

    lf = pl.scan_parquet(path)

    if columns is not None:
        wanted = [config.ID_COL] + [c for c in columns if c != config.ID_COL]
        lf = lf.select(wanted)

    if split is not None:
        lf = lf.filter(pl.col(config.ID_COL).is_in(load_split_ids(split)))

    return lf if lazy else lf.collect()
