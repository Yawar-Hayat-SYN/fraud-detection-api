"""Tests for ml.data, the single entry point for reading transactions.

Tests marked slow read the real training Parquet and skip if it is absent.
Skip them with:
    pytest -m "not slow"
"""

import polars as pl
import pytest

from ml import config
from ml.data import load_transactions

requires_parquet = pytest.mark.skipif(
    not config.TRAIN_PARQUET.exists(),
    reason=f"{config.TRAIN_PARQUET} not found -- run python ml/load_raw.py",
)


@pytest.fixture(scope="module")
def split_ids() -> dict[str, set[int]]:
    """TransactionIDs of each split, as actually returned by load_transactions."""
    return {
        split: set(load_transactions(columns=[config.ID_COL], split=split)[config.ID_COL])
        for split in ("train", "val")
    }


@pytest.mark.slow
@requires_parquet
def test_column_projection_adds_id():
    df = load_transactions(columns=["card1"])

    assert set(df.columns) == {"card1", config.ID_COL}


@pytest.mark.slow
@requires_parquet
def test_train_and_val_are_disjoint(split_ids):
    overlap = split_ids["train"] & split_ids["val"]

    assert not overlap, f"{len(overlap):,} IDs in both train and val, e.g. {sorted(overlap)[:5]}"


@pytest.mark.slow
@requires_parquet
def test_train_and_val_cover_full_set(split_ids):
    full_rows = load_transactions(columns=[config.ID_COL], lazy=True).select(pl.len()).collect().item()

    assert len(split_ids["train"]) + len(split_ids["val"]) == full_rows


def test_missing_parquet_raises_helpful_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRAIN_PARQUET", tmp_path / "train.parquet")

    with pytest.raises(FileNotFoundError) as exc:
        load_transactions()

    message = str(exc.value)
    assert str(tmp_path / "train.parquet") in message
    assert f"kaggle competitions download -c {config.KAGGLE_COMPETITION}" in message


def test_missing_split_file_names_generator(tmp_path, monkeypatch):
    pl.DataFrame({config.ID_COL: [1, 2]}).write_parquet(tmp_path / "train.parquet")
    monkeypatch.setattr(config, "TRAIN_PARQUET", tmp_path / "train.parquet")
    monkeypatch.setattr(config, "SPLIT_PATH", tmp_path / "missing.json")

    with pytest.raises(FileNotFoundError, match="python -m ml.make_split"):
        load_transactions(split="train")
