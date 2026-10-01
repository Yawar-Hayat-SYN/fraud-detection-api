"""Tests for ml.pipeline.

The fast tests run the full pipeline on a small synthetic dataset, with every
path in ml.config pointed at a temporary directory. The slow tests run it on
the real data.
"""

import json
import logging

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml import config
from ml.data import get_split_metadata
from ml.pipeline import build_features, run

HOUR = 3600


def write_dataset(root, train_only: bool = False) -> None:
    """Write a synthetic train.parquet, split file and null-blocks file.

    20 cards, ~400 transactions over 6 days, timestamps every few minutes so
    the 1h/24h windows and std (needs 2 prior amounts) all get values.
    """
    rng = np.random.default_rng(0)
    n = 400
    dt = np.sort(rng.integers(86_400, 86_400 + 6 * 86_400, n))
    card = rng.integers(0, 20, n)
    first_day = {c: int(rng.integers(0, 30)) for c in range(20)}
    day = dt // 86_400
    df = pl.DataFrame(
        {
            config.ID_COL: np.arange(n) + 1_000,
            config.TARGET_COL: (rng.random(n) < 0.1).astype(np.int64),
            config.PRODUCT_COL: rng.choice(["W", "C", "H"], n),
            config.TIME_COL: dt,
            config.AMT_COL: rng.gamma(2, 50, n).round(2),
            # D1 = days since the card's first use, so D1n is constant per card.
            config.D1_COL: [float(d - 1 + first_day[c]) for d, c in zip(day, card)],
            "card1": card + 10_000,
            "addr1": [None if c == 0 else float(100 + c % 3) for c in card],
            "M4": [None if x < 0.3 else "M0" for x in rng.random(n)],
            config.RARE_ID_COL: [None if x < 0.9 else 1.0 for x in rng.random(n)],
            "V1": rng.random(n),  # a raw column no stage needs
        }
    )
    cutoff = int(np.quantile(dt, 0.8))
    is_train = pl.col(config.TIME_COL) < cutoff
    train_ids = df.filter(is_train)[config.ID_COL].to_list()
    val_ids = [] if train_only else df.filter(~is_train)[config.ID_COL].to_list()
    if train_only:
        df = df.filter(is_train)

    (root / "raw").mkdir(parents=True, exist_ok=True)
    df.write_parquet(root / "raw" / "train.parquet")
    (root / "split.json").write_text(
        json.dumps(
            {
                "version": "test",
                "cutoff_transaction_dt": cutoff,
                "cutoff_datetime": "n/a",
                "train": {"rows": len(train_ids)},
                "val": {"rows": len(val_ids)},
                "train_ids": train_ids,
                "val_ids": val_ids,
            }
        )
    )
    block = {"n_columns": 1, "null_rate": 0.3, "null_rate_by_product": {}}
    (root / "null_blocks.json").write_text(
        json.dumps(
            [
                {**block, "columns": [config.ID_COL], "kind": "uniform"},
                {**block, "columns": ["M4"], "kind": "mixed"},
                {**block, "columns": [config.RARE_ID_COL], "kind": "mixed"},
            ]
        )
    )


def point_config_at(monkeypatch, root) -> None:
    monkeypatch.setattr(config, "TRAIN_PARQUET", root / "raw" / "train.parquet")
    monkeypatch.setattr(config, "SPLIT_PATH", root / "split.json")
    monkeypatch.setattr(config, "NULL_BLOCKS_PATH", root / "null_blocks.json")
    monkeypatch.setattr(config, "PROCESSED_DIR", root / "processed")


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    write_dataset(tmp_path)
    point_config_at(monkeypatch, tmp_path)
    return tmp_path


@pytest.fixture
def built(synthetic) -> dict[str, pl.DataFrame]:
    return {split: build_features(split).collect() for split in ("train", "val")}


# --- The three required properties, on synthetic data ---------------------------


def test_train_and_val_have_identical_schemas(built):
    assert built["train"].schema == built["val"].schema


def test_no_output_column_is_entirely_null(built):
    for split, df in built.items():
        all_null = [c for c in df.columns if df[c].null_count() == df.height]
        assert not all_null, f"{split}: entirely null columns {all_null}"


def test_row_counts_match_split_file(built):
    meta = get_split_metadata()
    for split, df in built.items():
        assert df.height == meta[f"{split}_rows"]


# --- Design properties -----------------------------------------------------------


def test_split_ids_match_exactly(built):
    split = json.loads(config.SPLIT_PATH.read_text())
    for name, df in built.items():
        assert df[config.ID_COL].to_list() == split[f"{name}_ids"]


def test_val_rows_see_train_history(built):
    """A card's first val transaction must see its train transactions."""
    val = built["val"]
    first_val_per_card = val.group_by("card1", maintain_order=True).first()

    assert (first_val_per_card["card1_txn_count_prior"] > 0).all()


def test_train_features_unaffected_by_val_rows(built, tmp_path, monkeypatch):
    """Building over all rows must give train the same features as building on
    train alone. If val rows changed train features, the pipeline would leak."""
    alone = tmp_path / "train_only"
    write_dataset(alone, train_only=True)
    point_config_at(monkeypatch, alone)

    assert_frame_equal(build_features("train").collect(), built["train"])


def test_only_needed_raw_columns_are_loaded(built):
    assert "V1" not in built["train"].columns


def test_expected_feature_columns_present(built):
    cols = set(built["train"].columns)
    for entity in config.AGG_ENTITIES:
        assert f"{entity}_txn_count_prior" in cols
        assert f"{entity}_txn_count_7d" in cols
    assert {"day", "hour", "D1n", "uid", "uid_coarse", "uid_tier"} <= cols
    assert {"has_block_01", "has_block_02", config.RARE_ID_FLAG} <= cols


def test_run_writes_parquet_and_logs_stages(synthetic, caplog):
    with caplog.at_level(logging.INFO, logger="ml.pipeline"):
        path = run("val")

    assert path == config.PROCESSED_DIR / "val_features.parquet"
    assert_frame_equal(pl.read_parquet(path), build_features("val").collect())

    logged = caplog.text
    for stage in ("time_features", "uid_tier", "availability_flags", "aggregates_card1", "collect"):
        assert f"stage={stage}" in logged
    assert f"rows={get_split_metadata()['val_rows']}" in logged


def test_invalid_split_raises(synthetic):
    with pytest.raises(ValueError, match="split must be one of"):
        build_features("test")


# --- Real data -------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_built() -> dict[str, pl.DataFrame]:
    for path in (config.TRAIN_PARQUET, config.SPLIT_PATH, config.NULL_BLOCKS_PATH):
        if not path.exists():
            pytest.skip(f"{path} not found")
    return {split: build_features(split).collect() for split in ("train", "val")}


@pytest.mark.slow
def test_real_train_and_val_have_identical_schemas(real_built):
    assert real_built["train"].schema == real_built["val"].schema


@pytest.mark.slow
def test_real_no_output_column_is_entirely_null(real_built):
    for split, df in real_built.items():
        all_null = [c for c in df.columns if df[c].null_count() == df.height]
        assert not all_null, f"{split}: entirely null columns {all_null}"


@pytest.mark.slow
def test_real_row_counts_match_split_file(real_built):
    meta = get_split_metadata()
    assert real_built["train"].height == meta["train_rows"]
    assert real_built["val"].height == meta["val_rows"]
