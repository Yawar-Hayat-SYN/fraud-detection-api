"""Tests for ml.analysis.null_blocks."""

import json

import polars as pl
import pytest

from ml import config
from ml.analysis.null_blocks import (
    BLOCK_KEYS,
    KINDS,
    MIXED,
    NON_W_ONLY,
    UNIFORM,
    W_ONLY,
    build_null_blocks,
    classify_block,
)
from ml.data import load_transactions


def test_shared_null_pattern_forms_one_block():
    """a and b go null on the same rows; c never does.

    c lands in the same block as ProductCD, which also has no nulls, so the
    frame splits into exactly two blocks.
    """
    df = pl.DataFrame(
        {
            config.PRODUCT_COL: ["W", "W", "C", "H"],
            "a": [1.0, None, 3.0, None],
            "b": ["x", None, "y", None],
            "c": [1, 2, 3, 4],
        }
    )

    blocks = build_null_blocks(df)

    assert len(blocks) == 2
    assert sorted(sorted(b["columns"]) for b in blocks) == [
        ["ProductCD", "c"],
        ["a", "b"],
    ]
    ab = next(b for b in blocks if "a" in b["columns"])
    assert ab["n_columns"] == 2
    assert ab["null_rate"] == 0.5
    assert ab["null_rate_by_product"] == {"C": 0.0, "H": 1.0, "W": 0.5}


def test_block_order_does_not_depend_on_column_position():
    """Reordering columns changes nothing but block order."""
    df = pl.DataFrame(
        {config.PRODUCT_COL: ["W", "C"], "a": [None, 1], "b": [None, 2], "c": [1, None]}
    )

    forward = build_null_blocks(df)
    reversed_ = build_null_blocks(df.select(reversed(df.columns)))

    def as_sets(blocks):
        return sorted((sorted(b["columns"]), b["kind"]) for b in blocks)

    assert as_sets(forward) == as_sets(reversed_)


@pytest.mark.parametrize(
    ("rates", "expected"),
    [
        ({"W": 0.0, "C": 0.005, "H": 0.0}, UNIFORM),
        ({"W": 1.0, "C": 0.995, "H": 1.0}, UNIFORM),
        ({"W": 0.2, "C": 1.0, "H": 0.999}, W_ONLY),
        ({"W": 1.0, "C": 0.3, "H": 1.0}, NON_W_ONLY),
        ({"W": 0.5, "C": 0.5, "H": 0.0}, MIXED),
    ],
    ids=["uniform-present", "uniform-blank", "w_only", "non_w_only", "mixed"],
)
def test_classify_block(rates, expected):
    assert classify_block(rates) == expected


@pytest.fixture(scope="module")
def real_blocks() -> list[dict]:
    if not config.TRAIN_PARQUET.exists():
        pytest.skip(f"{config.TRAIN_PARQUET} not found -- run python ml/load_raw.py")
    return build_null_blocks(load_transactions(lazy=True))


def block_containing(blocks: list[dict], column: str) -> dict:
    return next(b for b in blocks if column in b["columns"])


@pytest.mark.slow
def test_real_data_has_about_70_blocks(real_blocks):
    assert 60 <= len(real_blocks) <= 80, f"got {len(real_blocks)} blocks"


@pytest.mark.slow
def test_real_transaction_id_block_is_uniform(real_blocks):
    assert block_containing(real_blocks, config.ID_COL)["kind"] == UNIFORM


@pytest.mark.slow
def test_real_device_info_block_is_non_w_only(real_blocks):
    assert block_containing(real_blocks, "DeviceInfo")["kind"] == NON_W_ONLY


# --- The committed blocks file --------------------------------------------------

# Spelled out here rather than imported, so changing the schema in code
# without updating this spec fails a test.
REQUIRED_KEYS = {"columns", "n_columns", "null_rate", "kind", "null_rate_by_product"}


def test_build_null_blocks_writes_the_documented_keys():
    df = pl.DataFrame({config.PRODUCT_COL: ["W", "C"], "a": [None, 1]})

    for entry in build_null_blocks(df):
        assert set(entry) == REQUIRED_KEYS == set(BLOCK_KEYS)


def test_committed_null_blocks_file_exists_and_is_well_formed():
    """data/null_blocks.json is committed, so this must pass on a fresh clone.

    ml.features.availability reads it at runtime. Not skipped when missing:
    a missing file is exactly the failure this test is here to catch.
    """
    path = config.NULL_BLOCKS_PATH
    assert path.exists(), f"{path} is missing. Run: python -m ml.analysis.null_blocks, then commit it."

    blocks = json.loads(path.read_text())

    assert isinstance(blocks, list) and blocks, "expected a non-empty list of blocks"
    for i, entry in enumerate(blocks):
        missing = REQUIRED_KEYS - set(entry)
        assert not missing, f"block {i} is missing {sorted(missing)}"
        assert entry["kind"] in KINDS, f"block {i} has unknown kind {entry['kind']!r}"
        assert entry["n_columns"] == len(entry["columns"]), f"block {i}: n_columns != len(columns)"
