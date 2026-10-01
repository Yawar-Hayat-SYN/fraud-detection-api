"""Presence flags for blocks of columns that go null together.

Nulls in this dataset come in two kinds:

- Structural: the field does not exist for that product. W transactions come
  through a separate pipeline and never have identity fields; dist1 exists only
  for W. A flag for a structural null is a copy of ProductCD and adds nothing.
- Within-product: the field exists for the product but is missing on some
  rows. That variation carries real information.

So only blocks classified "mixed" by ml.analysis.null_blocks get a flag.
Blocks marked "uniform", "w_only" and "non_w_only" are skipped.

Run python -m ml.analysis.null_blocks first to generate the blocks file.
"""

import json
from pathlib import Path
from typing import TypeVar

import polars as pl

from ml import config
from ml.analysis.null_blocks import MIXED

Frame = TypeVar("Frame", pl.DataFrame, pl.LazyFrame)


def _read_blocks(blocks_path: Path | None) -> list[dict]:
    path = blocks_path if blocks_path is not None else config.NULL_BLOCKS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Null blocks file not found: {path}\n"
            f"Generate it from the repo root with:\n"
            f"    python -m ml.analysis.null_blocks"
        )
    with open(path) as f:
        return json.load(f)


def _flag_name(block_index: int) -> str:
    # Numbered by position in the blocks file, not among mixed blocks only, so
    # has_block_07 always refers to entry 7 of null_blocks.json.
    return f"has_block_{block_index:02d}"


def _flagged_blocks(blocks: list[dict]) -> dict[str, str]:
    """Map flag name to the column that represents its block.

    Every column in a block shares one null mask, so the first one stands in
    for the rest.
    """
    return {
        _flag_name(i): block["columns"][0]
        for i, block in enumerate(blocks)
        if block["kind"] == MIXED
    }


def get_flag_columns(blocks_path: Path | None = None) -> list[str]:
    """Names of the flags add_availability_flags will add, in order."""
    return [*_flagged_blocks(_read_blocks(blocks_path)), config.RARE_ID_FLAG]


def add_availability_flags(df: Frame, blocks_path: Path | None = None) -> Frame:
    """Add a boolean has_block_NN flag per mixed block, plus has_rare_id_block.

    Each flag is True where that block's data is present. blocks_path
    defaults to ml.config.NULL_BLOCKS_PATH.
    """
    flagged = _flagged_blocks(_read_blocks(blocks_path))
    return df.with_columns(
        *(pl.col(col).is_not_null().alias(name) for name, col in flagged.items()),
        pl.col(config.RARE_ID_COL).is_not_null().alias(config.RARE_ID_FLAG),
    )
