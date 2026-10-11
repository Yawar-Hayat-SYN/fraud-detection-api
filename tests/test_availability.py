"""Tests for ml.features.availability."""

import json

import polars as pl
import pytest

from ml import config
from ml.features.availability import add_availability_flags, get_flag_columns


def block(columns: list[str], kind: str) -> dict:
    """A blocks-file entry. Rates are placeholders; only columns and kind are read."""
    return {
        "name": f"block_{min(columns)}",
        "columns": sorted(columns),
        "n_columns": len(columns),
        "null_rate": 0.5,
        "kind": kind,
        "null_rate_by_product": {"W": 0.5, "C": 0.5},
    }


# One block of each kind, plus a second mixed block. Flag names come from each
# block's name (block_ + first column), never from its position in this list.
BLOCKS = [
    block(["TransactionID", "ProductCD"], "uniform"),  # 00
    block(["dist1"], "w_only"),  # 01
    block(["DeviceInfo", "id_30"], "non_w_only"),  # 02
    block(["M4", "M5"], "mixed"),  # 03
    block([config.RARE_ID_COL], "non_w_only"),  # 04
    block(["D2"], "mixed"),  # 05
]


@pytest.fixture
def blocks_path(tmp_path):
    path = tmp_path / "null_blocks.json"
    path.write_text(json.dumps(BLOCKS))
    return path


@pytest.fixture
def frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "ProductCD": ["W", "W", "C", "H"],
            "dist1": [5.0, 6.0, None, None],
            "DeviceInfo": [None, None, "ios", "android"],
            "id_30": [None, None, "iOS 11", "Android 7"],
            "M4": ["M0", None, None, "M2"],
            "M5": ["T", None, None, "F"],
            config.RARE_ID_COL: [None, None, 252.0, None],
            "D2": [None, 3.0, 4.0, None],
        }
    )


def block_flags(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("has_block_")]


def test_no_flag_for_structural_blocks(frame, blocks_path):
    out = add_availability_flags(frame, blocks_path)

    for skipped in ("has_block_ProductCD", "has_block_dist1", "has_block_DeviceInfo", "has_block_id_21"):
        assert skipped not in out.columns


def test_one_flag_per_mixed_block(frame, blocks_path):
    out = add_availability_flags(frame, blocks_path)

    n_mixed = sum(b["kind"] == "mixed" for b in BLOCKS)
    assert len(block_flags(out)) == n_mixed
    assert block_flags(out) == ["has_block_D2", "has_block_M4"]


def test_flag_marks_presence_of_block(frame, blocks_path):
    out = add_availability_flags(frame, blocks_path)

    assert out["has_block_M4"].to_list() == [True, False, False, True]
    assert out["has_block_D2"].to_list() == [False, True, True, False]


def test_rare_id_flag_matches_id_21_presence(frame, blocks_path):
    out = add_availability_flags(frame, blocks_path)

    assert out[config.RARE_ID_FLAG].to_list() == frame[config.RARE_ID_COL].is_not_null().to_list()
    assert out[config.RARE_ID_FLAG].to_list() == [False, False, True, False]


def test_get_flag_columns_matches_transform(frame, blocks_path):
    out = add_availability_flags(frame, blocks_path)
    added = [c for c in out.columns if c not in frame.columns]

    assert get_flag_columns(blocks_path) == added


def test_lazy_and_eager_agree_and_input_unchanged(frame, blocks_path):
    before = frame.columns.copy()

    eager = add_availability_flags(frame, blocks_path)
    lazy = add_availability_flags(frame.lazy(), blocks_path)

    assert frame.columns == before
    assert isinstance(lazy, pl.LazyFrame)
    assert eager.equals(lazy.collect())


def test_missing_blocks_file_names_generator(frame, tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m ml.analysis.null_blocks"):
        add_availability_flags(frame, tmp_path / "missing.json")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "not valid JSON"),
        ("{not json", "not valid JSON"),
        ("[]", "non-empty list"),
        ('{"columns": ["a"]}', "non-empty list"),
        (json.dumps([{"columns": ["D2"], "kind": "mixed"}]), "missing keys"),
        (json.dumps([block(["D2"], "mostly")]), "unknown kind"),
        (json.dumps([{**block(["D2", "D3"], "mixed"), "name": "block_03"}]), "named 'block_03'"),
    ],
    ids=["empty-file", "truncated", "empty-list", "not-a-list", "old-format", "bad-kind", "positional-name"],
)
def test_malformed_blocks_file_raises_instead_of_giving_no_flags(frame, tmp_path, content, message):
    path = tmp_path / "null_blocks.json"
    path.write_text(content)

    for call in (lambda: add_availability_flags(frame, path), lambda: get_flag_columns(path)):
        with pytest.raises(ValueError, match=message) as exc:
            call()
        assert "python -m ml.analysis.null_blocks" in str(exc.value)


def test_missing_blocks_file_raises_from_get_flag_columns(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m ml.analysis.null_blocks"):
        get_flag_columns(tmp_path / "missing.json")


def test_flags_do_not_depend_on_column_order(tmp_path):
    """End to end: blocks generated from the same data with columns in a
    different order give the same flag names, each reading the same column.
    Positional has_block_NN names failed this."""
    from ml.analysis.null_blocks import build_null_blocks
    from ml.features.availability import get_source_columns

    df = pl.DataFrame(
        {
            config.PRODUCT_COL: ["W", "W", "C", "H"],
            "card2": [1.0, None, 3.0, 4.0],
            "V217": [None, 1.0, None, 2.0],
            "id_01": [None, None, 1.0, 1.0],
            "M4": [1, None, None, 2],
            "M5": [1, None, None, 2],
        }
    )
    mappings = []
    for i, cols in enumerate((df.columns, list(reversed(df.columns)))):
        path = tmp_path / f"blocks_{i}.json"
        path.write_text(json.dumps(build_null_blocks(df.select(cols))))
        mappings.append(dict(zip(get_flag_columns(path), get_source_columns(path))))

    assert mappings[0] == mappings[1]
    assert mappings[0]["has_block_M4"] == "M4"
