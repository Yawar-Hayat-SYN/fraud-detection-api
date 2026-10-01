"""Group columns into blocks that go null on exactly the same rows.

Columns with identical null masks almost certainly came from the same upstream
source (the identity table, a single vendor feed, and so on). Each block is
then classified by how its null rate varies across ProductCD.

Run from anywhere:
    python -m ml.analysis.null_blocks

Writes ml.config.NULL_BLOCKS_PATH.

Columns are referenced by name throughout, never by position. The column count
changes as features are added, and positional indexing into a numpy array
breaks silently when it does.
"""

import json
from hashlib import blake2b

import polars as pl

from ml import config
from ml.data import load_transactions

UNIFORM = "uniform"
W_ONLY = "w_only"
NON_W_ONLY = "non_w_only"
MIXED = "mixed"
KINDS = (UNIFORM, W_ONLY, NON_W_ONLY, MIXED)

# Keys every entry in the blocks file has. Readers validate against this.
BLOCK_KEYS = ("columns", "n_columns", "null_rate", "kind", "null_rate_by_product")

REGENERATE_COMMAND = "python -m ml.analysis.null_blocks"


def null_mask_hashes(masks: pl.DataFrame) -> dict[str, str]:
    """Map each column name to a hash of its boolean null mask."""
    return {
        name: blake2b(masks[name].to_numpy().tobytes(), digest_size=16).hexdigest()
        for name in masks.columns
    }


def group_columns_by_null_pattern(masks: pl.DataFrame) -> list[list[str]]:
    """Group column names whose null masks are identical.

    Blocks and the columns inside them keep the frame's column order.
    """
    blocks: dict[str, list[str]] = {}
    for name, digest in null_mask_hashes(masks).items():
        blocks.setdefault(digest, []).append(name)
    return list(blocks.values())


def classify_block(rate_by_product: dict[str, float]) -> str:
    """Label a block from its null rate per ProductCD value.

    - uniform: every product < LOW, or every product > HIGH.
    - w_only: W < HIGH and every non-W product > HIGH.
    - non_w_only: W > HIGH and at least one non-W product < HIGH.
    - mixed: anything else.
    """
    low, high = config.NULL_RATE_LOW, config.NULL_RATE_HIGH
    rates = list(rate_by_product.values())

    if all(r < low for r in rates) or all(r > high for r in rates):
        return UNIFORM

    w_rate = rate_by_product.get(config.W_PRODUCT)
    if w_rate is None:
        return MIXED
    non_w = [r for p, r in rate_by_product.items() if p != config.W_PRODUCT]

    if w_rate < high and all(r > high for r in non_w):
        return W_ONLY
    if w_rate > high and any(r < high for r in non_w):
        return NON_W_ONLY
    return MIXED


def build_null_blocks(df: pl.DataFrame | pl.LazyFrame) -> list[dict]:
    """Find the null blocks in df and describe each one.

    df must contain ProductCD. Every column, ProductCD included, is assigned
    to exactly one block.
    """
    lf = df.lazy()
    masks = lf.select(pl.all().is_null()).collect()
    blocks = group_columns_by_null_pattern(masks)

    # Every column in a block has the same mask, so its first column stands in
    # for the whole block. Aggregates are aliased by block number so a block
    # whose representative is ProductCD itself does not clash with the group key.
    representatives = {f"block_{i}": cols[0] for i, cols in enumerate(blocks)}

    by_product = (
        lf.group_by(config.PRODUCT_COL)
        .agg(pl.col(col).is_null().mean().alias(key) for key, col in representatives.items())
        .sort(config.PRODUCT_COL)
        .collect()
    )
    overall = masks.select(
        pl.col(col).mean().alias(key) for key, col in representatives.items()
    ).row(0, named=True)

    products = [str(p) for p in by_product[config.PRODUCT_COL]]

    result = []
    for key, cols in zip(representatives, blocks):
        rate_by_product = dict(zip(products, by_product[key].to_list()))
        result.append(
            {
                "columns": cols,
                "n_columns": len(cols),
                "null_rate": overall[key],
                "kind": classify_block(rate_by_product),
                "null_rate_by_product": rate_by_product,
            }
        )
    return result


def main() -> None:
    blocks = build_null_blocks(load_transactions(lazy=True))

    out = config.NULL_BLOCKS_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(blocks, f, indent=2)

    n_columns = sum(b["n_columns"] for b in blocks)
    print(f"{n_columns} columns in {len(blocks)} null blocks")
    for kind in (UNIFORM, W_ONLY, NON_W_ONLY, MIXED):
        of_kind = [b for b in blocks if b["kind"] == kind]
        cols = sum(b["n_columns"] for b in of_kind)
        print(f"  {kind:<11} {len(of_kind):>3} blocks, {cols:>4} columns")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
