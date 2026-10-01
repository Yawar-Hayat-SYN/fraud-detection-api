"""Compose the feature modules into one reproducible feature build.

Run from anywhere:
    python -m ml.pipeline --split train
    python -m ml.pipeline --split val

Writes ml.config.PROCESSED_DIR / "{split}_features.parquet".

Two design decisions matter here:

1. Features are built over ALL labelled rows, and only then filtered to the
   requested split. A val row's entity aggregates need that entity's history
   from train, exactly as production scoring would have it; building val on
   its own would make every entity look brand new at the split boundary.
   This does not leak: every aggregate looks strictly backwards in time, and
   every val row is later than every train row (tests/test_split_integrity.py).
   So train rows come out the same with or without val rows present, and
   tests/test_pipeline.py checks exactly that.

2. Only the raw columns the stages actually read are loaded. Each
   add_entity_aggregates call sorts the frame; with all 434 raw columns
   attached, every sort would copy a ~2 GB frame, which is how this project
   has already been OOM-killed. The output carries TransactionID, so raw
   columns can be joined back lazily when a model needs them.

The whole build is one lazy plan, collected once. Stages therefore cost almost
nothing when they are added; the real work happens at the single collect, and
that is where the time is logged. Every stage only adds columns, so the row
count cannot change until the final split filter.
"""

import argparse
import logging
import time
from collections.abc import Callable
from pathlib import Path

import polars as pl

from ml import config
from ml.data import features_path, get_split_metadata, load_split_ids, load_transactions
from ml.features.aggregates import add_entity_aggregates
from ml.features.availability import add_availability_flags, get_source_columns
from ml.features.time import add_time_features
from ml.features.uid import add_uid, add_uid_coarse, add_uid_tier

# Fixed name, not __name__, which is "__main__" under python -m.
logger = logging.getLogger("ml.pipeline")

SPLITS = ("train", "val")

Stage = tuple[str, Callable[[pl.LazyFrame], pl.LazyFrame]]


def stages() -> list[Stage]:
    """The feature stages, in the order they must run."""
    return [
        ("time_features", add_time_features),
        ("uid", add_uid),
        ("uid_coarse", add_uid_coarse),
        ("uid_tier", add_uid_tier),
        ("availability_flags", add_availability_flags),
        *(
            (f"aggregates_{entity}", lambda lf, entity=entity: add_entity_aggregates(lf, entity))
            for entity in config.AGG_ENTITIES
        ),
    ]


def source_columns() -> list[str]:
    """Raw columns the stages read, plus ones carried through to the output."""
    wanted = [
        config.TARGET_COL,
        config.PRODUCT_COL,
        config.TIME_COL,
        config.AMT_COL,
        config.D1_COL,  # D1n, a uid component, is derived from it
        *config.UID_COARSE_COMPONENTS,
        *get_source_columns(),
        *config.CATEGORICAL_RAW_COLS,  # model inputs, carried through as-is
    ]
    return list(dict.fromkeys(wanted))  # de-duplicate, keep order


def build_features(split: str) -> pl.LazyFrame:
    """The lazy plan for one split's features. Nothing is computed yet."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    lf = load_transactions(columns=source_columns(), lazy=True)
    n_cols = len(lf.collect_schema())
    logger.info("stage=load columns=%d", n_cols)

    for name, stage in stages():
        start = time.perf_counter()
        lf = stage(lf)
        # collect_schema resolves the plan's schema without touching data.
        added = len(lf.collect_schema()) - n_cols
        n_cols += added
        logger.info(
            "stage=%s columns=%d added=%d plan_seconds=%.3f",
            name, n_cols, added, time.perf_counter() - start,
        )

    return lf.filter(pl.col(config.ID_COL).is_in(load_split_ids(split)))


def run(split: str) -> Path:
    """Build one split's features, collect once, write to Parquet."""
    lf = build_features(split)

    start = time.perf_counter()
    df = lf.collect()
    logger.info(
        "stage=collect split=%s rows=%d columns=%d seconds=%.1f",
        split, df.height, df.width, time.perf_counter() - start,
    )

    expected = get_split_metadata()[f"{split}_rows"]
    if df.height != expected:
        raise RuntimeError(f"{split} has {df.height:,} rows, but {config.SPLIT_PATH} says {expected:,}")

    out = features_path(split)
    out.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    df.write_parquet(out)
    logger.info("stage=write path=%s seconds=%.1f", out, time.perf_counter() - start)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", required=True, choices=SPLITS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    run(args.split)


if __name__ == "__main__":
    main()
