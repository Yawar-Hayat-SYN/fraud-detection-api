"""Generate a versioned temporal train/validation split.

Run from the repo root:
    python -m ml.make_split

Writes data/splits/v1.json. That file is committed to git -- it is the one
data artifact that lives in the repo, because every metric you report is
meaningless unless someone can reproduce the exact rows you trained on.
"""

import json
from datetime import datetime, timezone

import polars as pl

from ml.config import (
    ID_COL,
    SPLIT_PATH,
    SPLIT_VERSION,
    TARGET_COL,
    TIME_COL,
    TRAIN_PARQUET,
)
from ml.features.time import dt_to_timestamp

TRAIN_FRACTION = 0.80


def main() -> None:
    if not TRAIN_PARQUET.exists():
        raise SystemExit(f"Missing {TRAIN_PARQUET}. Run ml/load_raw.py first.")

    # Only three columns are needed, so only three are read. On a 434-column
    # file this is the difference between ~20 MB and several GB.
    df = pl.scan_parquet(TRAIN_PARQUET).select(ID_COL, TIME_COL, TARGET_COL).collect()

    # Linear interpolation, so the cutoff can fall between two TransactionDT
    # values and have a fractional part. Rows are compared against that exact
    # value, not a rounded one: rounding would move rows sitting at
    # floor(cutoff) from train to val.
    cutoff = df[TIME_COL].quantile(TRAIN_FRACTION, interpolation="linear")

    is_train = pl.col(TIME_COL) < cutoff
    # filter keeps the file's row order, which the ID lists below depend on.
    train_rows = df.filter(is_train)
    val_rows = df.filter(~is_train)

    def summarise(t: pl.DataFrame) -> dict:
        n = t.height
        frauds = t[TARGET_COL].sum()
        return {
            "rows": n,
            "frauds": int(frauds),
            "fraud_rate": round(frauds / n, 5),
            "first_timestamp": dt_to_timestamp(t[TIME_COL].min()).isoformat(),
            "last_timestamp": dt_to_timestamp(t[TIME_COL].max()).isoformat(),
        }

    train_stats = summarise(train_rows)
    val_stats = summarise(val_rows)

    cutoff_dt = dt_to_timestamp(val_rows[TIME_COL].min())

    split = {
        "version": SPLIT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "temporal percentile cutoff on TransactionDT",
        "train_fraction": TRAIN_FRACTION,
        "cutoff_transaction_dt": int(cutoff),
        "cutoff_datetime": cutoff_dt.isoformat(),
        "train": train_stats,
        "val": val_stats,
        "train_ids": train_rows[ID_COL].to_list(),
        "val_ids": val_rows[ID_COL].to_list(),
    }

    out = SPLIT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(split, f, indent=2)

    size_mb = out.stat().st_size / 1024**2

    print(f"Cutoff: TransactionDT={int(cutoff):,} ({cutoff_dt:%Y-%m-%d %H:%M})")
    print()
    for label, s in (("Train", train_stats), ("Val", val_stats)):
        print(f"{label}:")
        print(f"  rows       {s['rows']:,}")
        print(f"  frauds     {s['frauds']:,}")
        print(f"  fraud rate {s['fraud_rate']:.2%}")
        print(f"  span       {s['first_timestamp'][:10]} to {s['last_timestamp'][:10]}")
        print()

    drift = val_stats["fraud_rate"] - train_stats["fraud_rate"]
    print(f"Fraud rate difference val - train: {drift:+.2%}")
    if abs(drift) > 0.01:
        print("Worth noting in the README -- the base rate shifted over time,")
        print("which is exactly the drift a random split would have hidden.")

    print(f"\nWrote {out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()